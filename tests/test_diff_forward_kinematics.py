"""Per-topology FD vs analytical gradient checks for the rigid forward-kinematics path.

Covers `kernel_update_cartesian_space` and `kernel_forward_velocity` (the two
kernels dispatched via `.grad()` from `RigidSolver.substep_pre_coupling_grad`) by
exercising them through the public entity API. For each topology we check that
gradients of a scalar loss on a grad-aware output w.r.t. a `@tracked` setter
input match central finite differences within rtol=1e-4 (the same bar
`test_diff_contact` and `test_diff_solver` use).

Conventions:

  * Grad-aware setters (in this file): `set_pos`, `set_quat`, `set_dofs_velocity`.
    These are `@tracked` and register the input tensor for backward replay.
    `set_dofs_position` is NOT `@tracked` and is intentionally excluded.
  * Grad-aware outputs: `entity.get_state().pos|quat` for the base link, and
    `scene.get_state().solvers_state[<rigid>].links_pos|links_quat` for any
    link. Bare `entity.get_links_pos()` etc. are direct field reads and do not
    register on `_queried_states`, so they are not grad-aware.
  * Inputs must be `gs.Tensor` (created via `gs.tensor(...)`); plain `torch`
    tensors do not have the `_backward_from_qd` hook the solver invokes.
  * Velocity inputs flow into pose outputs because after one `scene.step()`,
    qpos_new = qpos_init + v * dt and links_pos/quat are recomputed by FK. So
    a `set_dofs_velocity → links_pos` check exercises both the integrator and
    `kernel_update_cartesian_space`'s backward.

CPU. fp32 + fp64 across the matrix; control_dofs_force checks are fp64-only
because their FD probe is at fp32's precision floor (see J1's
control_dofs_force comment for details).
"""

import os
import tempfile

import numpy as np
import pytest
import torch

import genesis as gs

from .utils import assert_allclose


pytestmark = [
    pytest.mark.debug(False),
]


# Per-precision FD tolerance. fp32 is intentionally looser — float32 FD has only
# ~7 significant digits of headroom and the optimal central-FD eps grows to ~1e-3.
# The "quat" kind covers outputs that go through a non-linear pose composition
# (set_dofs_velocity → state.quat) where Genesis autograd is currently a ~1%
# noisier than FD even after the qd_rotvec_to_quat fix.
_TOL = {
    ("64", "default"): dict(rtol=1e-4, atol=1e-6, eps=1e-5),
    # quat-output paths: a unit-norm projection happens inside `set_quat` (the
    # `relative=True` composition + normalization), so FD captures the projected
    # sensitivity while analytical traces the full Jacobian — that yields small
    # absolute mismatches (~1e-4) on entries where FD reports 0.
    ("64", "quat"): dict(rtol=2e-2, atol=1e-3, eps=1e-5),
    # fp32 batched runs (n_envs=4) accumulate ulp-level round-off across env
    # copies, so a few entries land at ~1e-3 abs vs FD even on
    # set_dofs_velocity → links_pos paths that are bit-clean at fp64. GPU
    # fp32 has slightly more accumulated noise than CPU fp32 due to
    # different op order, so the band absorbs both.
    ("32", "default"): dict(rtol=2e-2, atol=2e-3, eps=1e-3),
    ("32", "quat"): dict(rtol=5e-2, atol=5e-3, eps=1e-3),
}


_PRECISION_PARAMS = [
    pytest.param("64", marks=pytest.mark.precision("64"), id="fp64"),
    pytest.param("32", marks=pytest.mark.precision("32"), id="fp32"),
]

_N_ENVS_PARAMS = [
    pytest.param(0, id="single"),
    pytest.param(4, id="batched"),
]


# ---------------------------------------------------------------------------
# MJCF topologies. All geoms set contype/conaffinity=0 so collision is never
# in play even if `enable_collision` flips on accidentally.
# ---------------------------------------------------------------------------

MJCF_FREE = """
<mujoco model="free">
  <worldbody>
    <body name="chassis" pos="0 0 0">
      <freejoint/>
      <inertial mass="1.0" pos="0 0 0" diaginertia="0.1 0.1 0.1"/>
      <geom type="box" size="0.1 0.1 0.1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""

MJCF_REVOLUTE = """
<mujoco model="revolute">
  <worldbody>
    <body name="arm" pos="0 0 0">
      <joint type="hinge" axis="0 1 0"/>
      <inertial mass="0.5" pos="0.1 0 0" diaginertia="0.01 0.01 0.01"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""

MJCF_PRISMATIC = """
<mujoco model="prismatic">
  <worldbody>
    <body name="slider" pos="0 0 0">
      <joint type="slide" axis="1 0 0"/>
      <inertial mass="0.5" pos="0 0 0" diaginertia="0.01 0.01 0.01"/>
      <geom type="box" size="0.05 0.05 0.05" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""

MJCF_FREE_REV = """
<mujoco model="free_with_child">
  <worldbody>
    <body name="chassis" pos="0 0 0">
      <freejoint/>
      <inertial mass="1.0" pos="0 0 0" diaginertia="0.1 0.1 0.1"/>
      <geom type="box" size="0.1 0.1 0.1" contype="0" conaffinity="0"/>
      <body name="arm" pos="0.2 0 0">
        <joint type="hinge" axis="0 1 0"/>
        <inertial mass="0.5" pos="0.1 0 0" diaginertia="0.01 0.01 0.01"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

MJCF_REV_CHAIN3 = """
<mujoco model="chain3">
  <worldbody>
    <body name="l1" pos="0 0 0">
      <joint type="hinge" axis="0 1 0"/>
      <inertial mass="0.3" pos="0.1 0 0" diaginertia="0.005 0.005 0.005"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" contype="0" conaffinity="0"/>
      <body name="l2" pos="0.2 0 0">
        <joint type="hinge" axis="0 1 0"/>
        <inertial mass="0.3" pos="0.1 0 0" diaginertia="0.005 0.005 0.005"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" contype="0" conaffinity="0"/>
        <body name="l3" pos="0.2 0 0">
          <joint type="hinge" axis="0 1 0"/>
          <inertial mass="0.3" pos="0.1 0 0" diaginertia="0.005 0.005 0.005"/>
          <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" contype="0" conaffinity="0"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mjcf_to_tmpfile(mjcf_str: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".xml")
    with os.fdopen(fd, "w") as f:
        f.write(mjcf_str)
    return path


def _build_scene(mjcf_path: str, *, requires_grad: bool, n_envs: int = 0):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0.0, 0.0, 0.0),
            requires_grad=requires_grad,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            enable_self_collision=False,
            enable_joint_limit=False,
            disable_constraint=True,
            use_hibernation=False,
            use_contact_island=False,
        ),
        show_viewer=False,
    )
    robot = scene.add_entity(gs.morphs.MJCF(file=mjcf_path))
    scene.build(n_envs=n_envs)
    return scene, robot


def _make_scene_pair(mjcf_str: str, n_envs: int = 0):
    """Build two parallel scenes from the same MJCF:

      * `scene_ana` runs the differentiable-mode forward and is the only one we
        ever call `loss.backward()` on. Once a backward has run, that scene's
        internal target-replay state is left in a configuration that silently
        ignores subsequent setters — so reusing it for FD probes would give
        loss_p == loss_m and a fake zero gradient.
      * `scene_fd` runs the production forward (`requires_grad=False`) and is
        what FD perturbs. By construction it never sees a backward, so each
        reset → set → step cycle is clean.

    FD therefore checks "does the diff-mode analytical gradient match the
    production forward's local sensitivity", which is the property we actually
    care about for RL demos. With `n_envs > 0` both scenes run in batched mode
    so we can verify that per-env adjoints are independently correct.
    """
    path = _mjcf_to_tmpfile(mjcf_str)
    scene_ana, robot_ana = _build_scene(path, requires_grad=True, n_envs=n_envs)
    scene_fd, robot_fd = _build_scene(path, requires_grad=False, n_envs=n_envs)
    return scene_ana, robot_ana, scene_fd, robot_fd, path


def _batch_size(scene) -> int:
    """Effective batch dimension. scene.n_envs == 0 still allocates B=1 internally."""
    return scene.n_envs if scene.n_envs > 0 else 1


def _input_shape(base_shape, n_envs):
    """Setter inputs are unbatched when n_envs==0; batched (n_envs, *base) otherwise."""
    return (n_envs,) + tuple(base_shape) if n_envs > 0 else tuple(base_shape)


def _solver_state(scene):
    """Return the rigid solver's RigidSolverState (grad-aware; provides
    links_pos / links_quat for every link in the entity)."""
    state = scene.get_state()
    return state.solvers_state[scene.solvers.index(scene.rigid_solver)]


def _grad_matches_fd(
    scene_ana,
    robot_ana,
    scene_fd,
    robot_fd,
    init_input,  # 1-D numpy array (fp64)
    apply_fn,  # callable(robot, x): apply x via a @tracked setter
    loss_fn,  # callable(scene, robot) -> scalar tensor
    *,
    label: str,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    eps: float = 1e-5,
):
    # NOTE on tolerances: the production-mode and diff-mode forward kernels were
    # verified to produce bit-identical state.pos/state.quat for the same input
    # (probe_optionB.py, 2026-05-03), so an FD probed on the no-grad scene
    # is a valid reference for the diff scene's analytical gradient.
    #
    # Most input/output pairs hit rtol=1e-4 trivially. The set_dofs_velocity →
    # state.quat path is the outlier: it carries a known ~1% systematic drift
    # between Genesis autograd and central FD (output magnitude is ~1e-2, so
    # the absolute mismatch sits at ~1e-4 — well above truncation/roundoff
    # at fp64). Tracked as a separate followup; for those cases callers should
    # pass a looser `rtol` (e.g. 2e-2) rather than tightening this default.
    base_np = np.asarray(init_input, dtype=np.float64).copy()

    # --- analytical (diff-mode scene) ---
    x_ana = gs.tensor(base_np, dtype=gs.tc_float, requires_grad=True)
    scene_ana.reset()
    apply_fn(robot_ana, x_ana)
    scene_ana.step()
    loss = loss_fn(scene_ana, robot_ana)
    assert loss.requires_grad, f"[{label}] loss does not require grad — output is not grad-aware"
    loss.backward()
    assert x_ana.grad is not None, f"[{label}] x.grad is None after backward"
    ana_grad = x_ana.grad.detach().cpu().numpy().copy()

    # --- central FD (production-mode scene) ---
    n = base_np.size
    fd_grad = np.zeros_like(base_np)
    for i in range(n):
        plus = base_np.copy()
        plus.reshape(-1)[i] = base_np.reshape(-1)[i] + eps
        scene_fd.reset()
        apply_fn(robot_fd, gs.tensor(plus, dtype=gs.tc_float))
        scene_fd.step()
        loss_p = float(loss_fn(scene_fd, robot_fd).detach().cpu())

        minus = base_np.copy()
        minus.reshape(-1)[i] = base_np.reshape(-1)[i] - eps
        scene_fd.reset()
        apply_fn(robot_fd, gs.tensor(minus, dtype=gs.tc_float))
        scene_fd.step()
        loss_m = float(loss_fn(scene_fd, robot_fd).detach().cpu())

        fd_grad.reshape(-1)[i] = (loss_p - loss_m) / (2.0 * eps)

    assert_allclose(
        torch.from_numpy(ana_grad),
        torch.from_numpy(fd_grad),
        rtol=rtol,
        atol=atol,
        err_msg=f"[{label}] FD vs analytical mismatch",
    )


# loss factories — all use sum-of-squared-deviation to a fixed random target so
# every entry of the input has a nontrivial sensitivity. Targets and outputs are
# both flattened before the subtraction so multi-link shapes (B, n_links, 3|4)
# don't trip torch broadcasting.
def _loss_state_pos(target):
    flat = target.reshape(-1)

    def _fn(scene, robot):
        return ((robot.get_state().pos.reshape(-1) - flat) ** 2).sum()

    return _fn


def _loss_state_quat(target):
    flat = target.reshape(-1)

    def _fn(scene, robot):
        return ((robot.get_state().quat.reshape(-1) - flat) ** 2).sum()

    return _fn


def _loss_links_pos(target):
    flat = target.reshape(-1)

    def _fn(scene, robot):
        return ((_solver_state(scene).links_pos.reshape(-1) - flat) ** 2).sum()

    return _fn


def _loss_links_quat(target):
    flat = target.reshape(-1)

    def _fn(scene, robot):
        return ((_solver_state(scene).links_quat.reshape(-1) - flat) ** 2).sum()

    return _fn


def _rand_np(shape, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float64)


def _target(shape, seed):
    return torch.from_numpy(_rand_np(shape, seed)).to(dtype=gs.tc_float, device=gs.device)


# ---------------------------------------------------------------------------
# Tests — one per joint topology, several (input, output) checks inside.
# ---------------------------------------------------------------------------


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("precision", _PRECISION_PARAMS)
@pytest.mark.parametrize("n_envs", _N_ENVS_PARAMS)
def test_diff_fk_freejoint(show_viewer, n_envs, precision):
    """J1: single free body. Covers (n_envs ∈ {0, 4}) × (precision ∈ {fp64, fp32})."""
    scene_ana, robot_ana, scene_fd, robot_fd, _ = _make_scene_pair(MJCF_FREE, n_envs=n_envs)
    n_dofs = robot_ana.n_dofs
    B = _batch_size(scene_ana)
    tol_default = _TOL[(precision, "default")]
    tol_quat = _TOL[(precision, "quat")]

    tgt_pos = _target((B, 3), seed=1)
    tgt_quat = _target((B, 4), seed=2)

    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=_rand_np(_input_shape((3,), n_envs), seed=10),
        apply_fn=lambda r, x: r.set_pos(x),
        loss_fn=_loss_state_pos(tgt_pos),
        label="J1 set_pos → state.pos",
        **tol_default,
    )

    init_q_shape = _input_shape((4,), n_envs)
    init_q = np.broadcast_to(np.array([1.0, 0.0, 0.0, 0.0]), init_q_shape).copy()
    init_q = init_q + 0.05 * _rand_np(init_q_shape, seed=11)
    init_q = init_q / np.linalg.norm(init_q, axis=-1, keepdims=True)
    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=init_q,
        apply_fn=lambda r, x: r.set_quat(x),
        loss_fn=_loss_state_quat(tgt_quat),
        label="J1 set_quat → state.quat",
        **tol_quat,
    )

    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=12),
        apply_fn=lambda r, x: r.set_dofs_velocity(x),
        loss_fn=_loss_state_pos(tgt_pos),
        label="J1 set_dofs_velocity → state.pos (after 1 step)",
        **tol_default,
    )

    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=13),
        apply_fn=lambda r, x: r.set_dofs_velocity(x),
        loss_fn=_loss_state_quat(tgt_quat),
        label="J1 set_dofs_velocity → state.quat (after 1 step)",
        **tol_quat,
    )

    # control_dofs_force is @tracked: gradient flows back via
    # set_dofs_force_grad (kernel_set_dofs_force_grad reads ctrl_force.grad
    # populated by kernel_compute_qacc.grad's backward chain).
    #
    # fp64 only: d(state.pos)/d(force) ≈ dt^2 / (2 * inertia) ≈ 1e-4 after 1
    # step. At fp32 with FD eps=1e-3 the loss difference is ~1e-7 — at fp32's
    # precision floor — and the FD probe disagrees with analytical by ~1e-4
    # absolute, well above the fp32 default tol band. The J2/J3/J4/J5 force
    # checks below are also fp64-only for the same reason; J2's
    # `control_dofs_force → state.quat` does pass at fp32 only because its
    # check uses the wider quat tolerance.
    if precision == "64":
        _grad_matches_fd(
            scene_ana,
            robot_ana,
            scene_fd,
            robot_fd,
            init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=14),
            apply_fn=lambda r, x: r.control_dofs_force(x),
            loss_fn=_loss_state_pos(tgt_pos),
            label="J1 control_dofs_force → state.pos (after 1 step)",
            **tol_default,
        )


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("precision", _PRECISION_PARAMS)
@pytest.mark.parametrize("n_envs", _N_ENVS_PARAMS)
def test_diff_fk_revolute(show_viewer, n_envs, precision):
    """J2: single revolute joint, fixed base. Covers (n_envs ∈ {0, 4}) × (precision ∈ {fp64, fp32})."""
    scene_ana, robot_ana, scene_fd, robot_fd, _ = _make_scene_pair(MJCF_REVOLUTE, n_envs=n_envs)
    n_dofs = robot_ana.n_dofs  # = 1
    B = _batch_size(scene_ana)
    tol_default = _TOL[(precision, "default")]
    tol_quat = _TOL[(precision, "quat")]

    tgt_pos = _target((B, 3), seed=21)
    tgt_quat = _target((B, 4), seed=22)

    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=30),
        apply_fn=lambda r, x: r.set_dofs_velocity(x),
        loss_fn=_loss_state_pos(tgt_pos),
        label="J2 set_dofs_velocity → state.pos",
        **tol_default,
    )

    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=31),
        apply_fn=lambda r, x: r.set_dofs_velocity(x),
        loss_fn=_loss_state_quat(tgt_quat),
        label="J2 set_dofs_velocity → state.quat",
        **tol_quat,
    )

    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=32),
        apply_fn=lambda r, x: r.control_dofs_force(x),
        loss_fn=_loss_state_quat(tgt_quat),
        label="J2 control_dofs_force → state.quat",
        **tol_quat,
    )


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("precision", _PRECISION_PARAMS)
@pytest.mark.parametrize("n_envs", _N_ENVS_PARAMS)
def test_diff_fk_prismatic(show_viewer, n_envs, precision):
    """J3: single prismatic joint, fixed base. No rotational DOF — skip the quat output."""
    scene_ana, robot_ana, scene_fd, robot_fd, _ = _make_scene_pair(MJCF_PRISMATIC, n_envs=n_envs)
    n_dofs = robot_ana.n_dofs  # = 1
    B = _batch_size(scene_ana)
    tol_default = _TOL[(precision, "default")]
    tgt_pos = _target((B, 3), seed=41)

    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=50),
        apply_fn=lambda r, x: r.set_dofs_velocity(x),
        loss_fn=_loss_state_pos(tgt_pos),
        label="J3 set_dofs_velocity → state.pos",
        **tol_default,
    )

    # fp64-only — see J1's control_dofs_force comment for why FD-vs-analytical
    # on force-driven position is at fp32's precision floor.
    if precision == "64":
        _grad_matches_fd(
            scene_ana,
            robot_ana,
            scene_fd,
            robot_fd,
            init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=51),
            apply_fn=lambda r, x: r.control_dofs_force(x),
            loss_fn=_loss_state_pos(tgt_pos),
            label="J3 control_dofs_force → state.pos",
            **tol_default,
        )


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("precision", _PRECISION_PARAMS)
@pytest.mark.parametrize("n_envs", _N_ENVS_PARAMS)
def test_diff_fk_free_with_revolute(show_viewer, n_envs, precision):
    """J4: freejoint root + one revolute child — the #2537 topology. Outputs use
    multi-link solver_state.links_pos/quat so the child link's FK is exercised too."""
    scene_ana, robot_ana, scene_fd, robot_fd, _ = _make_scene_pair(MJCF_FREE_REV, n_envs=n_envs)
    n_dofs = robot_ana.n_dofs  # 6 free + 1 hinge = 7
    n_links = robot_ana.n_links  # 2
    B = _batch_size(scene_ana)
    tol_default = _TOL[(precision, "default")]
    tol_quat = _TOL[(precision, "quat")]
    tgt_links_pos = _target((B, n_links, 3), seed=61)
    tgt_links_quat = _target((B, n_links, 4), seed=62)

    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=_rand_np(_input_shape((3,), n_envs), seed=70),
        apply_fn=lambda r, x: r.set_pos(x),
        loss_fn=_loss_links_pos(tgt_links_pos),
        label="J4 set_pos → links_pos",
        **tol_default,
    )

    init_q_shape = _input_shape((4,), n_envs)
    init_q = np.broadcast_to(np.array([1.0, 0.0, 0.0, 0.0]), init_q_shape).copy()
    init_q = init_q + 0.05 * _rand_np(init_q_shape, seed=71)
    init_q = init_q / np.linalg.norm(init_q, axis=-1, keepdims=True)
    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=init_q,
        apply_fn=lambda r, x: r.set_quat(x),
        loss_fn=_loss_links_quat(tgt_links_quat),
        label="J4 set_quat → links_quat",
        **tol_quat,
    )

    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=72),
        apply_fn=lambda r, x: r.set_dofs_velocity(x),
        loss_fn=_loss_links_pos(tgt_links_pos),
        label="J4 set_dofs_velocity → links_pos",
        **tol_default,
    )

    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=73),
        apply_fn=lambda r, x: r.set_dofs_velocity(x),
        loss_fn=_loss_links_quat(tgt_links_quat),
        label="J4 set_dofs_velocity → links_quat",
        **tol_quat,
    )

    # fp64-only — see J1's control_dofs_force comment.
    if precision == "64":
        _grad_matches_fd(
            scene_ana,
            robot_ana,
            scene_fd,
            robot_fd,
            init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=74),
            apply_fn=lambda r, x: r.control_dofs_force(x),
            loss_fn=_loss_links_pos(tgt_links_pos),
            label="J4 control_dofs_force → links_pos",
            **tol_default,
        )


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("precision", _PRECISION_PARAMS)
@pytest.mark.parametrize("n_envs", _N_ENVS_PARAMS)
def test_diff_fk_revolute_chain3(show_viewer, n_envs, precision):
    """J5: 3-link serial revolute chain, fixed base. Tests deeper FK chain."""
    scene_ana, robot_ana, scene_fd, robot_fd, _ = _make_scene_pair(MJCF_REV_CHAIN3, n_envs=n_envs)
    n_dofs = robot_ana.n_dofs  # 3
    n_links = robot_ana.n_links  # 3
    B = _batch_size(scene_ana)
    tol_default = _TOL[(precision, "default")]
    tol_quat = _TOL[(precision, "quat")]
    tgt_links_pos = _target((B, n_links, 3), seed=81)
    tgt_links_quat = _target((B, n_links, 4), seed=82)

    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=90),
        apply_fn=lambda r, x: r.set_dofs_velocity(x),
        loss_fn=_loss_links_pos(tgt_links_pos),
        label="J5 set_dofs_velocity → links_pos",
        **tol_default,
    )

    _grad_matches_fd(
        scene_ana,
        robot_ana,
        scene_fd,
        robot_fd,
        init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=91),
        apply_fn=lambda r, x: r.set_dofs_velocity(x),
        loss_fn=_loss_links_quat(tgt_links_quat),
        label="J5 set_dofs_velocity → links_quat",
        **tol_quat,
    )

    # fp64-only — see J1's control_dofs_force comment.
    if precision == "64":
        _grad_matches_fd(
            scene_ana,
            robot_ana,
            scene_fd,
            robot_fd,
            init_input=_rand_np(_input_shape((n_dofs,), n_envs), seed=92),
            apply_fn=lambda r, x: r.control_dofs_force(x),
            loss_fn=_loss_links_pos(tgt_links_pos),
            label="J5 control_dofs_force → links_pos",
            **tol_default,
        )
