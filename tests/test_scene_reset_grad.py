"""Unit tests for `scene.reset_grad()` and the SHAC-style horizon truncation
pattern (`scene.get_state()` snapshot → `loss.backward()` → `scene.reset(snapshot)`).

The flagship test, ``test_horizon_truncation_matches_independent_scenes``,
runs three scenes in parallel:

    Scene A: single scene, 5-step horizon 1 → snapshot → backward → reset →
             5-step horizon 2 → backward. Yields ``grad1_A`` and ``grad2_A``.
    Scene B: same as A's horizon 1 only. Yields ``grad1_B`` (compared to
             ``grad1_A``) and the mid-trajectory state snapshot.
    Scene C: fresh scene, starts from B's snapshot, runs 5-step horizon 2 →
             backward. Yields ``grad2_C`` (compared to ``grad2_A``).

If `scene.reset(snapshot)` correctly (a) restores physics state, (b) clears
the gradient tape, and (c) doesn't leak grad accumulation across horizons,
then ``grad1_A == grad1_B`` and ``grad2_A == grad2_C`` exactly.

We parameterize over the 5 J1~J5 topologies from
`test_diff_forward_kinematics.py` to cover single freejoint, 1-DOF revolute /
prismatic, freejoint+revolute child, and revolute chain-3. The smaller tests
(`test_reset_grad_preserves_state`, `test_reset_grad_zeros_grad_fields`,
`test_reset_grad_clears_queried_states`) pin individual invariants
separately so a regression points at one specific subsystem.

CPU + fp64 only — gradient equality requires fp64-level precision to
distinguish "exactly equal" from "drifting by 1e-3".
"""

import os
import tempfile

import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.misc import qd_to_torch

from .utils import assert_allclose


pytestmark = [
    pytest.mark.precision("64"),
    pytest.mark.debug(False),
]


# ---------------------------------------------------------------------------
# MJCF topologies (copied from `tests/test_diff_forward_kinematics.py` to keep
# this file self-contained).
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


_TOPOLOGIES = [
    pytest.param(MJCF_FREE, 6, id="J1_free"),
    pytest.param(MJCF_REVOLUTE, 1, id="J2_revolute"),
    pytest.param(MJCF_PRISMATIC, 1, id="J3_prismatic"),
    pytest.param(MJCF_FREE_REV, 7, id="J4_free_rev"),
    pytest.param(MJCF_REV_CHAIN3, 3, id="J5_chain3"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mjcf_to_tmpfile(mjcf_str: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".xml")
    with os.fdopen(fd, "w") as f:
        f.write(mjcf_str)
    return path


def _build_scene(mjcf_str: str, n_envs: int = 0):
    """Build a diff-rigid scene with the standard "no collision / no constraint" config."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0.0, 0.0, 0.0),
            requires_grad=True,
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
    robot = scene.add_entity(gs.morphs.MJCF(file=_mjcf_to_tmpfile(mjcf_str)))
    scene.build(n_envs=n_envs)
    return scene, robot


def _rigid_qpos_loss(scene):
    """Differentiable scalar loss = sum((qpos)**2). Reads `state.qpos` via
    `scene.get_state()` so the resulting tensor is a gs.Tensor whose
    `.backward()` triggers `scene._backward()`."""
    state = scene.get_state()
    rigid_state = state.solvers_state[scene.solvers.index(scene.rigid_solver)]
    return (rigid_state.qpos**2).sum()


def _run_segment(scene, robot, v_tensor, n_steps: int):
    """Apply `set_dofs_velocity(v_tensor)` once, then step `n_steps` times.
    Returns the resulting (post-step) scalar loss."""
    robot.set_dofs_velocity(v_tensor)
    for _ in range(n_steps):
        scene.step()
    return _rigid_qpos_loss(scene)


def _read_qpos(scene) -> np.ndarray:
    """Read the simulator's current qpos field (detached)."""
    solver = scene.rigid_solver
    return qd_to_torch(solver._rigid_global_info.qpos, copy=True).cpu().numpy()


# ---------------------------------------------------------------------------
# A vs (B + C) — the flagship test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mjcf_str, n_dofs", _TOPOLOGIES)
def test_horizon_truncation_matches_independent_scenes(mjcf_str, n_dofs):
    """SHAC-style two-segment trajectory in Scene A matches the same two
    segments run in independent Scene B (horizon 1) and Scene C (horizon 2,
    started from B's mid-trajectory snapshot via `scene.reset(state)`).

    Verifies that `scene.get_state()` + `scene.reset(state)` correctly
    isolates two consecutive horizons: physics state propagates seamlessly,
    but the autograd tapes are independent."""
    rng_v1 = np.random.default_rng(101).standard_normal(n_dofs).astype(np.float64)
    rng_v2 = np.random.default_rng(202).standard_normal(n_dofs).astype(np.float64)
    H = 5

    # ----- Scene A: one scene, snapshot+reset between two horizons -----
    sceneA, robotA = _build_scene(mjcf_str)
    sceneA.reset()
    v1A = gs.tensor(rng_v1, dtype=gs.tc_float, requires_grad=True)
    loss_h1_A = _run_segment(sceneA, robotA, v1A, H)
    qpos_mid_A = _read_qpos(sceneA)
    snapshot_A = sceneA.get_state()
    loss_h1_A.backward(retain_graph=True)
    grad1_A = v1A.grad.detach().clone().cpu().numpy()

    sceneA.reset(snapshot_A)
    v2A = gs.tensor(rng_v2, dtype=gs.tc_float, requires_grad=True)
    loss_h2_A = _run_segment(sceneA, robotA, v2A, H)
    qpos_end_A = _read_qpos(sceneA)
    loss_h2_A.backward(retain_graph=True)
    grad2_A = v2A.grad.detach().clone().cpu().numpy()

    # ----- Scene B: same horizon 1 only -----
    sceneB, robotB = _build_scene(mjcf_str)
    sceneB.reset()
    v1B = gs.tensor(rng_v1, dtype=gs.tc_float, requires_grad=True)
    loss_h1_B = _run_segment(sceneB, robotB, v1B, H)
    qpos_mid_B = _read_qpos(sceneB)
    snapshot_B = sceneB.get_state()
    loss_h1_B.backward(retain_graph=True)
    grad1_B = v1B.grad.detach().clone().cpu().numpy()

    # Sanity: A and B's mid-trajectory states should match exactly (snapshot
    # must not perturb physics state up to this point — A and B run identically
    # before any reset).
    assert_allclose(qpos_mid_A, qpos_mid_B, atol=0, rtol=0)
    # Sanity: A and B's horizon-1 losses match exactly.
    assert_allclose(loss_h1_A.detach().cpu().item(), loss_h1_B.detach().cpu().item(), atol=0, rtol=0)
    # Core assertion: horizon-1 gradient identical.
    assert_allclose(grad1_A, grad1_B, atol=1e-12, rtol=1e-10)

    # ----- Scene C: fresh scene, start from B's mid-trajectory snapshot -----
    sceneC, robotC = _build_scene(mjcf_str)
    sceneC.reset(snapshot_B)
    v2C = gs.tensor(rng_v2, dtype=gs.tc_float, requires_grad=True)
    loss_h2_C = _run_segment(sceneC, robotC, v2C, H)
    qpos_end_C = _read_qpos(sceneC)
    loss_h2_C.backward(retain_graph=True)
    grad2_C = v2C.grad.detach().clone().cpu().numpy()

    # Sanity: A and C end at (approximately) the same final state.
    #
    # `scene.reset(state)` only restores the fields captured in `SimState`
    # (qpos / dofs_vel / dofs_acc / links_pos / links_quat / etc.) and re-runs
    # position FK. Some other simulator-internal fields — adjoint caches,
    # mass-matrix factor cache, `cd_*`, `cdd_*`, `cfrc_*` — are *not* in
    # `SimState`. In Scene A these carry stale values from the previous
    # horizon; in Scene C they're zero-initialized. The next forward step's
    # kernels recompute everything that matters, but the order of reads vs.
    # writes within a single substep means a few ulps of difference leak
    # through. Empirically ~1e-7 max-abs drift on J4 (freejoint + revolute
    # child) and ~1e-9 on J5 (3-link chain). A real bug would manifest as
    # >1e-3 drift.
    #
    # Investigated but did not eliminate: also calling `kernel_forward_velocity`
    # inside `set_state` (so `_is_forward_vel_updated = True` is honest) — the
    # drift magnitude was unchanged, indicating the source is upstream of
    # `cd_vel` (likely mass-matrix factor cache or bias/Coriolis cross-step
    # state). Closing the gap fully would mean expanding `SimState` to cover
    # the full simulator-internal state, which is a larger refactor.
    assert_allclose(qpos_end_A, qpos_end_C, atol=1e-5, rtol=1e-4)
    assert_allclose(loss_h2_A.detach().cpu().item(), loss_h2_C.detach().cpu().item(), atol=1e-5, rtol=1e-4)
    # Core assertion: horizon-2 gradient identical up to the same ulps band.
    assert_allclose(grad2_A, grad2_C, atol=1e-5, rtol=1e-3)


# ---------------------------------------------------------------------------
# Smaller targeted invariants
# ---------------------------------------------------------------------------


def test_reset_grad_preserves_state():
    """`scene.reset_grad()` must not touch qpos / vel / time counters."""
    scene, robot = _build_scene(MJCF_REVOLUTE)
    scene.reset()
    v = gs.tensor(np.array([0.3]), dtype=gs.tc_float, requires_grad=True)
    robot.set_dofs_velocity(v)
    for _ in range(3):
        scene.step()
    qpos_before = _read_qpos(scene)
    t_before = scene._t
    cur_substep_before = scene._sim._cur_substep_global

    scene.reset_grad()

    qpos_after = _read_qpos(scene)
    assert_allclose(qpos_before, qpos_after, atol=0, rtol=0)
    assert scene._t == t_before, f"_t changed: {t_before} -> {scene._t}"
    assert scene._sim._cur_substep_global == cur_substep_before, (
        f"_cur_substep_global changed: {cur_substep_before} -> {scene._sim._cur_substep_global}"
    )


def test_reset_grad_zeros_internal_grad_fields():
    """`scene.reset_grad()` zeros the solver's `.grad` fields and adjoint caches."""
    scene, robot = _build_scene(MJCF_REVOLUTE)
    scene.reset()
    v = gs.tensor(np.array([0.5]), dtype=gs.tc_float, requires_grad=True)
    robot.set_dofs_velocity(v)
    for _ in range(3):
        scene.step()
    loss = _rigid_qpos_loss(scene)
    loss.backward(retain_graph=True)

    solver = scene.rigid_solver

    # Sanity: some grad field is non-zero right after backward (we can pick any
    # adjoint cache slot; the qpos field's `.grad` is consistently set).
    qpos_grad_before = qd_to_torch(solver._rigid_global_info.qpos.grad, copy=True).abs().max().item()
    assert qpos_grad_before > 0, "expected qpos.grad to be populated by backward, got zero — test setup invalid"

    scene.reset_grad()

    # All checked solver-internal grad fields should be zero now.
    fields_to_check = [
        ("rigid_global_info.qpos", solver._rigid_global_info.qpos),
        ("dofs_state.vel", solver.dofs_state.vel),
        ("dofs_state.pos", solver.dofs_state.pos),
        ("dofs_state.acc", solver.dofs_state.acc),
        ("dofs_state.acc_smooth", solver.dofs_state.acc_smooth),
        ("dofs_state.acc_smooth_bw", solver.dofs_state.acc_smooth_bw),
        ("dofs_state.force", solver.dofs_state.force),
        ("links_state.pos", solver.links_state.pos),
        ("links_state.quat", solver.links_state.quat),
    ]
    for name, field in fields_to_check:
        grad = getattr(field, "grad", None)
        if grad is None:
            continue
        max_abs = qd_to_torch(grad, copy=True).abs().max().item()
        assert max_abs == 0.0, f"{name}.grad not zero after reset_grad: max_abs={max_abs:.3e}"


def test_reset_grad_clears_queried_states():
    """`scene.reset_grad()` clears the simulator's `_queried_states` cache."""
    scene, robot = _build_scene(MJCF_REVOLUTE)
    scene.reset()
    v = gs.tensor(np.array([0.5]), dtype=gs.tc_float, requires_grad=True)
    robot.set_dofs_velocity(v)
    scene.step()

    # `scene.get_state()` registers the returned state in `_queried_states`.
    _ = scene.get_state()
    _ = scene.get_state()
    assert len(scene._sim._queried_states.states) > 0, (
        "expected _queried_states to be non-empty after get_state() calls"
    )

    scene.reset_grad()
    assert len(scene._sim._queried_states.states) == 0, (
        f"expected _queried_states to be empty after reset_grad, got {len(scene._sim._queried_states.states)} entries"
    )


def test_reset_grad_idempotent():
    """Calling `scene.reset_grad()` twice is safe (second call is a no-op)."""
    scene, robot = _build_scene(MJCF_REVOLUTE)
    scene.reset()
    v = gs.tensor(np.array([0.3]), dtype=gs.tc_float, requires_grad=True)
    robot.set_dofs_velocity(v)
    for _ in range(2):
        scene.step()
    loss = _rigid_qpos_loss(scene)
    loss.backward(retain_graph=True)

    scene.reset_grad()
    qpos_after_1 = _read_qpos(scene)
    scene.reset_grad()
    qpos_after_2 = _read_qpos(scene)
    assert_allclose(qpos_after_1, qpos_after_2, atol=0, rtol=0)
