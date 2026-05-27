"""Forward enforcement + backward FD for the **joint-limit** inequality
constraint (`enable_joint_limit=True`).

Within the differentiable-rigid test suite, this is the joint-limit layer. The
siblings isolate the other layers:
  - test_diff_forward_kinematics : unconstrained FK + velocity gradient (no
      constraints) — the base local-gradient bar.
  - test_diff_joint_limit        : *this file* — the joint-limit inequality
      constraint reverse, the only constraint exercised here.
  - test_diff_contact            : the collision constraint + diff-GJK reverse.
  - test_diff_scene_backward     : the scene.backward() API + horizon truncation.
  - test_diff_optim              : end-to-end optimization convergence.

Cases: a single sliding cart on x with `range="-4 4"` (cartpole's `slider`
layout), and the production hopper (planar floating base + limited leg joints).
Two checks per case:

  1. **Forward enforcement**. Drive the joint hard enough that it would leave
     its band if unconstrained, then assert the coordinate stays bounded when
     the limit is on (and is unbounded when off, as a control).

  2. **Backward FD agreement**. Set `dofs_velocity` / per-step force from a leaf
     tensor, roll out, and check `d(loss)/d(input)` from diff-mode autograd
     against central FD within the same tolerance `test_diff_forward_kinematics`
     uses — i.e. the joint-limit constraint reverse contributes a correct
     gradient, not just the unconstrained dynamics.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import genesis as gs

from .utils import assert_allclose


# Production hopper MJCF (planar floating base = slide-x + slide-z + hinge-y on
# one body, then thigh/leg/foot hinges with joint ranges). Built collision-free
# here so only the joint-limit constraint — not contact — is exercised, while
# the multi-joint base still runs through the FK backward.
MJCF_HOPPER = (Path(__file__).resolve().parent.parent / "examples" / "diffrl" / "envs" / "hopper.xml").read_text()


MJCF_SLIDER_LIMIT = """
<mujoco model="slider_limit">
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="cart" pos="0 0 0">
      <joint name="slider" type="slide" axis="1 0 0" range="-4 4" damping="0.0"/>
      <inertial pos="0 0 0" mass="1.0" diaginertia="1.0 1.0 1.0"/>
      <geom type="box" size="0.25 0.25 0.1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


def _mjcf_tmpfile(s: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".xml")
    with os.fdopen(fd, "w") as f:
        f.write(s)
    return path


def _build(mjcf_path: str, *, requires_grad: bool, enable_joint_limit: bool):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            substeps=4,
            gravity=(0.0, 0.0, 0.0),
            requires_grad=requires_grad,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            enable_self_collision=False,
            enable_joint_limit=enable_joint_limit,
            disable_constraint=not enable_joint_limit,
            use_hibernation=False,
            use_contact_island=False,
        ),
        show_viewer=False,
    )
    robot = scene.add_entity(gs.morphs.MJCF(file=mjcf_path))
    scene.build(n_envs=0)
    return scene, robot


def _rigid_state(scene):
    return scene.get_state().solvers_state[scene.solvers.index(scene.rigid_solver)]


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_diff_joint_limit_forward_enforcement(show_viewer):
    """With `enable_joint_limit=True` and slider range=[-4,4], pushing the
    cart at v=100 for 60 steps must keep |x| bounded; with the limit off
    the cart drifts past 90 m (control case)."""
    mjcf_path = _mjcf_tmpfile(MJCF_SLIDER_LIMIT)

    # Control: limit OFF
    scene, robot = _build(mjcf_path, requires_grad=False, enable_joint_limit=False)
    scene.reset()
    robot.set_dofs_velocity(gs.tensor([100.0], dtype=gs.tc_float))
    for _ in range(60):
        scene.step()
    x_off = float(_rigid_state(scene).qpos[0, 0].detach())
    assert abs(x_off) > 50.0, f"control (limit OFF) cart should drift past 50m, got x={x_off}"

    # Limit ON — should stay bounded.
    scene, robot = _build(mjcf_path, requires_grad=False, enable_joint_limit=True)
    scene.reset()
    robot.set_dofs_velocity(gs.tensor([100.0], dtype=gs.tc_float))
    for _ in range(60):
        scene.step()
    x_on = float(_rigid_state(scene).qpos[0, 0].detach())
    assert abs(x_on) <= 4.5, f"limit ON should keep |x| <= 4.5 (small margin for soft constraint), got x={x_on}"


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("init_vel", [0.5, 5.0])
def test_diff_joint_limit_backward_finite_no_limit_hit(show_viewer, init_vel):
    """When the rollout stays well inside the joint range, the joint-limit
    branch should *not* activate (`pos_delta >= 0`), and the gradient should
    match the no-limit baseline almost byte-exactly. This pins the
    "limit-on but inactive" path through the constraint solver."""
    mjcf_path = _mjcf_tmpfile(MJCF_SLIDER_LIMIT)
    N_STEPS = 1  # short — cart doesn't reach limit

    grads = {}
    for limit in (False, True):
        scene, robot = _build(mjcf_path, requires_grad=True, enable_joint_limit=limit)
        scene.reset()
        v = gs.tensor([init_vel], dtype=gs.tc_float, requires_grad=True)
        robot.set_dofs_velocity(v)
        for _ in range(N_STEPS):
            scene.step()
        loss = (_rigid_state(scene).qpos[0, 0]) ** 2
        loss.backward()
        assert v.grad is not None, f"limit={limit}: v.grad is None"
        g = float(v.grad[0])
        assert math.isfinite(g), f"limit={limit}: gradient is not finite ({g})"
        grads[limit] = g

    # Limit-inactive case should match the no-limit baseline tightly — the
    # constraint branch only runs `n_constraints += 0`, so the autograd tape
    # should be identical up to floating-point.
    assert_allclose(grads[True], grads[False], rtol=1e-6, atol=1e-9)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_diff_joint_limit_backward_fd_one_step(show_viewer):
    """FD vs analytical gradient when the cart starts well inside the limit
    and takes a single step — verifies the constraint-solver-inclusive
    forward+backward chain still satisfies central FD."""
    mjcf_path = _mjcf_tmpfile(MJCF_SLIDER_LIMIT)
    init_vel = 2.0
    eps = 1e-5

    # Analytical
    scene_ana, robot_ana = _build(mjcf_path, requires_grad=True, enable_joint_limit=True)
    scene_ana.reset()
    v = gs.tensor([init_vel], dtype=gs.tc_float, requires_grad=True)
    robot_ana.set_dofs_velocity(v)
    scene_ana.step()
    loss = (_rigid_state(scene_ana).qpos[0, 0]) ** 2
    loss.backward()
    ana = float(v.grad[0])

    # FD
    scene_fd, robot_fd = _build(mjcf_path, requires_grad=False, enable_joint_limit=True)

    def loss_at(val: float) -> float:
        scene_fd.reset()
        robot_fd.set_dofs_velocity(gs.tensor([val], dtype=gs.tc_float))
        scene_fd.step()
        return float((_rigid_state(scene_fd).qpos[0, 0]) ** 2)

    fd = (loss_at(init_vel + eps) - loss_at(init_vel - eps)) / (2 * eps)

    assert_allclose(ana, fd, rtol=1e-3, atol=1e-6)


# (init_vel, n_steps) cases where the cart actually crosses |x|=4 during the
# rollout. Each case engages the constraint solver during the integration —
# they cover the `M^{-1} J^T λ` correction path that the unconstrained
# `kernel_manual_compute_qacc_bw` could not produce. Resolved 2026-05-25 by
# wiring `constraint_solver.backward` + `kernel_manual_add_joint_limit_constraints_bw`
# into `substep_pre_coupling_grad`.
_FD_ACTIVE_CASES = [
    (500.0, 1),
    (200.0, 2),
    (100.0, 5),
    (50.0, 10),
]


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("init_vel,n_steps", _FD_ACTIVE_CASES)
def test_diff_joint_limit_backward_fd_active(show_viewer, init_vel, n_steps):
    """FD vs analytical when the cart actually crosses the joint limit during
    the rollout. Exercises the constrained backward path
    (`constraint_solver.backward` + `kernel_manual_add_joint_limit_constraints_bw`)
    on cases where the unconstrained IFT alone would drop the `M^{-1} J^T λ`
    correction and disagree with FD (often sign-flipped). Snapshot of the
    expected gradients (FP64 CPU, FD with eps=1e-4) at fix time:

        init_vel  n_steps  x_final   v.grad
        500       1        +4.464    +5.51e-2
        200       2        +4.203    +9.46e-2
        100       5        +3.892    -1.60e-1
         50      10        +3.606    -1.16e+0
    """
    mjcf_path = _mjcf_tmpfile(MJCF_SLIDER_LIMIT)
    eps = 1e-4

    # Analytical
    scene_ana, robot_ana = _build(mjcf_path, requires_grad=True, enable_joint_limit=True)
    scene_ana.reset()
    v = gs.tensor([init_vel], dtype=gs.tc_float, requires_grad=True)
    robot_ana.set_dofs_velocity(v)
    for _ in range(n_steps):
        scene_ana.step()
    x_final = float(_rigid_state(scene_ana).qpos[0, 0].detach())
    # Setup sanity: the cart must have entered the limit band, otherwise this
    # case wouldn't actually exercise the constraint correction path.
    assert abs(x_final) > 3.5, (
        f"setup error: init_vel={init_vel}, n_steps={n_steps} did not bring "
        f"the cart near the limit (x_final={x_final}); pick a larger v0 or "
        f"more steps."
    )
    loss = (_rigid_state(scene_ana).qpos[0, 0]) ** 2
    loss.backward()
    ana = float(v.grad[0])

    # FD
    scene_fd, robot_fd = _build(mjcf_path, requires_grad=False, enable_joint_limit=True)

    def loss_at(val: float) -> float:
        scene_fd.reset()
        robot_fd.set_dofs_velocity(gs.tensor([val], dtype=gs.tc_float))
        for _ in range(n_steps):
            scene_fd.step()
        return float((_rigid_state(scene_fd).qpos[0, 0]) ** 2)

    fd = (loss_at(init_vel + eps) - loss_at(init_vel - eps)) / (2 * eps)

    assert_allclose(ana, fd, rtol=1e-3, atol=1e-6)


# Per-step force horizons that drive the cart into the slider limit through
# `control_dofs_force`. Constant +500 N over `n_steps` accelerates the
# unit-mass cart past |x|=4 within ~10 substep-groups at dt=1/60, substeps=4
# (default solref); shorter horizons leave the cart inside the band and
# don't exercise the constraint backward, so we restrict to multi-step
# active cases. n_steps=10 probes whether the per-step `force.grad` for
# early-horizon steps leaks a wrong gradient when the constrained backward
# chain (`constraint_solver.backward` + manual joint-limit BW +
# `fwd_dynamics_without_qacc.grad` accumulation) runs across many substeps.
_FD_FORCE_CASES = [10]


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("n_steps", _FD_FORCE_CASES)
def test_diff_joint_limit_backward_fd_per_step_force(show_viewer, n_steps):
    """Central-FD vs analytical gradient on a per-step `control_dofs_force`
    time series that drives the cart into the slider limit during the
    rollout. Multi-step variant of `test_diff_joint_limit_backward_fd_active`.

    Goal: catch cross-step gradient *leak* through the constrained backward
    chain. Each step's `force.grad` must independently agree with the FD
    estimate of `d(loss)/d(force[t])`. A mismatch on early-horizon steps
    (before the cart reaches the limit) would indicate that later-step
    constraint backward contributions are bleeding into earlier-step
    gradients — the failure mode SHAC's 32-step actor backward exhibits.
    """
    mjcf_path = _mjcf_tmpfile(MJCF_SLIDER_LIMIT)
    eps = 1e-2
    force_value = 500.0
    init_force = np.full((n_steps, 1), force_value, dtype=np.float64)

    # Analytical
    scene_ana, robot_ana = _build(mjcf_path, requires_grad=True, enable_joint_limit=True)
    scene_ana.reset()
    forces = [gs.tensor(init_force[t], dtype=gs.tc_float, requires_grad=True) for t in range(n_steps)]
    for t in range(n_steps):
        robot_ana.control_dofs_force(forces[t])
        scene_ana.step()
    x_final = float(_rigid_state(scene_ana).qpos[0, 0].detach())
    # Setup sanity: the cart must have entered the limit band, otherwise this
    # case wouldn't actually exercise the multi-step constraint backward.
    assert abs(x_final) > 3.5, (
        f"setup error: n_steps={n_steps} at force={force_value} did not bring "
        f"the cart near the limit (x_final={x_final}); pick a larger force or "
        f"more steps."
    )
    loss = (_rigid_state(scene_ana).qpos[0, 0]) ** 2
    loss.backward()
    for t, f in enumerate(forces):
        assert f.grad is not None, f"step {t}: force.grad is None"
    ana = np.array([float(f.grad[0]) for f in forces])

    # FD per-step
    scene_fd, robot_fd = _build(mjcf_path, requires_grad=False, enable_joint_limit=True)

    def loss_at(perturbed: np.ndarray) -> float:
        scene_fd.reset()
        for t in range(n_steps):
            robot_fd.control_dofs_force(gs.tensor(perturbed[t], dtype=gs.tc_float))
            scene_fd.step()
        return float((_rigid_state(scene_fd).qpos[0, 0]) ** 2)

    fd = np.zeros(n_steps)
    for t in range(n_steps):
        plus = init_force.copy()
        plus[t, 0] += eps
        minus = init_force.copy()
        minus[t, 0] -= eps
        fd[t] = (loss_at(plus) - loss_at(minus)) / (2 * eps)

    # Per-step comparison so the failure message identifies the offending step.
    for t in range(n_steps):
        assert_allclose(
            ana[t],
            fd[t],
            rtol=1e-3,
            atol=1e-4,
            err_msg=(
                f"per-step force.grad mismatch at t={t}/{n_steps} "
                f"(ana={ana[t]:+.4e}, fd={fd[t]:+.4e}); full ana={ana}, fd={fd}"
            ),
        )


# Cartpole MJCF (slider + hinge, multi-body) for the multi-body variant of
# the per-step force FD test. Identical to `examples/diffrl/envs/cartpole.xml`,
# embedded here so the test is self-contained. The cart+pole cross-coupling
# is what the cart-only `MJCF_SLIDER_LIMIT` does *not* probe — the SHAC
# cartpole_swing_up training's grad explosion was traced to this multi-body
# case (project_diffrigid_joint_limit_grad_explosion.md, 2026-05-25).
MJCF_CARTPOLE = """
<mujoco model="cartpole">
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="cart" pos="0 0 0">
      <joint name="slider" type="slide" axis="1 0 0" range="-4 4" damping="0.0"/>
      <inertial pos="0 0 0" mass="1.0" diaginertia="1.0 1.0 1.0"/>
      <geom name="cart_g" type="box" size="0.25 0.25 0.1" contype="0" conaffinity="0"/>
      <body name="pole" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0.0"/>
        <inertial pos="0 0 0.5" mass="10.0" diaginertia="1.0 1.0 1.0"/>
        <geom name="pole_g" type="box" pos="0 0 0.5" size="0.025 0.025 0.5" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _build_cartpole(mjcf_path: str, *, requires_grad: bool):
    """Build a multi-body cart+pole scene with gravity + slider limit on.
    Same options as `examples/diffrl/envs/cartpole_swing_up.py`."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            substeps=4,
            gravity=(0.0, 0.0, -9.81),
            requires_grad=requires_grad,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            enable_self_collision=False,
            enable_joint_limit=True,
            disable_constraint=False,
            use_hibernation=False,
            use_contact_island=False,
        ),
        show_viewer=False,
    )
    robot = scene.add_entity(gs.morphs.MJCF(file=mjcf_path))
    scene.build(n_envs=0)
    return scene, robot


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("n_steps", [15])
def test_diff_joint_limit_backward_fd_per_step_force_cartpole(show_viewer, n_steps):
    """Multi-body variant of `test_diff_joint_limit_backward_fd_per_step_force`.

    Same per-step `control_dofs_force` setup but on the *cart+pole* MJCF
    (slider [-4, 4] + free-rotating hinge with gravity). cart DOF is
    actuated at `force_value`, pole DOF takes 0 force. Loss is `cart_x^2`
    at the terminal step. Both cart_force.grad AND pole_force.grad are
    compared to central FD — pole_force.grad is *non-zero* because a
    hinge torque accelerates the pole, the pole's swing exerts reactive
    horizontal force on the cart via the hinge, which moves cart_x.

    Cart-only test (`MJCF_SLIDER_LIMIT`) is CPU-PASS / GPU-xfail — but
    the SHAC cartpole_swing_up training exhibits grad explosion on
    *both* CPU and GPU. The difference is the cart+pole cross-coupling
    in the backward jacobian, which is what this test probes.
    """
    mjcf_path = _mjcf_tmpfile(MJCF_CARTPOLE)
    eps = 1e-2
    # cart+pole effective mass ≈ 11 (cart 1 + pole 10 horizontal-locked at
    # hanging), so cart force needs to be larger than the cart-only test
    # to reach the limit within `n_steps`.
    force_value = 2000.0
    # Force shape per step: (n_dofs,) = (cart_f, pole_f). pole stays at 0.
    init_force = np.zeros((n_steps, 2), dtype=np.float64)
    init_force[:, 0] = force_value

    # Initial state: cart at x=0, pole hanging down at theta=-pi (same as
    # `CartPoleSwingUpEnv._init_qpos`). Deterministic; same in ana / FD.
    init_qpos = [0.0, -math.pi]

    # Analytical
    scene_ana, robot_ana = _build_cartpole(mjcf_path, requires_grad=True)
    scene_ana.reset()
    robot_ana.set_dofs_position(gs.tensor(init_qpos, dtype=gs.tc_float))
    forces = [gs.tensor(init_force[t], dtype=gs.tc_float, requires_grad=True) for t in range(n_steps)]
    for t in range(n_steps):
        robot_ana.control_dofs_force(forces[t])
        scene_ana.step()
    x_final = float(_rigid_state(scene_ana).qpos[0, 0].detach())
    assert abs(x_final) > 3.5, (
        f"setup error: cart+pole at n_steps={n_steps}, force={force_value} "
        f"did not bring the cart near the limit (x_final={x_final}); pick a "
        f"larger force or more steps."
    )
    loss = (_rigid_state(scene_ana).qpos[0, 0]) ** 2
    loss.backward()
    for t, f in enumerate(forces):
        assert f.grad is not None, f"step {t}: force.grad is None"
    # cart-force grad per step (slot 0); slot 1 is pole-force grad, must be 0.
    ana_cart = np.array([float(f.grad[0]) for f in forces])
    ana_pole = np.array([float(f.grad[1]) for f in forces])

    # FD per-step on the cart-force slot only.
    scene_fd, robot_fd = _build_cartpole(mjcf_path, requires_grad=False)

    def loss_at(perturbed: np.ndarray) -> float:
        scene_fd.reset()
        robot_fd.set_dofs_position(gs.tensor(init_qpos, dtype=gs.tc_float))
        for t in range(n_steps):
            robot_fd.control_dofs_force(gs.tensor(perturbed[t], dtype=gs.tc_float))
            scene_fd.step()
        return float((_rigid_state(scene_fd).qpos[0, 0]) ** 2)

    fd_cart = np.zeros(n_steps)
    fd_pole = np.zeros(n_steps)
    for t in range(n_steps):
        plus = init_force.copy()
        plus[t, 0] += eps
        minus = init_force.copy()
        minus[t, 0] -= eps
        fd_cart[t] = (loss_at(plus) - loss_at(minus)) / (2 * eps)

        plus = init_force.copy()
        plus[t, 1] += eps
        minus = init_force.copy()
        minus[t, 1] -= eps
        fd_pole[t] = (loss_at(plus) - loss_at(minus)) / (2 * eps)

    # Cart-force grad — straight chain from action to cart_x.
    for t in range(n_steps):
        assert_allclose(
            ana_cart[t],
            fd_cart[t],
            rtol=1e-3,
            atol=1e-4,
            err_msg=(
                f"cart-pole cart_force.grad mismatch at t={t}/{n_steps} "
                f"(ana={ana_cart[t]:+.4e}, fd={fd_cart[t]:+.4e}); "
                f"full ana={ana_cart}, fd={fd_cart}"
            ),
        )
    # Pole-force grad — hinge torque chain: pole_force -> pole_angle ->
    # pole COM horizontal accel -> reactive force on cart via hinge ->
    # cart_x. Non-zero, must still match FD step-by-step.
    for t in range(n_steps):
        assert_allclose(
            ana_pole[t],
            fd_pole[t],
            rtol=1e-3,
            atol=1e-4,
            err_msg=(
                f"cart-pole pole_force.grad mismatch at t={t}/{n_steps} "
                f"(ana={ana_pole[t]:+.4e}, fd={fd_pole[t]:+.4e}); "
                f"full ana_pole={ana_pole}, fd_pole={fd_pole}"
            ),
        )


def _build_hopper(mjcf_path: str, *, requires_grad: bool):
    """Build the hopper collision-free with joint limits ON and gravity off.

    Collision is off so the joint-limit constraint is the only constraint in
    play (no foot-ground contact); gravity is off so the base doesn't drift,
    keeping the rollout focused on driving a leg joint into its range limit.
    """
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            substeps=4,
            gravity=(0.0, 0.0, 0.0),
            requires_grad=requires_grad,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            enable_self_collision=False,
            enable_joint_limit=True,
            disable_constraint=False,
            use_hibernation=False,
            use_contact_island=False,
        ),
        show_viewer=False,
    )
    robot = scene.add_entity(gs.morphs.MJCF(file=mjcf_path))
    scene.build(n_envs=0)
    return scene, robot


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("n_steps", [10])
def test_diff_joint_limit_backward_fd_per_step_force_hopper(show_viewer, n_steps):
    """Joint-limit backward FD check on the hopper — combines the multi-joint
    planar base (slide+slide+hinge on the torso) with an *active* joint-limit
    constraint, both collision-free.

    A constant torque on the foot joint drives it past its `[-0.785, 0.785]`
    range during the rollout, engaging the joint-limit inequality constraint.
    The loss is on the foot LINK world position (so the gradient flows through
    the full FK chain — exercising `kernel_manual_fk_only_bw`'s multi-joint
    reverse for the base — as well as the constraint backward). Every DOF's
    per-step `control_dofs_force.grad` is compared to central FD: the foot DOF
    is the forced + limited one; the other DOFs are reached only through the
    articulated coupling, so their FD sensitivity is non-trivial too.
    """
    mjcf_path = _mjcf_tmpfile(MJCF_HOPPER)
    n_dofs = 6  # rootx, rootz, rooty, thigh, leg, foot
    foot_dof = 5
    eps = 1e-2
    force_value = 200.0
    init_force = np.zeros((n_steps, n_dofs), dtype=np.float64)
    init_force[:, foot_dof] = force_value

    def _links_pos_sq_loss(scene):
        lp = _rigid_state(scene).links_pos
        return (lp.reshape(-1) ** 2).sum()

    # Analytical
    scene_ana, robot_ana = _build_hopper(mjcf_path, requires_grad=True)
    scene_ana.reset()
    forces = [gs.tensor(init_force[t], dtype=gs.tc_float, requires_grad=True) for t in range(n_steps)]
    for t in range(n_steps):
        robot_ana.control_dofs_force(forces[t])
        scene_ana.step()
    foot_q = float(_rigid_state(scene_ana).qpos[0, foot_dof].detach())
    # Setup sanity: the foot must have entered its limit band, else the
    # constraint backward isn't exercised.
    assert abs(foot_q) > 0.7, (
        f"setup error: n_steps={n_steps} at foot force={force_value} did not "
        f"drive the foot joint near its 0.785 limit (foot_q={foot_q}); pick a "
        f"larger force or more steps."
    )
    loss = _links_pos_sq_loss(scene_ana)
    loss.backward()
    for t, f in enumerate(forces):
        assert f.grad is not None, f"step {t}: force.grad is None"
    ana = np.array([[float(f.grad[d]) for d in range(n_dofs)] for f in forces])  # (n_steps, n_dofs)

    # FD per-step, per-dof
    scene_fd, robot_fd = _build_hopper(mjcf_path, requires_grad=False)

    def loss_at(perturbed: np.ndarray) -> float:
        scene_fd.reset()
        for t in range(n_steps):
            robot_fd.control_dofs_force(gs.tensor(perturbed[t], dtype=gs.tc_float))
            scene_fd.step()
        return float(_links_pos_sq_loss(scene_fd).detach())

    fd = np.zeros((n_steps, n_dofs))
    for t in range(n_steps):
        for d in range(n_dofs):
            plus = init_force.copy()
            plus[t, d] += eps
            minus = init_force.copy()
            minus[t, d] -= eps
            fd[t, d] = (loss_at(plus) - loss_at(minus)) / (2 * eps)

    for t in range(n_steps):
        for d in range(n_dofs):
            assert_allclose(
                ana[t, d],
                fd[t, d],
                rtol=1e-3,
                atol=1e-4,
                err_msg=(
                    f"hopper force.grad mismatch at t={t}/{n_steps}, dof={d} "
                    f"(ana={ana[t, d]:+.4e}, fd={fd[t, d]:+.4e})\nfull ana=\n{ana}\nfull fd=\n{fd}"
                ),
            )
