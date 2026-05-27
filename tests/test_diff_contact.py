"""FD vs analytical gradient for the rigid solver's *contact* backward path.

Within the differentiable-rigid test suite, this is the collision layer — the
only file that puts a contact in the backward chain. Siblings isolate the rest:
  - test_diff_forward_kinematics : unconstrained FK + velocity gradient (no
      contact / no constraints) — the base local-gradient bar.
  - test_diff_joint_limit        : the joint-limit inequality constraint.
  - test_diff_contact            : *this file* — the collision constraint reverse
      (`kernel_manual_add_collision_constraints_bw`) + diff-GJK narrow-phase
      reverse (`collider.backward`), wired into `substep_pre_coupling_grad`.
  - test_diff_scene_backward     : the scene.backward() API + horizon truncation.
  - test_diff_optim              : end-to-end optimization convergence.

A per-step `control_dofs_force` on a free convex resting on a fixed collider,
with the gradient flowing
    force -> qacc -> constraint solve -> contact-constraint reverse
          -> contact_data grads -> diff-GJK narrow-phase reverse -> geom poses
          -> FK -> qpos
checked against central finite differences. Two contact families:
  * box-box (general GJK diff path, `box_box_detection=False`) — the same diff
    narrow-phase that `test_grad.py::test_diff_contact` validates in isolation.
    CPU only: the GPU split narrow phase drops it under requires_grad (skipped).
  * plane-convex (plane + box / sphere / capsule) — the analytic plane paths
    don't fill `diff_contact_input`, so it is reconstructed differentiably via
    `func_differentiable_plane_contact` from the stored convex support core.
    CPU + GPU (this path sidesteps the split narrow phase).

IMPORTANT — contact-pair preservation: the diff-GJK gradient assumes the contact
*set* (which pairs, how many) is fixed; an FD perturbation that adds/removes a
contact injects a discontinuity the smooth gradient can't capture, making FD
invalid. The scenarios settle into a stable, persistent contact and keep the FD
step small enough to leave `n_contacts` unchanged — asserted across base/±eps.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

import genesis as gs

from .utils import assert_allclose


def _build_box_box(*, requires_grad: bool):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=2,
            gravity=(0.0, 0.0, -9.81),
            requires_grad=requires_grad,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            enable_self_collision=False,
            enable_joint_limit=False,
            disable_constraint=False,
            use_hibernation=False,
            use_contact_island=False,
            box_box_detection=False,  # general convex-convex GJK (differentiable) path
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Box(size=(2.0, 2.0, 0.2), pos=(0.0, 0.0, 0.1), fixed=True))
    box = scene.add_entity(gs.morphs.Box(size=(0.4, 0.4, 0.4), pos=(0.0, 0.0, 0.4)))
    scene.build(n_envs=0)
    return scene, box


def _make_capsule_mjcf(tmp_path, radius: float, half_length: float) -> str:
    mjcf = ET.Element("mujoco", model="capsule")
    ET.SubElement(mjcf, "compiler", angle="degree")
    worldbody = ET.SubElement(mjcf, "worldbody")
    body = ET.SubElement(worldbody, "body", name="capsule", pos="0 0 0")
    ET.SubElement(body, "geom", type="capsule", size=f"{radius} {half_length}")
    ET.SubElement(body, "joint", name="capsule_joint", type="free")
    path = tmp_path / "capsule.xml"
    ET.ElementTree(mjcf).write(path)
    return str(path)


def _build_plane_convex(shape: str, tmp_path, *, requires_grad: bool):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=2,
            gravity=(0.0, 0.0, -9.81),
            requires_grad=requires_grad,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            enable_self_collision=False,
            enable_joint_limit=False,
            disable_constraint=False,
            use_hibernation=False,
            use_contact_island=False,
            box_box_detection=False,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    if shape == "box":
        obj = scene.add_entity(gs.morphs.Box(size=(0.4, 0.4, 0.4), pos=(0.0, 0.0, 0.3)))
    elif shape == "sphere":
        obj = scene.add_entity(gs.morphs.Sphere(radius=0.2, pos=(0.0, 0.0, 0.3)))
    elif shape == "capsule":
        obj = scene.add_entity(gs.morphs.MJCF(file=_make_capsule_mjcf(tmp_path, 0.1, 0.2), align=False))
    else:
        raise ValueError(shape)
    scene.build(n_envs=0)
    return scene, obj


def _rigid_state(scene):
    return scene.get_state().solvers_state[scene.solvers.index(scene.rigid_solver)]


def _n_contacts(scene) -> int:
    return int(scene.rigid_solver.collider._collider_state.n_contacts.to_numpy()[0])


def _settle(scene, obj, n_settle: int):
    zero = gs.tensor([0.0] * 6, dtype=gs.tc_float)
    for _ in range(n_settle):
        obj.control_dofs_force(zero)
        scene.step()


def _run_fd_per_step_force(build_fn, rest_dofs, *, base_force, n_settle, n_steps, fd_dofs, eps, rtol, atol):
    """Analytical-vs-central-FD driver for a free convex pressed into a fixed
    collider by a per-step downward force.

    The force is purely downward/centered: a lateral or torque component would
    tip the body and change the contact manifold (breaking contact-pair
    preservation), so only the load-bearing z DOF is FD-checked. Its gradient
    still runs through contact_pos / normal / penetration inside the constraint
    reverse and the differentiable narrow phase. The FD scene runs in
    `requires_grad=True` (forward only) so it produces the *same* contact
    manifold as the analytical scene; a production-mode scene would take a
    different (non-diff) narrow-phase path with a different contact set.
    """
    init_force = np.broadcast_to(base_force, (n_steps, 6)).copy()

    # --- analytical ---
    scene_ana, obj_ana = build_fn(requires_grad=True)
    scene_ana.reset()
    obj_ana.set_dofs_position(gs.tensor(rest_dofs, dtype=gs.tc_float).sceneless())
    _settle(scene_ana, obj_ana, n_settle)
    nc = _n_contacts(scene_ana)
    assert nc > 0, f"setup error: not in contact after settle (n_contacts={nc})"

    forces = [gs.tensor(init_force[t], dtype=gs.tc_float, requires_grad=True) for t in range(n_steps)]
    for t in range(n_steps):
        obj_ana.control_dofs_force(forces[t])
        scene_ana.step()
        assert _n_contacts(scene_ana) == nc, "contact set changed during grad window — FD invalid"
    loss = (_rigid_state(scene_ana).qpos[0, :3] ** 2).sum()
    scene_ana.backward(loss)
    ana = np.array([[float(f.grad[d]) for d in range(6)] for f in forces])  # (N, 6)

    # --- central FD, contact set preserved ---
    scene_fd, obj_fd = build_fn(requires_grad=True)

    def loss_at(perturbed: np.ndarray) -> float:
        scene_fd.reset()
        obj_fd.set_dofs_position(gs.tensor(rest_dofs, dtype=gs.tc_float).sceneless())
        _settle(scene_fd, obj_fd, n_settle)
        for t in range(n_steps):
            obj_fd.control_dofs_force(gs.tensor(perturbed[t], dtype=gs.tc_float))
            scene_fd.step()
            assert _n_contacts(scene_fd) == nc, "contact set changed under FD perturbation"
        return float((_rigid_state(scene_fd).qpos[0, :3] ** 2).sum().detach())

    fd = np.full((n_steps, 6), np.nan)
    for t in range(n_steps):
        for d in fd_dofs:
            plus = init_force.copy()
            plus[t, d] += eps
            minus = init_force.copy()
            minus[t, d] -= eps
            fd[t, d] = (loss_at(plus) - loss_at(minus)) / (2 * eps)

    # Contact gradients are small (stiff contact barely moves), so the band is
    # absolute-dominated; rtol pins the load-bearing z entry.
    for t in range(n_steps):
        for d in fd_dofs:
            assert_allclose(
                ana[t, d],
                fd[t, d],
                rtol=rtol,
                atol=atol,
                err_msg=f"contact force.grad mismatch at t={t}/{n_steps}, dof={d}\nana=\n{ana}\nfd=\n{fd}",
            )


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize(
    "backend",
    [
        gs.cpu,
        pytest.param(
            gs.gpu,
            marks=pytest.mark.skip(
                reason="General convex-convex (box-box) differentiable contact is not supported on GPU: the "
                "GPU split (multicontact) narrow phase drops contacts under requires_grad=True (n_contacts=0). "
                "Only plane-convex is GPU-differentiable (see test_diff_contact_fd_plane_convex). "
                "Revisit when the split path's diff-contact handling is fixed."
            ),
        ),
    ],
)
def test_diff_contact_fd_per_step_force(show_viewer):
    # Box rests on the ground top (z=0.2) at center z=0.40; settle to a stable
    # multi-contact manifold, then a short grad window with a per-step push.
    _run_fd_per_step_force(
        _build_box_box,
        [0.0, 0.0, 0.40, 0.0, 0.0, 0.0],  # freejoint 6 DOFs: xyz + rotvec(=identity)
        base_force=np.array([0.0, 0.0, -8.0, 0.0, 0.0, 0.0], dtype=np.float64),
        n_settle=12,
        n_steps=2,
        fd_dofs=(2,),
        eps=1e-2,
        rtol=2e-3,
        atol=1e-10,
    )


# rest z so the body's lowest point sits on the plane (z=0): box/sphere half
# extent 0.2; capsule radius 0.1 + half_length 0.2 = 0.3 (upright).
_PLANE_REST = {
    "box": [0.0, 0.0, 0.20, 0.0, 0.0, 0.0],
    "sphere": [0.0, 0.0, 0.20, 0.0, 0.0, 0.0],
    "capsule": [0.0, 0.0, 0.30, 0.0, 0.0, 0.0],
}


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("shape", ["box", "sphere", "capsule"])
def test_diff_contact_fd_plane_convex(shape, tmp_path, show_viewer):
    # Plane (fixed) + free convex. The analytic plane contact is reconstructed
    # differentiably via `func_differentiable_plane_contact` (stored convex
    # support core + radius), so the same FD chain as box-box applies.
    _run_fd_per_step_force(
        lambda *, requires_grad: _build_plane_convex(shape, tmp_path, requires_grad=requires_grad),
        _PLANE_REST[shape],
        base_force=np.array([0.0, 0.0, -8.0, 0.0, 0.0, 0.0], dtype=np.float64),
        n_settle=12,
        n_steps=2,
        fd_dofs=(2,),
        eps=1e-2,
        rtol=2e-3,
        atol=1e-10,
    )
