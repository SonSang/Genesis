"""Diagnostic: N=2 control_dofs_force, dump dofs_state.grad at each backward substep.

Compares ana_grad[0] and ana_grad[1] against the simple FD computation:
    loss = (q_after_step_2 - target)^2

If `ana[0] = FD[0] + FD[1]`, the leak field is whatever has non-zero .grad
*before* substep f=0's BW kernels run (because f=1's BW left it set, and
process_input_grad only drains ctrl_force.grad — not intermediates).
"""

import os
import tempfile

import numpy as np
import torch

import genesis as gs

os.environ.setdefault("GENESIS_DEBUG_GRAD", "1")

MJCF = """<mujoco model="prismatic">
  <option timestep="0.01" gravity="0 0 0"/>
  <worldbody>
    <body name="b" pos="0 0 0">
      <joint name="j" type="slide" axis="1 0 0"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def build_scene(requires_grad: bool):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, 0), requires_grad=requires_grad),
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
    fd, p = tempfile.mkstemp(suffix=".xml")
    with os.fdopen(fd, "w") as f:
        f.write(MJCF)
    robot = scene.add_entity(gs.morphs.MJCF(file=p))
    scene.build(n_envs=0)
    return scene, robot


def loss_fn(scene, robot):
    target = torch.zeros(3)
    return ((robot.get_state().pos.reshape(-1) - target) ** 2).sum()


def main():
    gs.init(precision="64", backend=gs.cpu, performance_mode=False, logging_level="info")
    N = 2

    # --- analytical
    scene_ana, robot_ana = build_scene(requires_grad=True)
    u0 = gs.tensor(np.array([0.7]), dtype=gs.tc_float, requires_grad=True)
    u1 = gs.tensor(np.array([-0.3]), dtype=gs.tc_float, requires_grad=True)
    scene_ana.reset()
    print("\n=== FORWARD STEP 0 (u0=0.7) ===")
    robot_ana.control_dofs_force(u0)
    scene_ana.step()
    print("\n=== FORWARD STEP 1 (u1=-0.3) ===")
    robot_ana.control_dofs_force(u1)
    scene_ana.step()
    print("\n=== BACKWARD ===")
    loss = loss_fn(scene_ana, robot_ana)
    loss.backward()
    print(f"\nana_grad[0] (u0={u0.detach().item()}): {u0.grad.item():+.6e}")
    print(f"ana_grad[1] (u1={u1.detach().item()}): {u1.grad.item():+.6e}")

    # --- FD
    scene_fd, robot_fd = build_scene(requires_grad=False)
    eps = 1e-5

    def run(u0_v, u1_v):
        scene_fd.reset()
        robot_fd.control_dofs_force(gs.tensor(np.array([u0_v]), dtype=gs.tc_float))
        scene_fd.step()
        robot_fd.control_dofs_force(gs.tensor(np.array([u1_v]), dtype=gs.tc_float))
        scene_fd.step()
        return float(loss_fn(scene_fd, robot_fd).detach().cpu())

    fd0 = (run(0.7 + eps, -0.3) - run(0.7 - eps, -0.3)) / (2 * eps)
    fd1 = (run(0.7, -0.3 + eps) - run(0.7, -0.3 - eps)) / (2 * eps)
    print(f"\nfd_grad[0] = {fd0:+.6e}")
    print(f"fd_grad[1] = {fd1:+.6e}")

    print(f"\nratio[0] = {u0.grad.item() / fd0:.4f}  (expect ~1.5 if `ana[0] = FD[0] + FD[1]`, =1 if correct)")
    print(f"ratio[1] = {u1.grad.item() / fd1:.4f}  (expect ~1.0)")
    print(f"\nFD[0] + FD[1] = {fd0 + fd1:+.6e}")
    print(f"ana[0]        = {u0.grad.item():+.6e}  (match if leak hypothesis correct)")


if __name__ == "__main__":
    main()
