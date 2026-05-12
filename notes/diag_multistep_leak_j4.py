"""Diagnostic for J4 (free+revolute) multistep leak.

After fixing `acc_smooth_bw.grad`, J1/J2/J3 pass but J4/J5 (multi-link) still
fail on the rotation DOFs. This script does N=2 forward + backward and
dumps all .grad fields at each backward checkpoint to identify which
links_state/joints_state field carries stale grad between substeps.
"""

import os
import tempfile

import numpy as np
import torch

import genesis as gs

os.environ.setdefault("GENESIS_DEBUG_GRAD", "1")

MJCF_J4 = """<mujoco model="free_rev">
  <option timestep="0.01" gravity="0 0 0"/>
  <worldbody>
    <body name="base" pos="0 0 0">
      <joint name="root" type="free"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
      <body name="child" pos="0.2 0 0">
        <joint name="rev" type="hinge" axis="0 0 1"/>
        <geom type="box" size="0.1 0.1 0.1" mass="0.5"/>
      </body>
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
        f.write(MJCF_J4)
    robot = scene.add_entity(gs.morphs.MJCF(file=p))
    scene.build(n_envs=0)
    return scene, robot


def loss_fn(scene, robot):
    state = scene.get_state()
    solver_state = state.solvers_state[scene.solvers.index(scene.rigid_solver)]
    flat = solver_state.links_pos.reshape(-1)
    target = torch.zeros_like(flat)
    return ((flat - target) ** 2).sum()


def main():
    gs.init(precision="64", backend=gs.cpu, performance_mode=False, logging_level="info")

    # --- analytical
    scene_ana, robot_ana = build_scene(requires_grad=True)
    rng = np.random.default_rng(42)
    u_list = [rng.normal(size=7) * 0.3 for _ in range(2)]
    u_anas = [gs.tensor(u, dtype=gs.tc_float, requires_grad=True) for u in u_list]
    scene_ana.reset()
    for t in range(2):
        print(f"\n=== FORWARD STEP {t} ===")
        robot_ana.control_dofs_force(u_anas[t])
        scene_ana.step()
    print("\n=== BACKWARD ===")
    loss = loss_fn(scene_ana, robot_ana)
    loss.backward()
    for t, u in enumerate(u_anas):
        print(f"\nana_grad[{t}]: {u.grad.detach().cpu().numpy()}")

    # --- FD
    scene_fd, robot_fd = build_scene(requires_grad=False)
    eps = 1e-5

    def run(perturb_t, perturb_i, sign):
        scene_fd.reset()
        for t in range(2):
            inp = u_list[t].copy()
            if t == perturb_t:
                inp[perturb_i] += sign * eps
            robot_fd.control_dofs_force(gs.tensor(inp, dtype=gs.tc_float))
            scene_fd.step()
        return float(loss_fn(scene_fd, robot_fd).detach().cpu())

    fd_grads = [np.zeros(7) for _ in range(2)]
    for t in range(2):
        for i in range(7):
            lp = run(t, i, +1)
            lm = run(t, i, -1)
            fd_grads[t][i] = (lp - lm) / (2 * eps)
    for t in range(2):
        print(f"\nfd_grad[{t}]: {fd_grads[t]}")
        print(f"diff ana-fd[{t}]: {u_anas[t].grad.detach().cpu().numpy() - fd_grads[t]}")


if __name__ == "__main__":
    main()
