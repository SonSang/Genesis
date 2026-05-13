"""J4 multistep gradient dump — runs N=2 control_dofs_force backward and
captures `_debug_grad_dump` output across all stages. Goal: identify
where in the backward chain a `.grad` field unexpectedly collapses
(silent drop) vs stays load-bearing.

Run with:
    GENESIS_DEBUG_GRAD=1 python notes/diag_j4_grad_dump.py 2>&1 | tee notes/diag_j4_grad_dump.txt

Output rows look like:
    [tag] field_name: abs_max=X  L2=Y
We scan post-hoc for fields whose abs_max drops to 0 between adjacent tags,
or whose magnitude is suspiciously small relative to neighbours.
"""

import os
import sys
import tempfile

import numpy as np
import torch

import genesis as gs


MJCF_J4 = """<mujoco>
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
</mujoco>"""


def build(requires_grad):
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


def loss_fn(scene):
    state = scene.get_state()
    ss = state.solvers_state[scene.solvers.index(scene.rigid_solver)]
    return (ss.links_pos.reshape(-1) ** 2).sum()


def main():
    gs.init(precision="64", backend=gs.cpu, performance_mode=False, logging_level="info")
    N = 1
    n_dofs = 7
    seed = 1001  # worst seed at N=1

    rng = np.random.default_rng(seed)
    u_list = [rng.normal(size=n_dofs) * 0.3 for _ in range(N)]
    u_tensors = [gs.tensor(u, dtype=gs.tc_float, requires_grad=True) for u in u_list]

    scene, robot = build(True)
    scene.reset()
    for t in range(N):
        robot.control_dofs_force(u_tensors[t])
        scene.step()
    loss = loss_fn(scene)
    print(f"\n[loss] = {float(loss.detach().cpu()):.6e}\n")
    loss.backward()

    for t, u in enumerate(u_tensors):
        g = u.grad.detach().cpu().numpy()
        print(f"u[{t}].grad = {g}")


if __name__ == "__main__":
    main()
