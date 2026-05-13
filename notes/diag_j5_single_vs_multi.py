"""J5 single-step vs N=2: compare residual .grad fields after backward.

If a field has non-zero .grad after a single-step backward, the forward
chain involving that field has a silently-dropped reverse contribution
(chain didn't fully propagate to qpos.grad). Multi-step amplifies the
silent loss by atomic-adding into the same field across substeps.

J5 is a 3-revolute chain (no free joint), so isolates the multi-link
Coriolis / composite-inertia chain from the freejoint quat<->rotvec.
"""

import os
import sys
import tempfile

import numpy as np
import torch

import genesis as gs
from genesis.utils.misc import qd_to_torch


MJCF_J5 = """<mujoco model="chain3">
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
        f.write(MJCF_J5)
    robot = scene.add_entity(gs.morphs.MJCF(file=p))
    scene.build(n_envs=0)
    return scene, robot


def loss_fn(scene):
    state = scene.get_state()
    solver_state = state.solvers_state[scene.solvers.index(scene.rigid_solver)]
    flat = solver_state.links_pos.reshape(-1)
    target = torch.zeros_like(flat)
    return ((flat - target) ** 2).sum()


def dump_nonzero(solver, label):
    print(f"\n=== {label} ===")
    for struct_name in ("dofs_state", "links_state", "joints_state"):
        s = getattr(solver, struct_name)
        cls = type(s)
        try:
            anns = cls.__dict__["__annotations__"]
        except KeyError:
            continue
        for attr in anns:
            v = getattr(s, attr, None)
            if v is None:
                continue
            try:
                t = qd_to_torch(v.grad, copy=True)
                m = float(t.abs().max())
                if m > 1e-12:
                    print(f"  {struct_name}.{attr:25s}  max={m:.3e}  norm={float(t.norm()):.3e}")
            except Exception:
                continue


def run(N):
    print(f"\n############ N={N} ############")
    scene, robot = build_scene(requires_grad=True)
    rng = np.random.default_rng(42)
    u_list = [rng.normal(size=3) * 0.3 for _ in range(N)]
    u_anas = [gs.tensor(u, dtype=gs.tc_float, requires_grad=True) for u in u_list]
    scene.reset()
    for t in range(N):
        robot.control_dofs_force(u_anas[t])
        scene.step()
    loss = loss_fn(scene)

    solver = scene.rigid_solver
    orig = solver.substep_pre_coupling_grad

    def wrapped(f):
        result = orig(f)
        dump_nonzero(solver, f"end of substep f={f}, global={solver.sim.cur_substep_global}")
        return result

    solver.substep_pre_coupling_grad = wrapped
    loss.backward()

    print("\n--- ana_grads ---")
    for t, u in enumerate(u_anas):
        print(f"  step {t}: {u.grad.detach().cpu().numpy()}")


def main():
    gs.init(precision="64", backend=gs.cpu, performance_mode=False, logging_level="warning")
    run(1)
    run(2)


if __name__ == "__main__":
    main()
