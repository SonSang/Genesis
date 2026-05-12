"""Dump ALL .grad-bearing fields of dofs_state, links_state, joints_state at
the end of each backward substep — find every field that carries stale grad.

This is what `RigidSolver._debug_grad_dump` does but with a static hand-picked
field list. Here we use `dataclass __annotations__` to walk every slot of
every state struct.
"""

import os
import sys
import tempfile

import numpy as np
import torch

import genesis as gs
from genesis.utils.misc import qd_to_torch


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


def loss_fn(scene):
    state = scene.get_state()
    solver_state = state.solvers_state[scene.solvers.index(scene.rigid_solver)]
    flat = solver_state.links_pos.reshape(-1)
    target = torch.zeros_like(flat)
    return ((flat - target) ** 2).sum()


def dump_state_grads(state_obj, name):
    cls = type(state_obj)
    try:
        annotations = cls.__dict__["__annotations__"]
    except KeyError:
        return
    nonzero = []
    for attr in annotations:
        val = getattr(state_obj, attr, None)
        if val is None:
            continue
        try:
            grad = val.grad
            t = qd_to_torch(grad, copy=True)
            m = float(t.abs().max())
            if m > 0:
                nonzero.append((attr, m, float(t.norm())))
        except Exception:
            continue
    print(f"  {name}:")
    for a, mx, nm in nonzero:
        print(f"    {a:30s}  max={mx:.3e}  norm={nm:.3e}")


def main():
    gs.init(precision="64", backend=gs.cpu, performance_mode=False, logging_level="warning")
    scene, robot = build_scene(requires_grad=True)
    rng = np.random.default_rng(42)
    u_list = [rng.normal(size=7) * 0.3 for _ in range(2)]
    u_anas = [gs.tensor(u, dtype=gs.tc_float, requires_grad=True) for u in u_list]
    scene.reset()
    for t in range(2):
        robot.control_dofs_force(u_anas[t])
        scene.step()
    loss = loss_fn(scene)

    # Monkey-patch substep_pre_coupling_grad to dump grads at exit
    solver = scene.rigid_solver
    orig = solver.substep_pre_coupling_grad

    def wrapped(f):
        result = orig(f)
        print(f"\n=== END of backward substep f={f} (cur_substep_global={solver.sim.cur_substep_global}) ===")
        dump_state_grads(solver.dofs_state, "dofs_state")
        dump_state_grads(solver.links_state, "links_state")
        dump_state_grads(solver.joints_state, "joints_state")
        dump_state_grads(solver.geoms_state, "geoms_state")
        return result

    solver.substep_pre_coupling_grad = wrapped
    loss.backward()


if __name__ == "__main__":
    main()
