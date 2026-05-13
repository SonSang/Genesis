"""Localize the dominant J5 mismatch: per-step, per-DOF ana vs FD diff.

For each step t (0..N-1) and each DOF d, compute:
  ana[t][d] from one big loss.backward()
  FD[t][d] via central FD on u_t[d]
Print sorted by absolute mismatch.

Goal: find the (step, DOF) where the leak is concentrated. That points to
the forward chain segment responsible (e.g. if mismatch is mostly in
later-step DOFs that participate in `func_forward_velocity`, we know to
look at cd_*/cdofd_* chain).
"""

import os
import tempfile
import numpy as np
import torch
import genesis as gs


MJCF = """<mujoco model="chain3">
  <worldbody>
    <body name="l1" pos="0 0 0">
      <joint type="hinge" axis="0 1 0"/>
      <inertial mass="0.3" pos="0.1 0 0" diaginertia="0.005 0.005 0.005"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
      <body name="l2" pos="0.2 0 0">
        <joint type="hinge" axis="0 1 0"/>
        <inertial mass="0.3" pos="0.1 0 0" diaginertia="0.005 0.005 0.005"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
        <body name="l3" pos="0.2 0 0">
          <joint type="hinge" axis="0 1 0"/>
          <inertial mass="0.3" pos="0.1 0 0" diaginertia="0.005 0.005 0.005"/>
          <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def build_scene(requires_grad):
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


def loss_fn(scene):
    state = scene.get_state()
    ss = state.solvers_state[scene.solvers.index(scene.rigid_solver)]
    return (ss.links_pos.reshape(-1) ** 2).sum()


def main():
    gs.init(precision="64", backend=gs.cpu, performance_mode=False, logging_level="warning")
    N = 10
    n_dofs = 3
    rng = np.random.default_rng(165 * 100)  # seed matches test fixture
    u_list = [rng.normal(size=n_dofs) * 0.3 for _ in range(N)]

    # ana
    scene_a, robot_a = build_scene(requires_grad=True)
    u_anas = [gs.tensor(u, dtype=gs.tc_float, requires_grad=True) for u in u_list]
    scene_a.reset()
    for t in range(N):
        robot_a.control_dofs_force(u_anas[t])
        scene_a.step()
    loss_fn(scene_a).backward()
    ana = np.array([u.grad.detach().cpu().numpy() for u in u_anas])

    # FD
    scene_b, robot_b = build_scene(requires_grad=False)
    eps = 1e-5

    def run(perturb_t, perturb_d, sign):
        scene_b.reset()
        for t in range(N):
            inp = u_list[t].copy()
            if t == perturb_t:
                inp[perturb_d] += sign * eps
            robot_b.control_dofs_force(gs.tensor(inp, dtype=gs.tc_float))
            scene_b.step()
        return float(loss_fn(scene_b).detach().cpu())

    fd = np.zeros((N, n_dofs))
    for t in range(N):
        for d in range(n_dofs):
            lp = run(t, d, +1)
            lm = run(t, d, -1)
            fd[t, d] = (lp - lm) / (2 * eps)

    diff = ana - fd
    abs_diff = np.abs(diff)

    # Sort by abs diff, print top entries
    entries = [(t, d, ana[t, d], fd[t, d], diff[t, d], abs_diff[t, d]) for t in range(N) for d in range(n_dofs)]
    entries.sort(key=lambda x: -x[5])

    print("\nTop 10 (step, dof) mismatches:")
    print(f"{'step':>4} {'dof':>3} {'ana':>14} {'fd':>14} {'diff':>14} {'|diff|':>14} {'rel':>8}")
    for t, d, a, f, df, ad in entries[:10]:
        rel = abs(df) / max(abs(f), 1e-15)
        print(f"{t:>4} {d:>3} {a:>14.6e} {f:>14.6e} {df:>+14.6e} {ad:>14.6e} {rel:>8.4f}")

    print("\nPer-step max abs diff:")
    for t in range(N):
        print(
            f"  step {t}: max |diff| = {abs_diff[t].max():.3e}, "
            f"max |ana| = {np.abs(ana[t]).max():.3e}, "
            f"max |fd| = {np.abs(fd[t]).max():.3e}"
        )


if __name__ == "__main__":
    main()
