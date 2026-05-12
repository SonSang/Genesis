"""Multi-seed × multi-N stress test for J1~J5 multistep `control_dofs_force` gradient.

Measures worst-case gradient error across many seeds and horizons. Goal:
once we know which (topology, N, seed) configurations break, we can fix
the worst case rather than chase one-off unit-test failures.

Output: per-topology table of (N, max abs diff, max rel, 95-percentile rel)
plus the seed that hit the worst case.
"""

import os
import tempfile

import numpy as np
import torch

import genesis as gs


# Shared MJCF fixtures (match the unit-test ones).
MJCF_J1 = """<mujoco>
  <worldbody>
    <body name="b" pos="0 0 0">
      <freejoint/>
      <inertial mass="1.0" pos="0 0 0" diaginertia="0.1 0.1 0.1"/>
      <geom type="box" size="0.1 0.1 0.1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>"""

MJCF_J2 = """<mujoco>
  <worldbody>
    <body name="b" pos="0 0 0">
      <joint type="hinge" axis="0 1 0"/>
      <inertial mass="0.5" pos="0.1 0 0" diaginertia="0.01 0.01 0.01"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>"""

MJCF_J3 = """<mujoco>
  <worldbody>
    <body name="b" pos="0 0 0">
      <joint type="slide" axis="1 0 0"/>
      <inertial mass="0.5" pos="0 0 0" diaginertia="0.01 0.01 0.01"/>
      <geom type="box" size="0.05 0.05 0.05" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>"""

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

MJCF_J5 = """<mujoco>
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
</mujoco>"""


TOPOLOGIES = [
    ("J1_free", MJCF_J1, 6),
    ("J2_revolute", MJCF_J2, 1),
    ("J3_prismatic", MJCF_J3, 1),
    ("J4_free_rev", MJCF_J4, 7),
    ("J5_chain3", MJCF_J5, 3),
]


def build(mjcf, requires_grad):
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
        f.write(mjcf)
    robot = scene.add_entity(gs.morphs.MJCF(file=p))
    scene.build(n_envs=0)
    return scene, robot


def loss_fn(scene):
    state = scene.get_state()
    ss = state.solvers_state[scene.solvers.index(scene.rigid_solver)]
    return (ss.links_pos.reshape(-1) ** 2).sum()


def measure(mjcf, n_dofs, N, seed):
    """Returns (ana_grads, fd_grads) — both shape (N, n_dofs)."""
    sa, ra = build(mjcf, True)
    rng = np.random.default_rng(seed)
    u_list = [rng.normal(size=n_dofs) * 0.3 for _ in range(N)]
    u_anas = [gs.tensor(u, dtype=gs.tc_float, requires_grad=True) for u in u_list]
    sa.reset()
    for t in range(N):
        ra.control_dofs_force(u_anas[t])
        sa.step()
    loss_fn(sa).backward()
    ana = np.array([u.grad.detach().cpu().numpy() for u in u_anas])

    sb, rb = build(mjcf, False)
    eps = 1e-5
    fd = np.zeros((N, n_dofs))
    for t in range(N):
        for d in range(n_dofs):
            sb.reset()
            for t2 in range(N):
                inp = u_list[t2].copy()
                if t2 == t:
                    inp[d] += eps
                rb.control_dofs_force(gs.tensor(inp, dtype=gs.tc_float))
                sb.step()
            lp = float(loss_fn(sb).detach().cpu())
            sb.reset()
            for t2 in range(N):
                inp = u_list[t2].copy()
                if t2 == t:
                    inp[d] -= eps
                rb.control_dofs_force(gs.tensor(inp, dtype=gs.tc_float))
                sb.step()
            lm = float(loss_fn(sb).detach().cpu())
            fd[t, d] = (lp - lm) / (2 * eps)
    return ana, fd


def summarize(name, N, ana_list, fd_list, seeds):
    """Aggregate across seeds. Returns dict with worst-case info."""
    diffs = []
    rels = []
    worst_seed = seeds[0] if seeds else -1
    worst_abs = -1.0
    for ana, fd, seed in zip(ana_list, fd_list, seeds):
        d = np.abs(ana - fd)
        r = d / np.maximum(np.abs(fd), 1e-15)
        diffs.append(d)
        rels.append(r)
        if d.max() > worst_abs:
            worst_abs = d.max()
            worst_seed = seed
    all_diffs = np.concatenate([d.flatten() for d in diffs])
    all_rels = np.concatenate([r.flatten() for r in rels])
    return dict(
        name=name,
        N=N,
        max_abs=float(all_diffs.max()),
        p95_abs=float(np.percentile(all_diffs, 95)),
        median_abs=float(np.median(all_diffs)),
        max_rel=float(all_rels.max()),
        p95_rel=float(np.percentile(all_rels, 95)),
        median_rel=float(np.median(all_rels)),
        worst_seed=worst_seed,
    )


def main():
    gs.init(precision="64", backend=gs.cpu, performance_mode=False, logging_level="warning")
    n_seeds = 3  # tune for runtime vs coverage
    N_list = [4, 16, 32]

    print(
        f"\n{'topology':<14} {'N':>3} {'max|diff|':>11} {'p95|diff|':>11} "
        f"{'med|diff|':>11} {'max_rel':>9} {'p95_rel':>9} {'med_rel':>9} {'worst_seed':>11}"
    )
    print("-" * 105)
    for name, mjcf, n_dofs in TOPOLOGIES:
        for N in N_list:
            ana_list = []
            fd_list = []
            seeds = list(range(1000, 1000 + n_seeds))
            for seed in seeds:
                try:
                    ana, fd = measure(mjcf, n_dofs, N, seed)
                except Exception as e:
                    print(f"  {name} N={N} seed={seed}: ERROR {e}")
                    continue
                ana_list.append(ana)
                fd_list.append(fd)
            if not ana_list:
                continue
            s = summarize(name, N, ana_list, fd_list, seeds)
            print(
                f"{s['name']:<14} {s['N']:>3} "
                f"{s['max_abs']:>11.3e} {s['p95_abs']:>11.3e} {s['median_abs']:>11.3e} "
                f"{s['max_rel']:>9.3f} {s['p95_rel']:>9.3f} {s['median_rel']:>9.3f} "
                f"{s['worst_seed']:>11}"
            )


if __name__ == "__main__":
    main()
