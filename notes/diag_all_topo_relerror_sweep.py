"""J1~J5 — rel error sweep across topology × N (10 seeds each).

For each topology and each N, report max-over-seeds-and-DOFs of the
relative error vs FD on the t=0 input gradient (worst-case multi-step
accumulation site).

Uses cached scenes per (mjcf, requires_grad) to avoid Quadrants kernel
cache buildup that causes segfaults in long sweeps.
"""

import sys
import time
import numpy as np
import genesis as gs

sys.path.insert(0, "notes")
from diag_multistep_worst_case import TOPOLOGIES, build, loss_fn


_SCENE_CACHE = {}


def get_scenes(mjcf):
    if mjcf not in _SCENE_CACHE:
        sa, ra = build(mjcf, True)
        sb, rb = build(mjcf, False)
        _SCENE_CACHE[mjcf] = {"ana": (sa, ra), "fd": (sb, rb)}
    c = _SCENE_CACHE[mjcf]
    return c["ana"], c["fd"]


def measure(mjcf, n_dofs, N, seed):
    (sa, ra), (sb, rb) = get_scenes(mjcf)
    rng = np.random.default_rng(seed)
    u_list = [rng.normal(size=n_dofs) * 0.3 for _ in range(N)]
    u_anas = [gs.tensor(u, dtype=gs.tc_float, requires_grad=True) for u in u_list]
    sa.reset()
    for t in range(N):
        ra.control_dofs_force(u_anas[t])
        sa.step()
    loss_fn(sa).backward()
    ana = np.array([u.grad.detach().cpu().numpy() for u in u_anas])

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


def rel_err_t0(ana, fd, atol=1e-10):
    """Max rel error at t=0, considering only DOFs where |fd| > atol."""
    a0 = np.abs(ana[0])
    f0 = np.abs(fd[0])
    d0 = np.abs(ana[0] - fd[0])
    mask = f0 > atol
    if mask.any():
        return float((d0[mask] / f0[mask]).max())
    return 0.0


def main():
    gs.init(precision="64", backend=gs.cpu, performance_mode=False, logging_level="warning")
    Ns = [1, 2, 4, 8, 16, 32]
    seeds = list(range(1000, 1010))

    # results[topo_name][N] = (max_rel_over_seeds, mean_rel)
    results = {}
    for name, mjcf, n_dofs in TOPOLOGIES:
        print(f"\n=== {name} (n_dofs={n_dofs}) ===", flush=True)
        results[name] = {}
        # warm up scene cache once
        get_scenes(mjcf)
        for N in Ns:
            rels = []
            t0 = time.time()
            for seed in seeds:
                ana, fd = measure(mjcf, n_dofs, N, seed)
                rels.append(rel_err_t0(ana, fd))
            elapsed = time.time() - t0
            results[name][N] = (max(rels), float(np.mean(rels)))
            print(
                f"  N={N:>3}: max rel={max(rels):.3e}  mean rel={np.mean(rels):.3e}  "
                f"({elapsed:.1f}s)",
                flush=True,
            )

    # =================== Final table ===================
    print()
    print("=" * 110)
    print("FINAL TABLE — max rel error across 10 seeds at t=0")
    print("=" * 110)
    header = f"{'topology':<14} | " + " | ".join(f"{'N=' + str(N):>11}" for N in Ns)
    print(header)
    print("-" * 110)
    for name in [t[0] for t in TOPOLOGIES]:
        row = f"{name:<14} | " + " | ".join(f"{results[name][N][0]:>11.3e}" for N in Ns)
        print(row)

    print()
    print("=" * 110)
    print("FINAL TABLE — mean rel error across 10 seeds at t=0")
    print("=" * 110)
    print(header)
    print("-" * 110)
    for name in [t[0] for t in TOPOLOGIES]:
        row = f"{name:<14} | " + " | ".join(f"{results[name][N][1]:>11.3e}" for N in Ns)
        print(row)


if __name__ == "__main__":
    main()
