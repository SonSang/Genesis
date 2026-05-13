"""J1 — rel error sweep across multiple seeds × N.

Goal: validate that the analytical backward stays under 1% relative
error vs FD across a wide range of (seed, N) for J1 freejoint. If
confirmed, the 'leak' we previously chased is actually within
normal numerical tolerance for multi-step gradient simulation.

Outputs a table indexed by (N, seed) with rel error at t=0 (the
worst case, accumulating the most multi-step chain noise).
"""

import sys
import time
import numpy as np
import genesis as gs

sys.path.insert(0, "notes")
from diag_multistep_worst_case import TOPOLOGIES, build, loss_fn


# Scene cache to avoid build_scene accumulation across sweep iterations
# (each fresh build adds to Quadrants kernel cache; segfaults around ~80
# builds in long sweeps).
_SCENE_CACHE = {}


def get_scenes(mjcf):
    if "ana" not in _SCENE_CACHE:
        sa, ra = build(mjcf, True)
        sb, rb = build(mjcf, False)
        _SCENE_CACHE["ana"] = (sa, ra)
        _SCENE_CACHE["fd"] = (sb, rb)
    return _SCENE_CACHE["ana"], _SCENE_CACHE["fd"]


def measure(mjcf, n_dofs, N, seed):
    """Cached-scene version of diag_multistep_worst_case.measure()."""
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


def main():
    gs.init(precision="64", backend=gs.cpu, performance_mode=False, logging_level="warning")
    name_map = {t[0]: t for t in TOPOLOGIES}
    _, mjcf, n_dofs = name_map["J1_free"]

    Ns = [1, 2, 4, 8, 16, 32]
    seeds = list(range(1000, 1010))  # 10 seeds

    print(f"J1 rel error sweep — seeds={seeds[0]}..{seeds[-1]}, N in {Ns}")
    print("=" * 110)
    header = f"{'seed':>6} | " + " | ".join(f"{'N=' + str(N):>11}" for N in Ns)
    print(header)
    print("-" * 110)

    # results[seed][N] = (max_rel_err_t0, max_abs_diff_t0)
    rows = []
    for seed in seeds:
        cells = []
        for N in Ns:
            t0 = time.time()
            ana, fd = measure(mjcf, n_dofs, N, seed)
            elapsed = time.time() - t0
            # at t=0
            ana0 = np.abs(ana[0])
            fd0 = np.abs(fd[0])
            d0 = np.abs(ana[0] - fd[0])
            # rel error: only meaningful entries (|fd| > 1e-10) — skip
            # DOFs where both ana and fd are near zero (rotation DOFs in
            # the J1 freejoint test where ctrl_force on rotation produces
            # exactly-zero translation gradient).
            mask = fd0 > 1e-10
            if mask.any():
                rel = d0[mask] / fd0[mask]
                max_rel = float(rel.max())
            else:
                max_rel = 0.0
            max_abs = float(d0.max())
            cells.append((max_rel, max_abs, elapsed))
        rows.append((seed, cells))
        row_str = f"{seed:>6} | " + " | ".join(f"{rel:>5.1e}/{abs_:>4.0e}" for rel, abs_, _ in cells)
        print(row_str)

    print()
    print("(format: max|rel|/max|abs| at t=0)")
    print()

    # summary across seeds per N
    print(f"{'N':>3} | {'mean rel':>10} {'max rel':>10} {'pass 1%?':>10}")
    print("-" * 45)
    for ni, N in enumerate(Ns):
        rels = [rows[si][1][ni][0] for si in range(len(seeds))]
        mean_rel = float(np.mean(rels))
        max_rel = float(np.max(rels))
        passed = "YES" if max_rel < 0.01 else "NO"
        print(f"{N:>3} | {mean_rel:>10.3e} {max_rel:>10.3e} {passed:>10}")


if __name__ == "__main__":
    main()
