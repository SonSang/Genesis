"""J1 multistep leak — per-N scaling.

For N in {2, 3, 4, 5, 7, 10} measure how the ana-vs-FD discrepancy
grows. Both:
  - max|diff| at each step t (focus on t=0 which is processed LAST in
    backward, so it accumulates the most cross-substep effects);
  - max|diff| over all steps (worst case).
"""

import sys
import numpy as np
import genesis as gs

sys.path.insert(0, "notes")
from diag_multistep_worst_case import TOPOLOGIES, measure


def main():
    gs.init(precision="64", backend=gs.cpu, performance_mode=False, logging_level="warning")
    name_map = {t[0]: t for t in TOPOLOGIES}
    _, mjcf, n_dofs = name_map["J1_free"]
    assert n_dofs == 6

    Ns = [2, 3, 4, 5, 7, 10]
    seeds = [1000, 1001, 1002]

    print("J1 leak scaling vs N (3 seeds: 1000, 1001, 1002)")
    print("=" * 110)
    print(
        f"{'N':>3} | {'t=0 max|d|':>11} {'t=0 mean|d|':>11} | {'t=N-1 max|d|':>13} | {'all max|d|':>11} {'worst t':>7}"
    )
    print("-" * 110)

    for N in Ns:
        t0_diffs = []
        last_diffs = []
        all_max = -1.0
        worst_t = -1
        for seed in seeds:
            ana, fd = measure(mjcf, n_dofs, N, seed)
            # ana / fd shape: (N, n_dofs)
            d = np.abs(ana - fd)
            t0_diffs.append(d[0].max())
            last_diffs.append(d[-1].max())
            for t in range(N):
                if d[t].max() > all_max:
                    all_max = d[t].max()
                    worst_t = t

        t0_max = max(t0_diffs)
        t0_mean = np.mean(t0_diffs)
        last_max = max(last_diffs)

        print(f"{N:>3} | {t0_max:>11.3e} {t0_mean:>11.3e} | {last_max:>13.3e} | {all_max:>11.3e} {worst_t:>7}")

    print()
    print("(t=N-1 is processed FIRST in backward — should always be FP64 floor.")
    print(" t=0 is processed LAST — accumulates cross-substep effects.)")


if __name__ == "__main__":
    main()
