"""J1 — does ana[t=0] magnitude grow with N (butterfly effect)?

If |ana[t=0]| grows like ~2.7^N per step too, then the diff growth is
just proportional to the growing gradient — our analytical backward is
accurate to ~constant relative tolerance, not "leaking" anything.

Reports per N:
  - |ana[t=0]|_max          (magnitude of the gradient at the FIRST input)
  - |ana[t=N-1]|_max        (magnitude at the LAST input)
  - max|ana - fd|           (absolute error)
  - max|ana - fd| / |ana|   (relative error)
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

    Ns = [1, 2, 3, 4, 5, 7, 10]
    seed = 1001

    print(f"J1 seed={seed} — does ana[t=0] magnitude grow with N (butterfly effect)?")
    print("=" * 110)
    print(f"{'N':>3} | {'|ana[t=0]|_max':>15} {'|ana[t=N-1]|_max':>16} | {'max|d|':>11} {'rel err':>10}")
    print("-" * 110)

    prev_a0_max = None
    for N in Ns:
        a, fd = measure(mjcf, n_dofs, N, seed)
        a0_max = float(np.abs(a[0]).max())
        aL_max = float(np.abs(a[-1]).max())
        d = np.abs(a - fd)
        max_d = float(d[0].max())  # at t=0
        rel = max_d / max(a0_max, 1e-30)
        growth = "" if prev_a0_max is None else f"  (×{a0_max / prev_a0_max:.2f})"
        print(f"{N:>3} | {a0_max:>15.3e}{growth:<10} {aL_max:>16.3e} | {max_d:>11.3e} {rel:>10.3e}")
        prev_a0_max = a0_max


if __name__ == "__main__":
    main()
