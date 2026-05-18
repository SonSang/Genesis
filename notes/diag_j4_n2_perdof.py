"""J4 N=2 per-DOF deep dive — locate which DOF triggers the 1200% rel error.

For each seed in {1000..1004}, dump ana[t=0] vs fd[t=0] for all 7 DOFs
of J4 (free + revolute):
  root_x, root_y, root_z, root_wx, root_wy, root_wz, arm_revolute.

Flag DOFs whose |fd| > 1e-10 (meaningful gradient) but |ana - fd| / |fd|
exceeds 1% — those are the chain-rule breakage candidates.
"""

import sys
import numpy as np
import genesis as gs

sys.path.insert(0, "notes")
from diag_all_topo_relerror_sweep import measure
from diag_multistep_worst_case import TOPOLOGIES


DOF_LABELS = [
    "root_x",
    "root_y",
    "root_z",
    "root_wx",
    "root_wy",
    "root_wz",
    "arm_revolute",
]


def main():
    gs.init(precision="64", backend=gs.cpu, performance_mode=False, logging_level="warning")
    name_map = {t[0]: t for t in TOPOLOGIES}
    _, mjcf, n_dofs = name_map["J4_free_rev"]
    assert n_dofs == 7

    print("J4 N=2 per-DOF per-seed ana vs FD breakdown")
    print("=" * 110)

    for seed in [1000, 1001, 1002, 1003, 1004]:
        ana, fd = measure(mjcf, n_dofs, 2, seed)
        for t_label, t in [("t=1 (FIRST in backward — no cross-substep effect)", 1), ("t=0 (SECOND in backward — cross-substep accumulates)", 0)]:
            print()
            print(f"seed={seed}  step {t_label}")
            print(f"  {'DOF':<14}  {'analytical':>14}  {'FD':>14}  {'diff (a-f)':>14}  {'rel err':>10}")
            for i in range(n_dofs):
                a, f = ana[t, i], fd[t, i]
                d = a - f
                if abs(f) > 1e-12:
                    rel = abs(d) / abs(f)
                else:
                    rel = 0.0 if abs(d) < 1e-12 else float("inf")
                mark = ""
                if abs(f) > 1e-10 and rel > 0.01:
                    mark = "  <-- rel > 1%"
                elif abs(f) > 1e-10 and rel > 1e-4:
                    mark = "  <-- rel > 1e-4"
                print(f"  {DOF_LABELS[i]:<14}  {a:>14.3e}  {f:>14.3e}  {d:>14.3e}  {rel:>10.3e}{mark}")


if __name__ == "__main__":
    main()
