"""J4 N=1 per-DOF dump across 10 seeds — verify the 0.7% rel error
reported by diag_all_topo_relerror_sweep.

If we see large rel error in any DOF at N=1, that's a single-substep
backward bug we missed previously."""

import sys
import numpy as np
import genesis as gs

sys.path.insert(0, "notes")
from diag_all_topo_relerror_sweep import measure
from diag_multistep_worst_case import TOPOLOGIES


DOF_LABELS = ["root_x", "root_y", "root_z", "root_wx", "root_wy", "root_wz", "arm_revolute"]


def main():
    gs.init(precision="64", backend=gs.cpu, performance_mode=False, logging_level="warning")
    name_map = {t[0]: t for t in TOPOLOGIES}
    _, mjcf, n_dofs = name_map["J4_free_rev"]

    print("J4 N=1 per-DOF rel error across seeds (mask: |fd| > 1e-10)")
    print("=" * 110)

    biggest_rel = 0.0
    biggest_seed = -1
    biggest_dof = -1

    for seed in range(1000, 1010):
        ana, fd = measure(mjcf, n_dofs, 1, seed)
        print()
        print(f"seed={seed}")
        print(f"  {'DOF':<14}  {'ana':>14}  {'FD':>14}  {'diff':>14}  {'rel (if |fd|>1e-10)':>20}")
        for i in range(n_dofs):
            a, f = ana[0, i], fd[0, i]
            d = a - f
            if abs(f) > 1e-10:
                rel = abs(d) / abs(f)
                rel_s = f"{rel:.3e}"
                if rel > biggest_rel:
                    biggest_rel = rel
                    biggest_seed = seed
                    biggest_dof = i
            else:
                rel_s = "—"
            mark = ""
            if abs(f) > 1e-10 and abs(d) / abs(f) > 0.001:
                mark = "  <-- rel > 0.1%"
            print(f"  {DOF_LABELS[i]:<14}  {a:>14.3e}  {f:>14.3e}  {d:>14.3e}  {rel_s:>20}{mark}")

    print()
    print(f"biggest rel error: {biggest_rel:.3e} at seed={biggest_seed}, DOF={DOF_LABELS[biggest_dof]}")


if __name__ == "__main__":
    main()
