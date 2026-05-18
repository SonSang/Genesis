"""J1 N=4 per-DOF / per-step ana vs FD breakdown.

The N=4 sweep showed max|diff| = 8.83e-11 with max_rel = 117 — small
absolute but large relative on some DOF that's near-zero. Dump per-DOF
per-step to identify which DOF and which step is responsible.
"""

import sys
import numpy as np
import genesis as gs

sys.path.insert(0, "notes")
from diag_multistep_worst_case import TOPOLOGIES, measure


DOF_LABELS = [
    "d=0 root_x",
    "d=1 root_y",
    "d=2 root_z",
    "d=3 root_wx",
    "d=4 root_wy",
    "d=5 root_wz",
]


def main():
    gs.init(precision="64", backend=gs.cpu, performance_mode=False, logging_level="warning")
    name_map = {t[0]: t for t in TOPOLOGIES}
    _, mjcf, n_dofs = name_map["J1_free"]
    assert n_dofs == 6

    lines = ["J1 N=4 per-DOF per-step ana vs FD breakdown", "=" * 110]
    for seed in [1000, 1001, 1002]:
        ana, fd = measure(mjcf, n_dofs, 4, seed)
        # ana / fd shape: (N=4, n_dofs=6)
        lines.append("")
        lines.append(f"seed={seed}")
        for t in range(4):
            lines.append(f"  step t={t}")
            lines.append(f"    {'DOF':<14}  {'analytical':>14}  {'FD':>14}  {'diff':>14}  {'rel':>10}")
            for i in range(n_dofs):
                a, f = ana[t, i], fd[t, i]
                d = a - f
                if abs(f) > 1e-12:
                    rel = d / abs(f)
                else:
                    rel = d / 1.0
                mark = ""
                if abs(d) > 1e-12 and abs(f) < 1e-9 and abs(a) > 1e-9:
                    mark = "  <-- ana >> fd ~0"
                lines.append(f"    {DOF_LABELS[i]:<14}  {a:>14.3e}  {f:>14.3e}  {d:>14.3e}  {rel:>10.3f}{mark}")

    out = "notes/diag_j1_n4_perdof.txt"
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
