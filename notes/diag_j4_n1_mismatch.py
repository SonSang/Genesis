"""J4 N=1 ana vs FD mismatch — per-DOF breakdown for every seed.

Saves a human-readable table to `notes/diag_j4_n1_mismatch.txt`
showing the analytical gradient, FD reference, and their diff for each
of the 7 DOFs of J4 (free + revolute) at N=1.

Worst seed at N=1 was 1001 (max|diff| 2.11e-8). We dump all three seeds
(1000, 1001, 1002) so you can see seed-by-seed how the pattern reproduces.
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
    "d=3 root_wx (axis-angle)",
    "d=4 root_wy",
    "d=5 root_wz",
    "d=6 arm_revolute",
]


def main():
    gs.init(precision="64", backend=gs.cpu, performance_mode=False, logging_level="warning")
    name_map = {t[0]: t for t in TOPOLOGIES}
    _, mjcf, n_dofs = name_map["J4_free_rev"]
    assert n_dofs == 7

    lines = []
    lines.append("J4 N=1 ana vs FD mismatch (per-DOF, per-seed)")
    lines.append("=" * 100)

    for seed in [1000, 1001, 1002]:
        ana, fd = measure(mjcf, n_dofs, 1, seed)
        # ana / fd are shape (1, 7) — single step
        a = ana[0]
        f = fd[0]
        d = a - f
        rel = d / np.where(np.abs(f) > 1e-12, np.abs(f), 1.0)

        lines.append("")
        lines.append(f"seed={seed}")
        lines.append(f"  {'DOF':<28}  {'analytical':>14}  {'FD':>14}  {'diff (a-f)':>14}  {'rel':>10}")
        for i in range(n_dofs):
            mark = ""
            if abs(d[i]) > 1e-9:
                if abs(a[i]) > abs(f[i]) * 1.5:
                    mark = "  <-- ana > 1.5× fd (over-count)"
                elif abs(a[i]) < abs(f[i]) * 0.7:
                    mark = "  <-- ana < 0.7× fd (silent drop)"
            lines.append(f"  {DOF_LABELS[i]:<28}  {a[i]:>14.3e}  {f[i]:>14.3e}  {d[i]:>14.3e}  {rel[i]:>10.2f}{mark}")
        lines.append(f"  max|diff| = {float(np.abs(d).max()):.3e}")

    out_path = "notes/diag_j4_n1_mismatch.txt"
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
