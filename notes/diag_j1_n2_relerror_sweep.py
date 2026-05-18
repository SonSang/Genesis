"""J1 N=2 multistep rel-error sweep across many seeds.

Goal: with current fix (Step 1 BW active + manual kernel, NO leak zeroing),
quantify the per-step / per-DOF analytical-vs-FD relative error at N=2.
Since substep t=1 (processed first in backward) has no prior backward to
leak from, its results should be FP64-floor-clean. Substep t=0 (second
in backward) carries any cross-substep leak. Compare the two.

Sweep 8 seeds, report:
  - per-step max|abs diff|
  - per-step max rel error (where |fd| > 1e-12)
  - per-step rel error of translation gradient
  - per-step rel error of rotation gradient
"""
import sys
import numpy as np
import genesis as gs

sys.path.insert(0, "notes")
from diag_multistep_worst_case import TOPOLOGIES, measure


DOF_LABELS = ["root_x", "root_y", "root_z", "root_wx", "root_wy", "root_wz"]


def main():
    gs.init(precision="64", backend=gs.cpu, performance_mode=False, logging_level="warning")
    name_map = {t[0]: t for t in TOPOLOGIES}
    _, mjcf, n_dofs = name_map["J1_free"]
    assert n_dofs == 6

    seeds = list(range(1000, 1010))
    rows = []
    rows.append("J1 N=2 multistep rel-error sweep (current fix: Step 1 BW active, no leak zero)")
    rows.append("=" * 110)
    rows.append(f"{'seed':>6} {'step':>5} | {'max|diff|':>11} {'max|rel|*':>10} | "
                f"{'tx_rel_avg':>11} {'rot_abs_max':>12}")
    rows.append("-" * 110)
    rows.append("(* rel only where |fd| > 1e-12; rot_abs_max because rot fd is exactly 0)")
    rows.append("")

    for seed in seeds:
        try:
            ana, fd = measure(mjcf, n_dofs, 2, seed)
        except Exception as e:
            rows.append(f"  seed={seed} ERROR: {e}")
            continue
        # shape: (N=2, n_dofs=6)
        for t in range(2):
            diff = ana[t] - fd[t]
            max_abs = float(np.abs(diff).max())
            mask = np.abs(fd[t]) > 1e-12
            if mask.any():
                rel = np.where(mask, np.abs(diff) / np.where(mask, np.abs(fd[t]), 1.0), 0.0)
                max_rel = float(rel.max())
            else:
                max_rel = float("nan")
            # Translation (DOF 0-2): rel error of each, average
            tx_rel = []
            for i in range(3):
                if abs(fd[t, i]) > 1e-12:
                    tx_rel.append(abs(diff[i]) / abs(fd[t, i]))
            tx_rel_avg = float(np.mean(tx_rel)) if tx_rel else float("nan")
            # Rotation (DOF 3-5): abs max (fd is 0 here always)
            rot_abs_max = float(np.abs(diff[3:6]).max())
            rows.append(
                f"{seed:>6} {t:>5} | {max_abs:>11.3e} {max_rel:>10.3e} | "
                f"{tx_rel_avg:>11.3e} {rot_abs_max:>12.3e}"
            )
        rows.append("")

    text = "\n".join(rows)
    print(text)
    out = "notes/diag_j1_n2_relerror_sweep.txt"
    with open(out, "w") as fh:
        fh.write(text + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
