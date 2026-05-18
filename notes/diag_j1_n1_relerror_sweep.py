"""J1 N=1 rel-error sweep — baseline for comparison with N=2.

If N=1 (no cross-substep) shows clean FP64-floor rel error but N=2 t=0
shows 0.01-0.1% rel error, then the latter is purely cross-substep leak.
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

    seeds = list(range(1000, 1010))
    rows = ["J1 N=1 rel-error sweep (no cross-substep, baseline)",
            "=" * 90,
            f"{'seed':>6} | {'max|diff|':>11} {'max|rel|*':>10} | "
            f"{'tx_rel_avg':>11} {'rot_abs_max':>12}",
            "-" * 90,
            "(* rel only where |fd| > 1e-12; rot_abs_max because rot fd is exactly 0)",
            ""]
    for seed in seeds:
        try:
            ana, fd = measure(mjcf, n_dofs, 1, seed)
        except Exception as e:
            rows.append(f"  seed={seed} ERROR: {e}")
            continue
        # shape (1, 6)
        diff = ana[0] - fd[0]
        max_abs = float(np.abs(diff).max())
        mask = np.abs(fd[0]) > 1e-12
        rel = np.where(mask, np.abs(diff) / np.where(mask, np.abs(fd[0]), 1.0), 0.0)
        max_rel = float(rel.max())
        tx_rel = [abs(diff[i]) / abs(fd[0, i]) for i in range(3) if abs(fd[0, i]) > 1e-12]
        tx_rel_avg = float(np.mean(tx_rel)) if tx_rel else float("nan")
        rot_abs_max = float(np.abs(diff[3:6]).max())
        rows.append(
            f"{seed:>6} | {max_abs:>11.3e} {max_rel:>10.3e} | "
            f"{tx_rel_avg:>11.3e} {rot_abs_max:>12.3e}"
        )

    text = "\n".join(rows)
    print(text)
    with open("notes/diag_j1_n1_relerror_sweep.txt", "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
