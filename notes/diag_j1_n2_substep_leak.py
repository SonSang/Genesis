"""J1 N=2 multistep with grad dumps at every backward stage.

Goal: identify which field's `.grad` is NON-ZERO at the start of substep
t=0's backward (after substep t=1's backward completed). That non-zero
field is the cross-substep leak source.

Backward order: substep t=1 (processed FIRST) → substep t=0 (processed
SECOND). When entering substep t=0's backward, all gradient fields should
ideally be zero except for those carrying the chain from t=1's qpos/vel
inputs. Any other non-zero .grad indicates a leak.

Run with GENESIS_DEBUG_GRAD=1 (set inside this script).
"""
import os
os.environ["GENESIS_DEBUG_GRAD"] = "1"

import sys
import numpy as np
import torch

sys.path.insert(0, "notes")
from diag_multistep_worst_case import TOPOLOGIES, measure


def main():
    import genesis as gs
    gs.init(precision="64", backend=gs.cpu, performance_mode=False, logging_level="info")

    name_map = {t[0]: t for t in TOPOLOGIES}
    _, mjcf, n_dofs = name_map["J1_free"]
    assert n_dofs == 6

    # Just run measure once with N=2 and let the dumps print.
    print("=" * 80)
    print("J1 N=2 multistep — grad dump at every backward substep stage")
    print("=" * 80)
    ana, fd = measure(mjcf, n_dofs, 2, 1000)
    print("=" * 80)
    print(f"ana shape={ana.shape}, fd shape={fd.shape}")
    for t in range(2):
        print(f"step t={t}:")
        print(f"  ana = {ana[t]}")
        print(f"  fd  = {fd[t]}")
        print(f"  diff= {ana[t] - fd[t]}")


if __name__ == "__main__":
    main()
