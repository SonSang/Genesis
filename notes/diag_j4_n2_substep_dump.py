"""J4 N=2 — substep stage dump.

Goal: at each backward stage (entry / post-fwd_velocity / post-COM_links
/ post-UCS / post-begin / post-step_2 / post-compute_qacc / post-fwd_dyn),
print which fields are non-zero. Compare J4 (catastrophic at N=2) vs
J1 (well-behaved). Identify the J4-specific stage where the chain
diverges from FP64 floor.

Runs with GENESIS_DEBUG_GRAD=1 so `_debug_grad_dump` emits at every
checkpoint inside substep_pre_coupling_grad.
"""

import os
os.environ["GENESIS_DEBUG_GRAD"] = "2"

import sys
import numpy as np

sys.path.insert(0, "notes")
from diag_all_topo_relerror_sweep import measure  # noqa
from diag_multistep_worst_case import TOPOLOGIES


def main():
    import genesis as gs

    gs.init(precision="64", backend=gs.cpu, performance_mode=False, logging_level="info")
    name_map = {t[0]: t for t in TOPOLOGIES}
    _, mjcf, n_dofs = name_map["J4_free_rev"]

    # Drive a single N=2 measure with seed 1000 — leaves a full
    # GENESIS_DEBUG_GRAD dump trail in stdout.
    print("J4 N=2 seed=1000 — full backward stage dump")
    print("=" * 110)
    ana, fd = measure(mjcf, n_dofs, 2, 1000)

    print()
    print("ana[t=0] vs fd[t=0]:")
    for i, name in enumerate(["root_x", "root_y", "root_z", "root_wx", "root_wy", "root_wz", "arm_revolute"]):
        a, f = ana[0, i], fd[0, i]
        d = a - f
        rel = abs(d) / max(abs(f), 1e-30)
        print(f"  {name:<14}  ana={a:>14.3e}  fd={f:>14.3e}  diff={d:>14.3e}  rel={rel:>10.3e}")


if __name__ == "__main__":
    main()
