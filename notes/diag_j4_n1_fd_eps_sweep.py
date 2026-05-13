"""FD eps sweep for J4 N=1 seed=1001 ctrl_force.grad[d=4].

Test whether FD = +1.090e-8 (vs analytical -1.022e-8) is robust to eps
or just truncation/round-off artifact.

For each eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8], measure central FD
of ctrl_force.grad[d=4] and compare to analytical -1.022e-8.
"""

import sys
import numpy as np

import genesis as gs

sys.path.insert(0, "notes")


def main():
    gs.init(precision="64", backend=gs.cpu, performance_mode=False, logging_level="warning")
    from diag_multistep_worst_case import TOPOLOGIES, build, loss_fn

    name_map = {t[0]: t for t in TOPOLOGIES}
    _, mjcf, n_dofs = name_map["J4_free_rev"]

    seed = 1001
    rng = np.random.default_rng(seed)
    u_base = rng.normal(size=n_dofs) * 0.3
    target_d = 4

    sb, rb = build(mjcf, False)

    def loss_at(u):
        sb.reset()
        rb.control_dofs_force(gs.tensor(u, dtype=gs.tc_float))
        sb.step()
        return float(loss_fn(sb).detach().cpu())

    print(f"J4 N=1 seed={seed}, FD eps sweep for ctrl_force.grad[d={target_d}]")
    print("analytical (from dump/backward) = -1.022e-08")
    print()
    print(f"{'eps':>12}  {'loss(u+eps)':>20}  {'loss(u-eps)':>20}  {'diff':>14}  {'FD':>14}")
    print("-" * 100)
    for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]:
        up = u_base.copy()
        up[target_d] += eps
        um = u_base.copy()
        um[target_d] -= eps
        lp = loss_at(up)
        lm = loss_at(um)
        diff = lp - lm
        fd = diff / (2 * eps)
        print(f"{eps:>12.1e}  {lp:>20.13e}  {lm:>20.13e}  {diff:>14.3e}  {fd:>14.3e}")


if __name__ == "__main__":
    main()
