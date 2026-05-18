"""Verify kernel_manual_compute_qacc_bw on J4 N=2 t=0:
   force.grad = M^{-1} . acc.grad     (M from forward primal at step 0)

Capture (via monkey-patched _debug_grad_dump):
  - mass_mat (forward primal at step 0) — read just before compute_qacc.grad
  - acc.grad  — read just before compute_qacc.grad ("after step_2.grad" tag)
  - force.grad — read just after compute_qacc.grad

Compute M^{-1} . acc.grad in numpy and compare to force.grad.
If they match → manual_compute_qacc_bw is correct, error is upstream.
If they differ → bug inside manual_compute_qacc_bw.
"""

import os
os.environ["GENESIS_DEBUG_GRAD"] = "1"

import sys
import numpy as np

sys.path.insert(0, "notes")
from diag_all_topo_relerror_sweep import measure  # noqa: F401
from diag_multistep_worst_case import TOPOLOGIES, build, loss_fn
from genesis.utils.misc import qd_to_torch


def main():
    import genesis as gs

    gs.init(precision="64", backend=gs.cpu, performance_mode=False, logging_level="info")
    name_map = {t[0]: t for t in TOPOLOGIES}
    _, mjcf, n_dofs = name_map["J4_free_rev"]

    sa, ra = build(mjcf, True)
    captures = {}

    solver = sa.rigid_solver
    orig_dump = solver._debug_grad_dump

    def patched(tag):
        orig_dump(tag)
        # In t=0 (second backward call), capture mass_mat (forward primal) and acc.grad
        # right before compute_qacc.grad; capture force.grad right after.
        # The dump tags are emitted twice (t=1 and t=0). We want the SECOND occurrence.
        if "after step_2.grad" in tag:
            captures.setdefault("step_2_after", []).append({
                "mass_mat": qd_to_torch(solver._rigid_global_info.mass_mat, copy=True).clone().numpy(),
                "acc_grad": qd_to_torch(solver.dofs_state.acc.grad, copy=True).clone().numpy(),
            })
        if "after compute_qacc.grad" in tag:
            captures.setdefault("compute_qacc_after", []).append({
                "force_grad": qd_to_torch(solver.dofs_state.force.grad, copy=True).clone().numpy(),
            })

    solver._debug_grad_dump = patched

    # Run N=2 backward
    seed = 1000
    rng = np.random.default_rng(seed)
    N = 2
    u_list = [rng.normal(size=n_dofs) * 0.3 for _ in range(N)]
    u_anas = [gs.tensor(u, dtype=gs.tc_float, requires_grad=True) for u in u_list]
    sa.reset()
    for t in range(N):
        ra.control_dofs_force(u_anas[t])
        sa.step()
    loss_fn(sa).backward()

    print("\n" + "=" * 80)
    print("Manual verification: force.grad =? M^{-1} . acc.grad at t=0")
    print("=" * 80)

    # captures lists: index 0 = t=1's substep, index 1 = t=0's substep
    sa_pre_t0 = captures["step_2_after"][1]   # t=0's after step_2.grad
    sa_post_t0 = captures["compute_qacc_after"][1]  # t=0's after compute_qacc.grad

    mm = sa_pre_t0["mass_mat"]
    acc_g = sa_pre_t0["acc_grad"]
    f_g = sa_post_t0["force_grad"]

    # Squeeze batch dim (B=1)
    if mm.ndim == 3:
        mm = mm[..., 0]
    if acc_g.ndim == 2:
        acc_g = acc_g[..., 0]
    if f_g.ndim == 2:
        f_g = f_g[..., 0]

    print(f"\nmass_mat shape: {mm.shape}")
    print(f"acc.grad shape: {acc_g.shape}")
    print(f"force.grad shape: {f_g.shape}")

    print("\nmass_mat (forward primal at step 0):")
    for row in mm:
        print("  " + " ".join(f"{v:>11.3e}" for v in row))

    print("\nacc.grad (after step_2.grad, before compute_qacc.grad):")
    print("  " + " ".join(f"{v:>11.3e}" for v in acc_g))

    print("\nforce.grad (after compute_qacc.grad, our manual kernel output):")
    print("  " + " ".join(f"{v:>11.3e}" for v in f_g))

    # Manual M^{-1} . acc.grad
    # Symmetrize lower-tri (since func_factor_mass only reads lower-tri).
    # But mass_mat is stored full — let's check symmetry:
    sym_err = float(np.abs(mm - mm.T).max())
    print(f"\nmass_mat symmetry error |M - M^T|_max = {sym_err:.3e}")

    # Use lower triangle to reconstruct symmetric M
    mm_sym = np.tril(mm) + np.tril(mm, -1).T

    # Solve M . force_grad_manual = acc.grad
    force_grad_manual = np.linalg.solve(mm_sym, acc_g)

    print("\nManual: force.grad_manual = M^{-1} . acc.grad")
    print("  " + " ".join(f"{v:>11.3e}" for v in force_grad_manual))

    diff = f_g - force_grad_manual
    print("\nforce.grad (kernel) - force.grad_manual:")
    print("  " + " ".join(f"{v:>11.3e}" for v in diff))
    print(f"max|diff| = {float(np.abs(diff).max()):.3e}")
    rel = np.abs(diff) / np.maximum(np.abs(force_grad_manual), 1e-30)
    print(f"max rel  = {float(rel.max()):.3e}")


if __name__ == "__main__":
    main()
