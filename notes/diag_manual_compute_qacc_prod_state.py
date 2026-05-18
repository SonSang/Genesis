"""Production-state verify of `kernel_manual_compute_qacc_bw`.

Forward: acc_smooth = M^{-1} @ force (via LDLT M = L^T D L).
Reverse: force.grad = M^{-1} @ acc_smooth.grad (M symmetric, so M^-T = M^-1).

Capture production state at compute_qacc.grad entry (i.e. after step_2.grad)
and at exit. Compare against numpy LDLT solve with same primal.
"""
import os
os.environ["GENESIS_DEBUG_GRAD"] = "0"
import sys, numpy as np
import genesis as gs
gs.init(precision="64", backend=gs.cpu, performance_mode=False, logging_level="warning")

from genesis.utils.misc import qd_to_torch
sys.path.insert(0, "notes")
from diag_multistep_worst_case import TOPOLOGIES, build, loss_fn


def numpy_ldlt_solve(mass_mat, b):
    """Solve M @ x = b where M = L^T D L (Genesis transposed convention).
    Reads lower-tri of mass_mat. Strict left-to-right FP order.
    """
    n = b.shape[0]
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            M[i, j] = mass_mat[i, j]
            if i != j:
                M[j, i] = mass_mat[i, j]
    # LDLT factor (Genesis convention: M = L^T D L, where L is upper-tri unit)
    # i.e. L_ji for i < j stored at mass_mat_L[j, i] in solver.
    # But we'll just use numpy solve here as reference.
    x = np.linalg.solve(M, b)
    return x


def main():
    name_map = {t[0]: t for t in TOPOLOGIES}
    _, mjcf, n_dofs = name_map["J4_free_rev"]
    sa, ra = build(mjcf, True)
    solver = sa.rigid_solver
    captures = {}
    orig_dump = solver._debug_grad_dump

    def patched(tag):
        orig_dump(tag)
        if "after step_2.grad" in tag:
            # Input to compute_qacc.grad
            captures.setdefault("inputs", []).append({
                "acc_grad": qd_to_torch(solver.dofs_state.acc.grad, copy=True).numpy()[..., 0].copy(),
                "acc_smooth_grad": qd_to_torch(solver.dofs_state.acc_smooth.grad, copy=True).numpy()[..., 0].copy(),
                "mass_mat": qd_to_torch(solver._rigid_global_info.mass_mat, copy=True).numpy()[..., 0].copy(),
                "acc_smooth": qd_to_torch(solver.dofs_state.acc_smooth, copy=True).numpy()[..., 0].copy(),
            })
        if "after compute_qacc.grad" in tag:
            # Output of compute_qacc.grad
            captures.setdefault("outputs", []).append({
                "force_grad": qd_to_torch(solver.dofs_state.force.grad, copy=True).numpy()[..., 0].copy(),
                "acc_grad": qd_to_torch(solver.dofs_state.acc.grad, copy=True).numpy()[..., 0].copy(),
            })

    solver._debug_grad_dump = patched

    rng = np.random.default_rng(1000)
    u_list = [rng.normal(size=n_dofs) * 0.3 for _ in range(2)]
    u_anas = [gs.tensor(u, dtype=gs.tc_float, requires_grad=True) for u in u_list]
    sa.reset()
    for t in range(2):
        ra.control_dofs_force(u_anas[t]); sa.step()
    loss_fn(sa).backward()

    # t=0 = LIFO last (index 1)
    inp = captures["inputs"][1]
    out = captures["outputs"][1]

    print("=== compute_qacc.grad input (after step_2.grad) ===")
    print(f"  acc.grad        = {inp['acc_grad']}")
    print(f"  acc_smooth.grad = {inp['acc_smooth_grad']}")
    print(f"  mass_mat shape  = {inp['mass_mat'].shape}")
    print(f"  acc_smooth      = {inp['acc_smooth']}")

    # Combined seed = acc.grad + acc_smooth.grad (per kernel_manual_compute_qacc_bw)
    combined_seed = inp["acc_grad"] + inp["acc_smooth_grad"]
    print(f"\n  combined seed = {combined_seed}")

    # Manual numpy: force.grad = M^{-1} @ combined_seed
    force_grad_manual = numpy_ldlt_solve(inp["mass_mat"], combined_seed)

    print(f"\n=== compute_qacc.grad output ===")
    print(f"  kernel force.grad = {out['force_grad']}")
    print(f"  manual force.grad = {force_grad_manual}")
    diff = out["force_grad"] - force_grad_manual
    print(f"  diff              = {diff}")
    print(f"  max|d|            = {float(np.abs(diff).max()):.3e}")
    print(f"  manual / kernel rel = {np.abs(diff) / np.maximum(np.abs(force_grad_manual), 1e-30)}")


if __name__ == "__main__":
    main()
