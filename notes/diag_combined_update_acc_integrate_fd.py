"""Isolated FD verify of `func_update_acc + func_integrate` combined chain.

Hypothesis: step_2.grad silent drop originates from kernel-level composition
of `func_update_acc` and `func_integrate` (default integrator skips
`func_implicit_damping`). Test by running both together in a single kernel,
seeding only vel_next.grad / qpos_next.grad (= production seed at backward
entry to step_2), and comparing Quadrants kernel.grad vs manual.

If diff matches the production step_2.grad silent drop (~1e-11), composition
is confirmed as the source. If diff stays at FP64 floor (~1e-15), the silent
drop must come from elsewhere (e.g. interaction with other state fields
populated by the wider self.substep replay).
"""
import os
os.environ["GENESIS_DEBUG_GRAD"] = "0"

import sys
import numpy as np
import genesis as gs

gs.init(precision="64", backend=gs.cpu, performance_mode=False, logging_level="warning")

import quadrants as qd
from genesis.utils.misc import qd_to_torch
from genesis.engine.solvers.rigid.abd.forward_dynamics import (
    kernel_func_integrate_standalone,
    kernel_update_acc_plus_integrate_standalone,
)

def build_j4_solver():
    sys.path.insert(0, "notes")
    from diag_multistep_worst_case import TOPOLOGIES, build
    name_map = {t[0]: t for t in TOPOLOGIES}
    _, mjcf, _ = name_map["J4_free_rev"]
    s, r = build(mjcf, True)
    return s, r


def run_kernel(kernel_call, solver, qpos, vel, acc, qpos_next_grad, vel_next_grad,
               extra_state_setter=None):
    """Run the given kernel forward, seed grads, then call .grad. Return outputs."""
    n_q = qpos.shape[0]
    n_dof = vel.shape[0]
    _B = 1
    solver._rigid_global_info.qpos.from_numpy(qpos.reshape(n_q, _B).astype(np.float64))
    solver.dofs_state.vel.from_numpy(vel.reshape(n_dof, _B).astype(np.float64))
    solver.dofs_state.acc.from_numpy(acc.reshape(n_dof, _B).astype(np.float64))

    # Zero all relevant grads
    for fld in [solver._rigid_global_info.qpos.grad,
                solver._rigid_global_info.qpos_next.grad,
                solver.dofs_state.vel.grad, solver.dofs_state.vel_next.grad,
                solver.dofs_state.acc.grad,
                solver.links_state.cdd_vel.grad, solver.links_state.cdd_ang.grad,
                solver.links_state.cacc_lin.grad, solver.links_state.cacc_ang.grad]:
        fld.from_numpy(np.zeros_like(fld.to_numpy()))

    # Zero output state too (forward outputs)
    solver._rigid_global_info.qpos_next.from_numpy(np.zeros_like(solver._rigid_global_info.qpos_next.to_numpy()))
    solver.dofs_state.vel_next.from_numpy(np.zeros_like(solver.dofs_state.vel_next.to_numpy()))
    solver.links_state.cdd_vel.from_numpy(np.zeros_like(solver.links_state.cdd_vel.to_numpy()))
    solver.links_state.cdd_ang.from_numpy(np.zeros_like(solver.links_state.cdd_ang.to_numpy()))
    solver.links_state.cacc_lin.from_numpy(np.zeros_like(solver.links_state.cacc_lin.to_numpy()))
    solver.links_state.cacc_ang.from_numpy(np.zeros_like(solver.links_state.cacc_ang.to_numpy()))

    if extra_state_setter is not None:
        extra_state_setter(solver)

    # Forward
    kernel_call(is_backward=True)

    # Seed
    solver._rigid_global_info.qpos_next.grad.from_numpy(qpos_next_grad.reshape(n_q, _B).astype(np.float64))
    solver.dofs_state.vel_next.grad.from_numpy(vel_next_grad.reshape(n_dof, _B).astype(np.float64))

    # Backward
    kernel_call.grad(is_backward=True)

    return (
        qd_to_torch(solver._rigid_global_info.qpos.grad, copy=True).numpy()[..., 0],
        qd_to_torch(solver.dofs_state.vel.grad, copy=True).numpy()[..., 0],
        qd_to_torch(solver.dofs_state.acc.grad, copy=True).numpy()[..., 0],
    )


def main():
    s, _ = build_j4_solver()
    solver = s.rigid_solver
    s.reset()
    dt = float(qd_to_torch(solver._rigid_global_info.substep_dt, copy=True).numpy())
    eps = float(qd_to_torch(solver._rigid_global_info.EPS, copy=True).numpy())

    rng = np.random.default_rng(1000)
    quat0 = np.array([1.0, 0.001, -0.0005, 0.0003]); quat0 /= np.linalg.norm(quat0)
    qpos = np.array([0.1, -0.05, 0.2, *quat0, 0.05])
    vel = rng.normal(size=7) * 0.1
    acc = rng.normal(size=7) * 1.0

    qpos_next_grad = rng.normal(size=8) * 1e-3
    vel_next_grad = rng.normal(size=7) * 1e-4

    print("=== Test 1: func_integrate alone (standalone) ===")
    def fi_call(is_backward=True):
        kernel_func_integrate_standalone(
            dofs_state=solver.dofs_state,
            links_info=solver.links_info,
            joints_info=solver.joints_info,
            rigid_global_info=solver._rigid_global_info,
            static_rigid_sim_config=solver._static_rigid_sim_config,
            is_backward=is_backward,
        )
    fi_call.grad = lambda is_backward=True: kernel_func_integrate_standalone.grad(
        dofs_state=solver.dofs_state,
        links_info=solver.links_info,
        joints_info=solver.joints_info,
        rigid_global_info=solver._rigid_global_info,
        static_rigid_sim_config=solver._static_rigid_sim_config,
        is_backward=is_backward,
    )

    qg_1, vg_1, ag_1 = run_kernel(fi_call, solver, qpos, vel, acc, qpos_next_grad, vel_next_grad)

    print("=== Test 2: func_update_acc + func_integrate (combined) ===")
    def comb_call(is_backward=True):
        kernel_update_acc_plus_integrate_standalone(
            dofs_state=solver.dofs_state,
            links_info=solver.links_info,
            joints_info=solver.joints_info,
            links_state=solver.links_state,
            entities_info=solver.entities_info,
            rigid_global_info=solver._rigid_global_info,
            static_rigid_sim_config=solver._static_rigid_sim_config,
            is_backward=is_backward,
        )
    comb_call.grad = lambda is_backward=True: kernel_update_acc_plus_integrate_standalone.grad(
        dofs_state=solver.dofs_state,
        links_info=solver.links_info,
        joints_info=solver.joints_info,
        links_state=solver.links_state,
        entities_info=solver.entities_info,
        rigid_global_info=solver._rigid_global_info,
        static_rigid_sim_config=solver._static_rigid_sim_config,
        is_backward=is_backward,
    )

    # For func_update_acc to do anything non-trivial, we need cdofd_*/cdof_* fields populated.
    # Reset solver state for combined run.
    n_dof_arr = solver.dofs_state.cdofd_vel.to_numpy().shape
    rng2 = np.random.default_rng(2000)
    def setup_cdof_cdofd(solver):
        # Random cdof / cdofd values, magnitude ~1.0 (matches realistic state)
        sh = solver.dofs_state.cdofd_vel.to_numpy().shape
        solver.dofs_state.cdofd_vel.from_numpy(rng2.normal(size=sh) * 0.5)
        solver.dofs_state.cdofd_ang.from_numpy(rng2.normal(size=sh) * 0.5)
        solver.dofs_state.cdof_vel.from_numpy(rng2.normal(size=sh) * 0.5)
        solver.dofs_state.cdof_ang.from_numpy(rng2.normal(size=sh) * 0.5)
        # gravity is constant (no-op)

    qg_2, vg_2, ag_2 = run_kernel(comb_call, solver, qpos, vel, acc, qpos_next_grad, vel_next_grad,
                                  extra_state_setter=setup_cdof_cdofd)

    print("\n=== Compare: func_integrate ALONE vs (update_acc + integrate) COMBINED ===")
    print("(Both seeded with same qpos_next.grad, vel_next.grad; cdd_*.grad / cacc_*.grad = 0)")
    for name, alone, comb in [("qpos.grad", qg_1, qg_2),
                              ("vel.grad", vg_1, vg_2),
                              ("acc.grad", ag_1, ag_2)]:
        diff = alone - comb
        mx = float(np.abs(diff).max())
        print(f"{name}: alone  = {alone}")
        print(f"{name}: comb   = {comb}")
        print(f"{name}: diff   = {diff}")
        print(f"{name}: max|d| = {mx:.3e}")


if __name__ == "__main__":
    main()
