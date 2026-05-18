"""Verify hypothesis X1: silent drop is production state-specific.

Approach:
  1. Run N=1 simulation (which writes the same state as the t=0 backward
     replay forward) and capture qpos, vel, acc, cdof_*, cdofd_*, gravity.
  2. Inject these into the standalone kernel_update_acc_plus_integrate_standalone.
  3. Run its forward + .grad with the same seed used by production step_2.grad.
  4. Compare standalone .grad output vs manual computation.

If standalone diff ~1.14e-11 → silent drop is production-state-dependent (X1).
If standalone diff = 0 → silent drop must come from cross-kernel stash (X2).
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
    kernel_update_acc_plus_integrate_standalone,
)

sys.path.insert(0, "notes")
from diag_multistep_worst_case import TOPOLOGIES, build


# -----------------------------------------------------------------------------
# Numpy manual reverse (same as diag_func_integrate_isolated_fd.py)
# -----------------------------------------------------------------------------
def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])


def rotvec_to_quat(rotvec, eps):
    rx, ry, rz = rotvec
    thetasq = rx*rx + ry*ry + rz*rz
    theta_reg = np.sqrt(thetasq + eps*eps)
    c = np.cos(theta_reg / 2)
    sinc = np.sin(theta_reg / 2) / theta_reg
    return np.array([c, sinc*rx, sinc*ry, sinc*rz])


def manual_reverse(qpos, vel, acc, dt, eps, qpos_next_grad, vel_next_grad):
    vel_next = vel + dt * acc
    qpos_grad = np.zeros_like(qpos)
    vel_grad = np.zeros_like(vel)
    acc_grad = np.zeros_like(acc)

    qpos_grad[0:3] += qpos_next_grad[0:3]
    vel_next_grad_local = vel_next_grad.copy()
    vel_next_grad_local[0:3] += dt * qpos_next_grad[0:3]

    ang = vel_next[3:6] * dt
    qrot = rotvec_to_quat(ang, eps)
    rot0 = qpos[3:7]
    rot_grad = qpos_next_grad[3:7]
    bw, bx, by, bz = qrot
    ogw, ogx, ogy, ogz = rot_grad
    rot0_grad = np.array([
        ogw*bw + ogx*bx + ogy*by + ogz*bz,
        -ogw*bx + ogx*bw - ogy*bz + ogz*by,
        -ogw*by + ogx*bz + ogy*bw - ogz*bx,
        -ogw*bz - ogx*by + ogy*bx + ogz*bw,
    ])
    qpos_grad[3:7] += rot0_grad

    aw, ax, ay, az = rot0
    qrot_grad = np.array([
        ogw*aw + ogx*ax + ogy*ay + ogz*az,
        -ogw*ax + ogx*aw + ogy*az - ogz*ay,
        -ogw*ay - ogx*az + ogy*aw + ogz*ax,
        -ogw*az + ogx*ay - ogy*ax + ogz*aw,
    ])

    rx, ry, rz = ang
    thetasq = rx*rx + ry*ry + rz*rz
    theta_reg = np.sqrt(thetasq + eps*eps)
    theta_half = 0.5 * theta_reg
    sin_h = np.sin(theta_half)
    cos_h = np.cos(theta_half)
    sinc = sin_h / theta_reg
    dsinc_dtheta = (0.5 * cos_h - sinc) / theta_reg
    qg_w, qg_x, qg_y, qg_z = qrot_grad
    qg_dot_r = qg_x*rx + qg_y*ry + qg_z*rz
    coeff = -0.5 * sin_h / theta_reg * qg_w + dsinc_dtheta / theta_reg * qg_dot_r
    ang_grad = np.array([
        coeff*rx + sinc*qg_x,
        coeff*ry + sinc*qg_y,
        coeff*rz + sinc*qg_z,
    ])
    vel_next_grad_local[3:6] += dt * ang_grad

    qpos_grad[7] += qpos_next_grad[7]
    vel_next_grad_local[6] += dt * qpos_next_grad[7]

    vel_grad += vel_next_grad_local
    acc_grad += dt * vel_next_grad_local
    return qpos_grad, vel_grad, acc_grad


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    name_map = {t[0]: t for t in TOPOLOGIES}
    _, mjcf, n_dofs = name_map["J4_free_rev"]

    # === Step A: N=1 simulation to populate production state ===
    sa, ra = build(mjcf, True)
    solver = sa.rigid_solver

    rng = np.random.default_rng(1000)
    u0 = rng.normal(size=n_dofs) * 0.3

    sa.reset()
    # Capture pre-integrate state (= state[t=0] at backward replay time)
    qpos_pre = qd_to_torch(solver._rigid_global_info.qpos, copy=True).numpy()[..., 0].copy()
    vel_pre = qd_to_torch(solver.dofs_state.vel, copy=True).numpy()[..., 0].copy()

    ra.control_dofs_force(gs.tensor(u0, dtype=gs.tc_float))
    sa.step()

    # After step: state has the FULL forward state (post-step_2 = post-integrate)
    # We need the state at the moment of step_2's *forward* — which is exactly
    # the state populated by kernel_step_1 (which writes cdof_*, cdofd_*, acc).
    # Capture cdof_*, cdofd_*, acc — these are computed by kernel_step_1's
    # forward_dynamics chain and are present *during* step_2 forward.
    cdofd_vel = qd_to_torch(solver.dofs_state.cdofd_vel, copy=True).numpy()
    cdofd_ang = qd_to_torch(solver.dofs_state.cdofd_ang, copy=True).numpy()
    cdof_vel = qd_to_torch(solver.dofs_state.cdof_vel, copy=True).numpy()
    cdof_ang = qd_to_torch(solver.dofs_state.cdof_ang, copy=True).numpy()
    acc_post = qd_to_torch(solver.dofs_state.acc, copy=True).numpy()[..., 0].copy()
    gravity = qd_to_torch(solver._rigid_global_info.gravity, copy=True).numpy()
    dt = float(qd_to_torch(solver._rigid_global_info.substep_dt, copy=True).numpy())
    eps = float(qd_to_torch(solver._rigid_global_info.EPS, copy=True).numpy())

    print(f"Captured production state:")
    print(f"  qpos_pre = {qpos_pre}")
    print(f"  vel_pre  = {vel_pre}")
    print(f"  acc_post = {acc_post}")
    print(f"  dt={dt}, eps={eps:.3e}")
    print(f"  |cdofd_vel|_max = {np.abs(cdofd_vel).max():.3e}")
    print(f"  |cdof_vel|_max  = {np.abs(cdof_vel).max():.3e}")

    # === Step B: Replay N=2 backward up to begin_backward_substep, capture seed ===
    sb, rb = build(mjcf, True)
    solver_b = sb.rigid_solver
    captures = {}
    orig_dump = solver_b._debug_grad_dump

    def patched(tag):
        orig_dump(tag)
        if "after begin_backward_substep" in tag:
            captures.setdefault("seed", []).append({
                "qpos_next_grad": qd_to_torch(solver_b._rigid_global_info.qpos_next.grad, copy=True).numpy()[..., 0].copy(),
                "vel_next_grad": qd_to_torch(solver_b.dofs_state.vel_next.grad, copy=True).numpy()[..., 0].copy(),
            })

    solver_b._debug_grad_dump = patched

    u_list = [u0, rng.normal(size=n_dofs) * 0.3]
    u_anas = [gs.tensor(u, dtype=gs.tc_float, requires_grad=True) for u in u_list]
    sb.reset()
    for t in range(2):
        rb.control_dofs_force(u_anas[t])
        sb.step()
    from diag_multistep_worst_case import loss_fn
    loss_fn(sb).backward()

    # t=0 is the SECOND backward call (LIFO), index [1]
    seed = captures["seed"][1]
    qpos_next_grad = seed["qpos_next_grad"]
    vel_next_grad = seed["vel_next_grad"]
    print(f"\nCaptured production seed at t=0 backward:")
    print(f"  qpos_next.grad = {qpos_next_grad}")
    print(f"  vel_next.grad  = {vel_next_grad}")

    # === Step C: Run standalone combined kernel with production state + seed ===
    sc, rc = build(mjcf, True)
    solver_c = sc.rigid_solver
    sc.reset()

    # Inject production state into clean solver
    n_q = qpos_pre.shape[0]
    n_dof = vel_pre.shape[0]
    _B = 1
    solver_c._rigid_global_info.qpos.from_numpy(qpos_pre.reshape(n_q, _B).astype(np.float64))
    solver_c.dofs_state.vel.from_numpy(vel_pre.reshape(n_dof, _B).astype(np.float64))
    solver_c.dofs_state.acc.from_numpy(acc_post.reshape(n_dof, _B).astype(np.float64))
    solver_c.dofs_state.cdofd_vel.from_numpy(cdofd_vel)
    solver_c.dofs_state.cdofd_ang.from_numpy(cdofd_ang)
    solver_c.dofs_state.cdof_vel.from_numpy(cdof_vel)
    solver_c.dofs_state.cdof_ang.from_numpy(cdof_ang)
    solver_c._rigid_global_info.gravity.from_numpy(gravity)

    # Zero all grads
    for fld in [solver_c._rigid_global_info.qpos.grad,
                solver_c._rigid_global_info.qpos_next.grad,
                solver_c.dofs_state.vel.grad, solver_c.dofs_state.vel_next.grad,
                solver_c.dofs_state.acc.grad,
                solver_c.links_state.cdd_vel.grad, solver_c.links_state.cdd_ang.grad,
                solver_c.links_state.cacc_lin.grad, solver_c.links_state.cacc_ang.grad]:
        fld.from_numpy(np.zeros_like(fld.to_numpy()))

    # Zero output state
    solver_c._rigid_global_info.qpos_next.from_numpy(
        np.zeros_like(solver_c._rigid_global_info.qpos_next.to_numpy()))
    solver_c.dofs_state.vel_next.from_numpy(np.zeros_like(solver_c.dofs_state.vel_next.to_numpy()))
    for fld in [solver_c.links_state.cdd_vel, solver_c.links_state.cdd_ang,
                solver_c.links_state.cacc_lin, solver_c.links_state.cacc_ang]:
        fld.from_numpy(np.zeros_like(fld.to_numpy()))

    # Forward
    kernel_update_acc_plus_integrate_standalone(
        dofs_state=solver_c.dofs_state,
        links_info=solver_c.links_info,
        joints_info=solver_c.joints_info,
        links_state=solver_c.links_state,
        entities_info=solver_c.entities_info,
        rigid_global_info=solver_c._rigid_global_info,
        static_rigid_sim_config=solver_c._static_rigid_sim_config,
        is_backward=True,
    )

    # Seed grads
    solver_c._rigid_global_info.qpos_next.grad.from_numpy(
        qpos_next_grad.reshape(n_q, _B).astype(np.float64))
    solver_c.dofs_state.vel_next.grad.from_numpy(
        vel_next_grad.reshape(n_dof, _B).astype(np.float64))

    kernel_update_acc_plus_integrate_standalone.grad(
        dofs_state=solver_c.dofs_state,
        links_info=solver_c.links_info,
        joints_info=solver_c.joints_info,
        links_state=solver_c.links_state,
        entities_info=solver_c.entities_info,
        rigid_global_info=solver_c._rigid_global_info,
        static_rigid_sim_config=solver_c._static_rigid_sim_config,
        is_backward=True,
    )

    qg_s = qd_to_torch(solver_c._rigid_global_info.qpos.grad, copy=True).numpy()[..., 0]
    vg_s = qd_to_torch(solver_c.dofs_state.vel.grad, copy=True).numpy()[..., 0]
    ag_s = qd_to_torch(solver_c.dofs_state.acc.grad, copy=True).numpy()[..., 0]

    # === Step D: Manual reverse ===
    qg_m, vg_m, ag_m = manual_reverse(qpos_pre, vel_pre, acc_post, dt, eps,
                                       qpos_next_grad, vel_next_grad)

    # === Compare ===
    print("\n=== STANDALONE (production state) vs MANUAL ===")
    for name, s, m in [("qpos.grad", qg_s, qg_m),
                       ("vel.grad", vg_s, vg_m),
                       ("acc.grad", ag_s, ag_m)]:
        diff = s - m
        mx = float(np.abs(diff).max())
        print(f"{name}:")
        print(f"  standalone = {s}")
        print(f"  manual     = {m}")
        print(f"  diff       = {diff}")
        print(f"  max|diff|  = {mx:.3e}")


if __name__ == "__main__":
    main()
