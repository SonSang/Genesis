"""Isolated FD verify of `func_integrate` reverse on J4 (free + revolute).

Forward chain (single-substep, FREE + revolute):
  1. vel_next[i] = vel[i] + dt * acc[i]                                       (line 1275-1277 fwd_dyn.py)
  2. translation:  qpos_next[0..2] = qpos[0..2] + dt * vel_next[0..2]         (line 1318-1330)
  3. rotation:     ang  = vel_next[3..5] * dt
                   qrot = qd_rotvec_to_quat(ang, EPS)
                   rot  = qd_transform_quat_by_quat(qrot, qpos[3..6])
                         = qd_quat_mul(qpos[3..6], qrot)
                   qpos_next[3..6] = rot                                       (line 1331-1354)
  4. revolute:     qpos_next[7] = qpos[7] + dt * vel_next[6]                   (line 1355-1362)

Reverse (analytical, what we'd expect from func_integrate.grad):
  d(qpos[0..2])     += d(qpos_next[0..2])
  d(vel_next[0..2]) += dt * d(qpos_next[0..2])

  d(qpos[3..6]) += J_a^T(qpos[3..6], qrot) @ d(qpos_next[3..6])
  d(qrot)        = J_b^T(qpos[3..6], qrot) @ d(qpos_next[3..6])
  d(ang)         = J_rotvec_to_quat^T(ang) @ d(qrot)
  d(vel_next[3..5]) += dt * d(ang)

  d(qpos[7])    += d(qpos_next[7])
  d(vel_next[6]) += dt * d(qpos_next[7])

  d(vel[i]) += d(vel_next[i])
  d(acc[i]) += dt * d(vel_next[i])

We exercise this with:
  A. Quadrants AD `kernel_func_integrate_standalone.grad` (production state)
  B. Manual numpy reverse (analytical)
  C. FD on the forward (perturb vel/acc/qpos, measure qpos_next/vel_next change)

If B agrees with C to FP64 floor (~1e-11), we have a correct manual reverse.
If A disagrees with B by ~1e-11, the production kernel.grad has the silent drop
we've been hunting.
"""
import os
os.environ["GENESIS_DEBUG_GRAD"] = "0"

import numpy as np
import genesis as gs

gs.init(precision="64", backend=gs.cpu, performance_mode=False, logging_level="warning")

import quadrants as qd
import genesis.utils.geom as gu
from genesis.utils.misc import qd_to_torch


# =============================================================================
# Numpy helpers (manual computation)
# =============================================================================
EPS = 1e-12  # matches rigid_global_info.EPS default


def quat_mul(a, b):
    """Hamilton: out = a * b, matches qd_quat_mul."""
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


def integrate_forward(qpos, vel, acc, dt, eps):
    """Reproduce func_integrate forward (J4: free + revolute, n_q=8, n_dof=7)."""
    vel_next = vel + dt * acc
    qpos_next = np.zeros_like(qpos)
    # Translation
    qpos_next[0:3] = qpos[0:3] + dt * vel_next[0:3]
    # Rotation
    ang = vel_next[3:6] * dt
    qrot = rotvec_to_quat(ang, eps)
    rot0 = qpos[3:7]
    rot = quat_mul(rot0, qrot)  # qd_transform_quat_by_quat(qrot, rot0) = quat_mul(rot0, qrot)
    qpos_next[3:7] = rot
    # Revolute
    qpos_next[7] = qpos[7] + dt * vel_next[6]
    return qpos_next, vel_next


def integrate_manual_reverse(qpos, vel, acc, dt, eps, qpos_next_grad, vel_next_grad):
    """Manual reverse of func_integrate."""
    vel_next = vel + dt * acc
    qpos_grad = np.zeros_like(qpos)
    vel_grad = np.zeros_like(vel)
    acc_grad = np.zeros_like(acc)

    # Translation reverse
    qpos_grad[0:3] += qpos_next_grad[0:3]
    vel_next_grad_local = vel_next_grad.copy()
    vel_next_grad_local[0:3] += dt * qpos_next_grad[0:3]

    # Rotation reverse
    ang = vel_next[3:6] * dt
    qrot = rotvec_to_quat(ang, eps)
    rot0 = qpos[3:7]
    # quat_mul jac (a=rot0, b=qrot)
    rot_grad = qpos_next_grad[3:7]
    # d_quat_mul__dlhs(rot0, qrot) — d(out)/d(a) where out = a*b
    bw, bx, by, bz = qrot
    ogw, ogx, ogy, ogz = rot_grad
    rot0_grad = np.array([
        ogw*bw + ogx*bx + ogy*by + ogz*bz,
        -ogw*bx + ogx*bw - ogy*bz + ogz*by,
        -ogw*by + ogx*bz + ogy*bw - ogz*bx,
        -ogw*bz - ogx*by + ogy*bx + ogz*bw,
    ])
    qpos_grad[3:7] += rot0_grad

    # d_quat_mul__drhs(rot0, qrot)
    aw, ax, ay, az = rot0
    qrot_grad = np.array([
        ogw*aw + ogx*ax + ogy*ay + ogz*az,
        -ogw*ax + ogx*aw + ogy*az - ogz*ay,
        -ogw*ay - ogx*az + ogy*aw + ogz*ax,
        -ogw*az + ogx*ay - ogy*ax + ogz*aw,
    ])

    # rotvec_to_quat reverse: d(rotvec) = J^T @ d(quat)
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

    # Revolute reverse
    qpos_grad[7] += qpos_next_grad[7]
    vel_next_grad_local[6] += dt * qpos_next_grad[7]

    # vel_next = vel + dt*acc reverse
    vel_grad += vel_next_grad_local
    acc_grad += dt * vel_next_grad_local

    return qpos_grad, vel_grad, acc_grad


# =============================================================================
# Quadrants AD path
# =============================================================================
def build_j4_solver():
    """Build a J4 (free + revolute) entity solver, return solver + entity."""
    import sys
    sys.path.insert(0, "notes")
    from diag_multistep_worst_case import TOPOLOGIES, build
    name_map = {t[0]: t for t in TOPOLOGIES}
    _, mjcf, _ = name_map["J4_free_rev"]
    s, r = build(mjcf, True)
    return s, r


def run_quadrants_grad(qpos, vel, acc, qpos_next_grad, vel_next_grad):
    """Run forward + standalone func_integrate kernel + .grad. Return qpos/vel/acc grad."""
    s, r = build_j4_solver()
    solver = s.rigid_solver
    s.reset()

    n_q = qpos.shape[0]
    n_dof = vel.shape[0]
    _B = 1

    # Set inputs (single-env)
    qpos_np = qpos.reshape(n_q, _B).astype(np.float64)
    vel_np = vel.reshape(n_dof, _B).astype(np.float64)
    acc_np = acc.reshape(n_dof, _B).astype(np.float64)
    solver._rigid_global_info.qpos.from_numpy(qpos_np)
    solver.dofs_state.vel.from_numpy(vel_np)
    solver.dofs_state.acc.from_numpy(acc_np)

    # Zero all grads first
    solver._rigid_global_info.qpos.grad.from_numpy(np.zeros_like(qpos_np))
    solver._rigid_global_info.qpos_next.grad.from_numpy(np.zeros_like(qpos_np))
    solver.dofs_state.vel.grad.from_numpy(np.zeros_like(vel_np))
    solver.dofs_state.vel_next.grad.from_numpy(np.zeros_like(vel_np))
    solver.dofs_state.acc.grad.from_numpy(np.zeros_like(acc_np))

    # Forward
    from genesis.engine.solvers.rigid.abd.forward_dynamics import kernel_func_integrate_standalone
    kernel_func_integrate_standalone(
        dofs_state=solver.dofs_state,
        links_info=solver.links_info,
        joints_info=solver.joints_info,
        rigid_global_info=solver._rigid_global_info,
        static_rigid_sim_config=solver._static_rigid_sim_config,
        is_backward=True,
    )

    # Capture forward output (for comparison with manual)
    qpos_next_fwd = qd_to_torch(solver._rigid_global_info.qpos_next, copy=True).numpy()[..., 0]
    vel_next_fwd = qd_to_torch(solver.dofs_state.vel_next, copy=True).numpy()[..., 0]

    # Seed grads
    solver._rigid_global_info.qpos_next.grad.from_numpy(qpos_next_grad.reshape(n_q, _B).astype(np.float64))
    solver.dofs_state.vel_next.grad.from_numpy(vel_next_grad.reshape(n_dof, _B).astype(np.float64))

    # Backward
    kernel_func_integrate_standalone.grad(
        dofs_state=solver.dofs_state,
        links_info=solver.links_info,
        joints_info=solver.joints_info,
        rigid_global_info=solver._rigid_global_info,
        static_rigid_sim_config=solver._static_rigid_sim_config,
        is_backward=True,
    )

    qpos_grad = qd_to_torch(solver._rigid_global_info.qpos.grad, copy=True).numpy()[..., 0]
    vel_grad = qd_to_torch(solver.dofs_state.vel.grad, copy=True).numpy()[..., 0]
    acc_grad = qd_to_torch(solver.dofs_state.acc.grad, copy=True).numpy()[..., 0]
    dt = float(qd_to_torch(solver._rigid_global_info.substep_dt, copy=True).numpy())
    eps = float(qd_to_torch(solver._rigid_global_info.EPS, copy=True).numpy())
    return qpos_grad, vel_grad, acc_grad, qpos_next_fwd, vel_next_fwd, dt, eps


# =============================================================================
# Finite difference
# =============================================================================
def fd_grad(qpos, vel, acc, dt, eps, qpos_next_grad, vel_next_grad, h=1e-6):
    """FD: L = qpos_next_grad . qpos_next + vel_next_grad . vel_next.
    Compute dL/dqpos, dL/dvel, dL/dacc by finite diff on the manual forward.
    """
    def L(q, v, a):
        qn, vn = integrate_forward(q, v, a, dt, eps)
        return float(qpos_next_grad @ qn + vel_next_grad @ vn)

    qpos_grad_fd = np.zeros_like(qpos)
    for i in range(len(qpos)):
        qp = qpos.copy(); qp[i] += h
        qm = qpos.copy(); qm[i] -= h
        qpos_grad_fd[i] = (L(qp, vel, acc) - L(qm, vel, acc)) / (2*h)

    vel_grad_fd = np.zeros_like(vel)
    for i in range(len(vel)):
        vp = vel.copy(); vp[i] += h
        vm = vel.copy(); vm[i] -= h
        vel_grad_fd[i] = (L(qpos, vp, acc) - L(qpos, vm, acc)) / (2*h)

    acc_grad_fd = np.zeros_like(acc)
    for i in range(len(acc)):
        ap = acc.copy(); ap[i] += h
        am = acc.copy(); am[i] -= h
        acc_grad_fd[i] = (L(qpos, vel, ap) - L(qpos, vel, am)) / (2*h)

    return qpos_grad_fd, vel_grad_fd, acc_grad_fd


# =============================================================================
# Main
# =============================================================================
def main():
    rng = np.random.default_rng(1000)
    # Quat-friendly initial state
    quat0 = np.array([1.0, 0.001, -0.0005, 0.0003]); quat0 /= np.linalg.norm(quat0)
    qpos = np.array([0.1, -0.05, 0.2, quat0[0], quat0[1], quat0[2], quat0[3], 0.05])
    vel = rng.normal(size=7) * 0.1
    acc = rng.normal(size=7) * 1.0

    # Random output seeds
    qpos_next_grad = rng.normal(size=8) * 1e-3
    vel_next_grad = rng.normal(size=7) * 1e-4

    # === Quadrants AD ===
    qg_q, vg_q, ag_q, qn_fwd, vn_fwd, dt, eps_real = run_quadrants_grad(
        qpos, vel, acc, qpos_next_grad, vel_next_grad)
    print(f"dt = {dt:.6e}, eps = {eps_real:.3e}")
    print(f"qpos_next (Quadrants forward) = {qn_fwd}")
    print(f"vel_next  (Quadrants forward) = {vn_fwd}")

    # === Manual forward (sanity) ===
    qn_man, vn_man = integrate_forward(qpos, vel, acc, dt, eps_real)
    print(f"qpos_next (manual numpy)      = {qn_man}")
    print(f"vel_next  (manual numpy)      = {vn_man}")
    print(f"forward match? max|diff| qpos_next = {np.abs(qn_fwd - qn_man).max():.3e}, "
          f"vel_next = {np.abs(vn_fwd - vn_man).max():.3e}")

    # === Manual reverse ===
    qg_m, vg_m, ag_m = integrate_manual_reverse(
        qpos, vel, acc, dt, eps_real, qpos_next_grad, vel_next_grad)

    # === FD ===
    qg_f, vg_f, ag_f = fd_grad(qpos, vel, acc, dt, eps_real, qpos_next_grad, vel_next_grad)

    # === Compare ===
    print("\n=== Manual vs FD (FP64 floor target ~1e-9) ===")
    for name, m, f in [("qpos.grad", qg_m, qg_f),
                       ("vel.grad", vg_m, vg_f),
                       ("acc.grad", ag_m, ag_f)]:
        diff = m - f
        mx = float(np.abs(diff).max())
        print(f"{name}: manual = {m}")
        print(f"{name}:    fd   = {f}")
        print(f"{name}: max|diff| = {mx:.3e}")

    print("\n=== Quadrants kernel.grad vs Manual ===")
    for name, k, m in [("qpos.grad", qg_q, qg_m),
                       ("vel.grad", vg_q, vg_m),
                       ("acc.grad", ag_q, ag_m)]:
        diff = k - m
        mx = float(np.abs(diff).max())
        print(f"{name}: kernel = {k}")
        print(f"{name}: manual = {m}")
        print(f"{name}: max|diff| = {mx:.3e}")


if __name__ == "__main__":
    main()
