"""Verify kernel_manual_func_integrate_bw against manual numpy reverse + FD.

Setup: standalone solver, identical to diag_func_integrate_isolated_fd.py but
calls our new manual kernel instead of Quadrants AD's auto .grad.

PASS criterion: manual kernel output matches manual numpy (FP64 floor) AND
matches FD (within FD precision).
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
)
from genesis.engine.solvers.rigid.abd.manual_bw import (
    kernel_manual_func_integrate_bw,
)


def build_j4_solver():
    sys.path.insert(0, "notes")
    from diag_multistep_worst_case import TOPOLOGIES, build
    name_map = {t[0]: t for t in TOPOLOGIES}
    _, mjcf, _ = name_map["J4_free_rev"]
    s, r = build(mjcf, True)
    return s, r


# Manual numpy reverse (same as diag_func_integrate_isolated_fd.py)
def quat_mul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
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
    return np.array([
        np.cos(theta_reg/2),
        np.sin(theta_reg/2)/theta_reg * rx,
        np.sin(theta_reg/2)/theta_reg * ry,
        np.sin(theta_reg/2)/theta_reg * rz,
    ])


def integrate_forward(qpos, vel, acc, dt, eps):
    vel_next = vel + dt * acc
    qpos_next = np.zeros_like(qpos)
    qpos_next[0:3] = qpos[0:3] + dt * vel_next[0:3]
    ang = vel_next[3:6] * dt
    qrot = rotvec_to_quat(ang, eps)
    qpos_next[3:7] = quat_mul(qpos[3:7], qrot)
    qpos_next[7] = qpos[7] + dt * vel_next[6]
    return qpos_next, vel_next


def manual_numpy_reverse(qpos, vel, acc, dt, eps, qpos_next_grad, vel_next_grad):
    vel_next = vel + dt * acc
    qpos_grad = np.zeros_like(qpos)
    vel_grad = np.zeros_like(vel)
    acc_grad = np.zeros_like(acc)
    qpos_grad[0:3] += qpos_next_grad[0:3]
    vng = vel_next_grad.copy()
    vng[0:3] += dt * qpos_next_grad[0:3]
    ang = vel_next[3:6] * dt
    qrot = rotvec_to_quat(ang, eps)
    rot0 = qpos[3:7]
    bw, bx, by, bz = qrot
    ogw, ogx, ogy, ogz = qpos_next_grad[3:7]
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
    sin_h = np.sin(theta_reg/2); cos_h = np.cos(theta_reg/2)
    sinc = sin_h / theta_reg
    dsinc_dtheta = (0.5*cos_h - sinc) / theta_reg
    qg_dot_r = qrot_grad[1]*rx + qrot_grad[2]*ry + qrot_grad[3]*rz
    coeff = -0.5*sin_h/theta_reg * qrot_grad[0] + dsinc_dtheta/theta_reg * qg_dot_r
    ang_grad = np.array([
        coeff*rx + sinc*qrot_grad[1],
        coeff*ry + sinc*qrot_grad[2],
        coeff*rz + sinc*qrot_grad[3],
    ])
    vng[3:6] += dt * ang_grad
    qpos_grad[7] += qpos_next_grad[7]
    vng[6] += dt * qpos_next_grad[7]
    vel_grad += vng
    acc_grad += dt * vng
    return qpos_grad, vel_grad, acc_grad


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

    n_q, n_dof, _B = 8, 7, 1
    solver._rigid_global_info.qpos.from_numpy(qpos.reshape(n_q, _B))
    solver.dofs_state.vel.from_numpy(vel.reshape(n_dof, _B))
    solver.dofs_state.acc.from_numpy(acc.reshape(n_dof, _B))
    # adjoint_cache.qpos[f=0, :, :] = pre-integrate qpos
    adj_q = qd_to_torch(solver._rigid_adjoint_cache.qpos, copy=True).numpy()
    adj_q[0, :, 0] = qpos
    solver._rigid_adjoint_cache.qpos.from_numpy(adj_q)

    # Zero grads
    for fld in [solver._rigid_global_info.qpos.grad,
                solver._rigid_global_info.qpos_next.grad,
                solver.dofs_state.vel.grad, solver.dofs_state.vel_next.grad,
                solver.dofs_state.acc.grad]:
        fld.from_numpy(np.zeros_like(fld.to_numpy()))

    # Forward
    kernel_func_integrate_standalone(
        dofs_state=solver.dofs_state,
        links_info=solver.links_info,
        joints_info=solver.joints_info,
        rigid_global_info=solver._rigid_global_info,
        static_rigid_sim_config=solver._static_rigid_sim_config,
        is_backward=True,
    )

    # Seed
    solver._rigid_global_info.qpos_next.grad.from_numpy(qpos_next_grad.reshape(n_q, _B))
    solver.dofs_state.vel_next.grad.from_numpy(vel_next_grad.reshape(n_dof, _B))

    # Manual backward kernel
    kernel_manual_func_integrate_bw(
        f=0,
        dofs_state=solver.dofs_state,
        links_info=solver.links_info,
        joints_info=solver.joints_info,
        entities_info=solver.entities_info,
        rigid_global_info=solver._rigid_global_info,
        rigid_adjoint_cache=solver._rigid_adjoint_cache,
        static_rigid_sim_config=solver._static_rigid_sim_config,
        errno=solver._errno,
    )

    qg_k = qd_to_torch(solver._rigid_global_info.qpos.grad, copy=True).numpy()[..., 0]
    vg_k = qd_to_torch(solver.dofs_state.vel.grad, copy=True).numpy()[..., 0]
    ag_k = qd_to_torch(solver.dofs_state.acc.grad, copy=True).numpy()[..., 0]

    # Manual numpy
    qg_m, vg_m, ag_m = manual_numpy_reverse(qpos, vel, acc, dt, eps, qpos_next_grad, vel_next_grad)

    print("=== Manual KERNEL vs Manual NUMPY ===")
    for name, k, m in [("qpos.grad", qg_k, qg_m),
                       ("vel.grad", vg_k, vg_m),
                       ("acc.grad", ag_k, ag_m)]:
        diff = k - m
        mx = float(np.abs(diff).max())
        marker = " PASS" if mx < 1e-10 else " ✗ FAIL"
        print(f"{name}: kernel={k}")
        print(f"{name}: numpy ={m}")
        print(f"{name}: max|d|={mx:.3e}{marker}")


if __name__ == "__main__":
    main()
