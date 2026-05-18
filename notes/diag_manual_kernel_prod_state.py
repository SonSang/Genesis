"""Production-state inject test of `kernel_manual_func_integrate_bw`.

Goal: verify if our scalar-arithmetic manual kernel matches manual numpy when
running on production-captured state. If yes (max|d|=0), the silent drop comes
from kernel-call timing (cross-kernel chain at production state). If no
(max|d|~1e-11), the silent drop is from Quadrants's FMA fusion of our scalar
ops even with intermediate temps.
"""
import os
os.environ["GENESIS_DEBUG_GRAD"] = "0"
import sys
import numpy as np
import genesis as gs
gs.init(precision="64", backend=gs.cpu, performance_mode=False, logging_level="warning")

import quadrants as qd
from genesis.utils.misc import qd_to_torch
from genesis.engine.solvers.rigid.abd.manual_bw import kernel_manual_func_integrate_bw

sys.path.insert(0, "notes")
from diag_multistep_worst_case import TOPOLOGIES, build, loss_fn


def build_j4():
    name_map = {t[0]: t for t in TOPOLOGIES}
    _, mjcf, _ = name_map["J4_free_rev"]
    s, r = build(mjcf, True)
    return s, r


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
    c = np.cos(theta_reg/2); s = np.sin(theta_reg/2); sinc = s / theta_reg
    return np.array([c, sinc*rx, sinc*ry, sinc*rz])


def manual_numpy(qpos, vel, acc, dt, eps, qng, vng):
    vn = vel + dt * acc
    qg = np.zeros_like(qpos); vg = np.zeros_like(vel); ag = np.zeros_like(acc)
    qg[0:3] += qng[0:3]; vng2 = vng.copy()
    vng2[0:3] += dt * qng[0:3]
    ang = vn[3:6] * dt
    qrot = rotvec_to_quat(ang, eps)
    rot0 = qpos[3:7]
    bw, bx, by, bz = qrot
    ogw, ogx, ogy, ogz = qng[3:7]
    rot0_grad = np.array([
        ogw*bw + ogx*bx + ogy*by + ogz*bz,
        -ogw*bx + ogx*bw - ogy*bz + ogz*by,
        -ogw*by + ogx*bz + ogy*bw - ogz*bx,
        -ogw*bz - ogx*by + ogy*bx + ogz*bw,
    ])
    qg[3:7] += rot0_grad
    aw, ax, ay, az = rot0
    qrot_grad = np.array([
        ogw*aw + ogx*ax + ogy*ay + ogz*az,
        -ogw*ax + ogx*aw + ogy*az - ogz*ay,
        -ogw*ay - ogx*az + ogy*aw + ogz*ax,
        -ogw*az + ogx*ay - ogy*ax + ogz*aw,
    ])
    rx, ry, rz = ang
    tsq = rx*rx + ry*ry + rz*rz; tr = np.sqrt(tsq + eps*eps)
    sh = np.sin(tr/2); ch = np.cos(tr/2); sinc = sh/tr
    dsd = (0.5*ch - sinc)/tr
    qgw, qgx, qgy, qgz = qrot_grad
    qgr = qgx*rx + qgy*ry + qgz*rz
    coeff = -0.5*sh/tr*qgw + dsd/tr*qgr
    ang_grad = np.array([coeff*rx + sinc*qgx, coeff*ry + sinc*qgy, coeff*rz + sinc*qgz])
    vng2[3:6] += dt * ang_grad
    qg[7] += qng[7]; vng2[6] += dt * qng[7]
    vg += vng2; ag += dt * vng2
    return qg, vg, ag


def main():
    # Step 1: Capture production state from N=2 backward run
    s, r = build_j4()
    solver = s.rigid_solver
    captures = {}
    orig_dump = solver._debug_grad_dump

    def patched(tag):
        orig_dump(tag)
        if "after begin_backward_substep" in tag:
            captures.setdefault("inputs", []).append({
                "qpos": qd_to_torch(solver._rigid_global_info.qpos, copy=True).numpy()[..., 0].copy(),
                "vel_next": qd_to_torch(solver.dofs_state.vel_next, copy=True).numpy()[..., 0].copy(),
                "qpos_next_grad": qd_to_torch(solver._rigid_global_info.qpos_next.grad, copy=True).numpy()[..., 0].copy(),
                "vel_next_grad": qd_to_torch(solver.dofs_state.vel_next.grad, copy=True).numpy()[..., 0].copy(),
                "adj_qpos_f0": qd_to_torch(solver._rigid_adjoint_cache.qpos, copy=True).numpy()[0, :, 0].copy(),
            })

    solver._debug_grad_dump = patched
    rng = np.random.default_rng(1000)
    u_list = [rng.normal(size=7) * 0.3 for _ in range(2)]
    u_anas = [gs.tensor(u, dtype=gs.tc_float, requires_grad=True) for u in u_list]
    s.reset()
    for t in range(2):
        r.control_dofs_force(u_anas[t]); s.step()
    loss_fn(s).backward()

    # t=0 backward = index 1 (LIFO)
    inp = captures["inputs"][1]
    qpos_pre = inp["adj_qpos_f0"]  # pre-integrate qpos at step 0
    vel_next = inp["vel_next"]
    qpos_next_grad = inp["qpos_next_grad"]
    vel_next_grad = inp["vel_next_grad"]
    dt = float(qd_to_torch(solver._rigid_global_info.substep_dt, copy=True).numpy())
    eps = float(qd_to_torch(solver._rigid_global_info.EPS, copy=True).numpy())

    print(f"Production state at t=0 backward:")
    print(f"  qpos_pre  = {qpos_pre}")
    print(f"  vel_next  = {vel_next}")
    print(f"  qpos_next.grad = {qpos_next_grad}")
    print(f"  vel_next.grad  = {vel_next_grad}")

    # Step 2: Inject into standalone solver, call manual kernel directly
    s2, r2 = build_j4()
    solver2 = s2.rigid_solver
    s2.reset()

    n_q = 8; n_dof = 7; _B = 1
    solver2._rigid_global_info.qpos.from_numpy(qpos_pre.reshape(n_q, _B))
    solver2.dofs_state.vel_next.from_numpy(vel_next.reshape(n_dof, _B))
    adj = solver2._rigid_adjoint_cache.qpos.to_numpy()
    adj[0, :, 0] = qpos_pre
    solver2._rigid_adjoint_cache.qpos.from_numpy(adj)

    # Zero grads then seed
    for fld in [solver2._rigid_global_info.qpos.grad,
                solver2._rigid_global_info.qpos_next.grad,
                solver2.dofs_state.vel.grad, solver2.dofs_state.vel_next.grad,
                solver2.dofs_state.acc.grad]:
        fld.from_numpy(np.zeros_like(fld.to_numpy()))
    solver2._rigid_global_info.qpos_next.grad.from_numpy(qpos_next_grad.reshape(n_q, _B))
    solver2.dofs_state.vel_next.grad.from_numpy(vel_next_grad.reshape(n_dof, _B))

    kernel_manual_func_integrate_bw(
        f=0,
        dofs_state=solver2.dofs_state,
        links_info=solver2.links_info,
        joints_info=solver2.joints_info,
        entities_info=solver2.entities_info,
        rigid_global_info=solver2._rigid_global_info,
        rigid_adjoint_cache=solver2._rigid_adjoint_cache,
        static_rigid_sim_config=solver2._static_rigid_sim_config,
        errno=solver2._errno,
    )

    qg_k = solver2._rigid_global_info.qpos.grad.to_numpy()[..., 0]
    vg_k = solver2.dofs_state.vel.grad.to_numpy()[..., 0]
    ag_k = solver2.dofs_state.acc.grad.to_numpy()[..., 0]

    # Manual numpy with same inputs — but for fair comparison, vel passed in matches vel_pre
    # In func_integrate: vel_next = vel + dt*acc, so vel = vel_next - dt*acc.
    # We don't have vel_pre / acc — but we can use vel=0, acc=vel_next/dt to recover vel_next.
    # Simpler: directly use vel_next as input to manual numpy (skip vel→vel_next part).
    # Manual computation that mirrors kernel:
    # - kernel reads vel_next field, sets ang = vel_next[3..5] * dt.
    # - kernel writes to vel.grad and acc.grad via vel_next.grad += new contrib, then vel.grad += vel_next.grad, acc.grad += dt * vel_next.grad
    # Re-implement matching this:
    vel_pre = np.zeros(7); acc = vel_next / dt  # any choice, vel_next reconstructed
    qg_m, vg_m, ag_m = manual_numpy(qpos_pre, vel_pre, acc, dt, eps, qpos_next_grad, vel_next_grad)

    print("\n=== Manual KERNEL (scalar) vs Manual NUMPY (production state) ===")
    for name, k, m in [("qpos.grad", qg_k, qg_m),
                       ("vel.grad", vg_k, vg_m),
                       ("acc.grad", ag_k, ag_m)]:
        d = k - m; mx = float(np.abs(d).max())
        marker = " PASS" if mx < 1e-13 else " ✗ FAIL"
        print(f"{name}: kernel = {k}")
        print(f"{name}: numpy  = {m}")
        print(f"{name}: diff   = {d}")
        print(f"{name}: max|d| = {mx:.3e}{marker}")


if __name__ == "__main__":
    main()
