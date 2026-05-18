"""Manual verification of step_2.grad (integrator backward) on J4 N=2 t=0.

Forward (free joint, rotation part, single substep at step 0):
  vel_next[i] = vel[i] + dt * acc[i]                              (all DOFs)
  pos_next[0..2] = pos[0..2] + dt * vel_next[0..2]                (translation)
  ang = vel_next[3..5] * dt
  qrot = rotvec_to_quat(ang, EPS)
  q_next[3..6] = quat_mul(qrot, q[3..6])                          (rotation)
  q_next[7]_arm = q[7]_arm + dt * vel_next[6]                     (revolute)

Reverse (analytical, what step_2.grad should compute):
  Input: qpos_next.grad (= post-UCS.grad's qpos.grad), vel_next.grad (= post-UCS.grad's vel.grad)
  Output: qpos.grad (pre-integrate), vel.grad (pre-integrate), acc.grad

We capture:
  - forward primal (qpos_pre = initial, vel_next, dt) via a separate N=1 sim
  - input grads (qpos_next.grad, vel_next.grad) via the existing verbose dump

Then we manually compute the reverse and compare with step_2.grad's output.
"""

import os
os.environ["GENESIS_DEBUG_GRAD"] = "1"

import sys
import numpy as np

sys.path.insert(0, "notes")
from diag_all_topo_relerror_sweep import measure  # noqa: F401
from diag_multistep_worst_case import TOPOLOGIES, build, loss_fn
from genesis.utils.misc import qd_to_torch


def quat_mul(a, b):
    """Hamilton convention: out = a * b."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])


def quat_mul_jac_a(a, b):
    """d(quat_mul(a, b))/d(a) — 4x4 matrix.
    Rows = output components (w, x, y, z), columns = a's components.
    """
    bw, bx, by, bz = b
    return np.array([
        [ bw, -bx, -by, -bz],
        [ bx,  bw,  bz, -by],
        [ by, -bz,  bw,  bx],
        [ bz,  by, -bx,  bw],
    ])


def quat_mul_jac_b(a, b):
    """d(quat_mul(a, b))/d(b) — 4x4 matrix."""
    aw, ax, ay, az = a
    return np.array([
        [ aw, -ax, -ay, -az],
        [ ax,  aw, -az,  ay],
        [ ay,  az,  aw, -ax],
        [ az, -ay,  ax,  aw],
    ])


def rotvec_to_quat(rotvec, eps):
    """Forward: rotvec → quat (Hamilton). Matches geom.py qd_rotvec_to_quat."""
    rx, ry, rz = rotvec
    thetasq = rx*rx + ry*ry + rz*rz
    theta_reg = np.sqrt(thetasq + eps*eps)
    cos_h = np.cos(theta_reg / 2.0)
    sin_h = np.sin(theta_reg / 2.0)
    sinc = sin_h / theta_reg
    return np.array([cos_h, sinc*rx, sinc*ry, sinc*rz])


def rotvec_to_quat_jac(rotvec, eps):
    """d(qrot)/d(rotvec) — 4x3 matrix."""
    rx, ry, rz = rotvec
    thetasq = rx*rx + ry*ry + rz*rz
    theta_reg = np.sqrt(thetasq + eps*eps)
    theta_half = 0.5 * theta_reg
    cos_h = np.cos(theta_half)
    sin_h = np.sin(theta_half)
    sinc = sin_h / theta_reg
    dsinc_dtheta = (0.5*cos_h - sinc) / theta_reg

    J = np.zeros((4, 3))
    for j, r_j in enumerate([rx, ry, rz]):
        # d(qrot.w)/d(r_j) = -0.5 * sin_h * r_j / theta_reg
        J[0, j] = -0.5 * sin_h * r_j / theta_reg
        # d(qrot.x_or_y_or_z[i])/d(r_j) = δ(i, j) * sinc + r_i * dsinc_dtheta * r_j / theta_reg
        for i, r_i in enumerate([rx, ry, rz]):
            base = sinc if i == j else 0.0
            J[1+i, j] = base + r_i * dsinc_dtheta * r_j / theta_reg
    return J


def main():
    import genesis as gs
    gs.init(precision="64", backend=gs.cpu, performance_mode=False, logging_level="info")

    name_map = {t[0]: t for t in TOPOLOGIES}
    _, mjcf, n_dofs = name_map["J4_free_rev"]

    # === Step A: Capture forward primal via N=1 sim ===
    sb, rb = build(mjcf, True)
    seed = 1000
    rng = np.random.default_rng(seed)
    u_list = [rng.normal(size=n_dofs) * 0.3 for _ in range(2)]  # match N=2 seed
    u0 = u_list[0]

    sb.reset()
    # Capture qpos_pre (= initial state)
    qpos_pre = qd_to_torch(sb.rigid_solver._rigid_global_info.qpos, copy=True).numpy()[..., 0]
    vel_pre = qd_to_torch(sb.rigid_solver.dofs_state.vel, copy=True).numpy()[..., 0]
    print(f"qpos_pre = {qpos_pre}")
    print(f"vel_pre  = {vel_pre}")

    rb.control_dofs_force(gs.tensor(u0, dtype=gs.tc_float))
    sb.step()

    # Capture post-step values
    qpos_post = qd_to_torch(sb.rigid_solver._rigid_global_info.qpos, copy=True).numpy()[..., 0]
    vel_post = qd_to_torch(sb.rigid_solver.dofs_state.vel, copy=True).numpy()[..., 0]
    acc_post = qd_to_torch(sb.rigid_solver.dofs_state.acc, copy=True).numpy()[..., 0]
    dt = float(qd_to_torch(sb.rigid_solver._rigid_global_info.substep_dt, copy=True).numpy())
    eps = float(qd_to_torch(sb.rigid_solver._rigid_global_info.EPS, copy=True).numpy())
    print(f"qpos_post (=qpos_next) = {qpos_post}")
    print(f"vel_post  (=vel_next ) = {vel_post}")
    print(f"acc_post = {acc_post}")
    print(f"dt = {dt}, EPS = {eps:.3e}")

    # Reconstruct vel_next (= vel_pre + dt * acc) — should match vel_post
    vel_next_recon = vel_pre + dt * acc_post
    print(f"vel_next reconstructed = {vel_next_recon}")
    print(f"  matches vel_post? max|diff|={float(np.abs(vel_next_recon - vel_post).max()):.3e}")

    # === Step B: Replay N=2 backward and capture step_2.grad in/out ===
    sa, ra = build(mjcf, True)
    captures = {}
    solver = sa.rigid_solver
    orig_dump = solver._debug_grad_dump

    def patched(tag):
        orig_dump(tag)
        if "after begin_backward_substep" in tag:
            captures.setdefault("step_2_input", []).append({
                "qpos_next_grad": qd_to_torch(solver._rigid_global_info.qpos_next.grad, copy=True).numpy()[..., 0],
                "vel_next_grad": qd_to_torch(solver.dofs_state.vel_next.grad, copy=True).numpy()[..., 0],
                # Capture vel_next *field* — Quadrants forward's actual output
                # (uses FMA: vel_next = vel + acc*dt is a single fused instruction).
                # Without this, the python `vel_pre + dt * acc_post` reconstruction
                # introduces a ~1e-12 FP-order diff that propagates through ang →
                # qrot → quat_mul into the q_pre_grad output (~1e-11).
                "vel_next": qd_to_torch(solver.dofs_state.vel_next, copy=True).numpy()[..., 0],
            })
        if "after step_2.grad" in tag:
            captures.setdefault("step_2_output", []).append({
                "qpos_grad": qd_to_torch(solver._rigid_global_info.qpos.grad, copy=True).numpy()[..., 0],
                "vel_grad": qd_to_torch(solver.dofs_state.vel.grad, copy=True).numpy()[..., 0],
                "acc_grad": qd_to_torch(solver.dofs_state.acc.grad, copy=True).numpy()[..., 0],
            })

    solver._debug_grad_dump = patched

    u_anas = [gs.tensor(u, dtype=gs.tc_float, requires_grad=True) for u in u_list]
    sa.reset()
    for t in range(2):
        ra.control_dofs_force(u_anas[t])
        sa.step()
    loss_fn(sa).backward()

    # t=0 step_2 is the SECOND occurrence (index 1)
    inp = captures["step_2_input"][1]
    out = captures["step_2_output"][1]
    qpos_next_grad = inp["qpos_next_grad"]
    vel_next_grad_input = inp["vel_next_grad"]
    vel_next_captured = inp["vel_next"]  # Quadrants forward's actual vel_next
    qpos_grad_kernel = out["qpos_grad"]
    vel_grad_kernel = out["vel_grad"]
    acc_grad_kernel = out["acc_grad"]

    print("\n=== step_2.grad input (post-begin_backward_substep at t=0) ===")
    print(f"qpos_next.grad = {qpos_next_grad}")
    print(f"vel_next.grad  = {vel_next_grad_input}")

    # === Step C: Manual reverse ===
    # Forward chain:
    #   vel_next[i] = vel[i] + dt * acc[i]
    #   pos_next[0..2] = pos[0..2] + dt * vel_next[0..2]
    #   ang = vel_next[3..5] * dt
    #   qrot = rotvec_to_quat(ang, EPS)
    #   q_next[3..6] = quat_mul(qrot, q[3..6])
    #   q_next[7] = q[7] + dt * vel_next[6]  (revolute, treat like translation)

    # Initialize
    qpos_grad_manual = np.zeros_like(qpos_pre)  # 8 entries
    vel_next_grad_manual = vel_next_grad_input.copy()  # 7 entries, may receive additions

    # Translation (qpos[0..2] = qpos_pre[0..2], qpos_next[0..2] = qpos_pre[0..2] + dt*vel_next[0..2])
    for i in range(3):
        qpos_grad_manual[i] += qpos_next_grad[i]
        vel_next_grad_manual[i] += dt * qpos_next_grad[i]

    # Revolute (qpos[7] = qpos_pre[7], qpos_next[7] = qpos_pre[7] + dt*vel_next[6])
    qpos_grad_manual[7] += qpos_next_grad[7]
    vel_next_grad_manual[6] += dt * qpos_next_grad[7]

    # Rotation: qpos_next[3..6] = quat_mul(qrot, qpos[3..6])
    q_pre = qpos_pre[3:7]
    # Use the *captured* vel_next field (Quadrants forward's actual output) so the
    # ang/qrot primal matches what kernel_step_2.grad's internal reverse sees.
    # Reconstructing as `vel_pre + dt * acc_post` introduces a small FP-order
    # diff (Quadrants uses FMA for `vel + acc*dt`; numpy uses mul-then-add).
    ang = vel_next_captured[3:6] * dt
    qrot = rotvec_to_quat(ang, eps)
    print(f"\nForward primal for rotation:")
    print(f"  ang  = {ang}")
    print(f"  qrot = {qrot}")
    print(f"  q_pre = {q_pre}")

    # Reverse quat_mul (rot = quat_mul(q_pre, qrot)) — EXPLICIT SCALAR FORM.
    # Matches `kernel_manual_func_integrate_bw` and `func_integrate.grad` of
    # Quadrants AD which both compute mul-then-add in strict IEEE 754 order.
    # The earlier `J_a.T @ vec` / `J_b.T @ vec` matrix form uses numpy BLAS
    # which can fuse to FMA and produce a ~1e-11 numerically different value
    # (same math, different FP order). See diffrigid_handoff_n_ge_2_residual.md
    # rev 4 for the falsification of the "step_2 silent drop" hypothesis.
    qpos_next_grad_rot = qpos_next_grad[3:7]
    ogw, ogx, ogy, ogz = qpos_next_grad_rot
    aw, ax, ay, az = q_pre
    bw, bx, by, bz = qrot

    # d_quat_mul/d(a) — explicit scalar, strict left-to-right mul-then-add
    q_pre_grad = np.array([
        ogw*bw + ogx*bx + ogy*by + ogz*bz,
        -ogw*bx + ogx*bw - ogy*bz + ogz*by,
        -ogw*by + ogx*bz + ogy*bw - ogz*bx,
        -ogw*bz - ogx*by + ogy*bx + ogz*bw,
    ])

    # d_quat_mul/d(b)
    qrot_grad = np.array([
        ogw*aw + ogx*ax + ogy*ay + ogz*az,
        -ogw*ax + ogx*aw + ogy*az - ogz*ay,
        -ogw*ay - ogx*az + ogy*aw + ogz*ax,
        -ogw*az + ogx*ay - ogy*ax + ogz*aw,
    ])
    qpos_grad_manual[3:7] += q_pre_grad

    # === Decompose q_pre.x.grad into 4 contributions (sanity print) ===
    print(f"\n=== Decompose q_pre.x.grad (explicit scalar) ===")
    j_a_T_row1 = np.array([-bx, bw, -bz, by])
    print(f"J_a^T row 1 = {j_a_T_row1}  (entries = -bx, bw, -bz, by)")
    print(f"q_next.grad = {qpos_next_grad_rot}  (w, x, y, z)")
    for k in range(4):
        contrib = j_a_T_row1[k] * qpos_next_grad_rot[k]
        print(f"  contrib from q_next.grad[{k}] = {j_a_T_row1[k]:.4e} * {qpos_next_grad_rot[k]:.4e} = {contrib:.4e}")
    print(f"Sum (= manual q_pre.x.grad) = {q_pre_grad[1]:.4e}")
    print(f"Kernel q_pre.x.grad         = {qpos_grad_kernel[4]:.4e}")

    # Reverse rotvec_to_quat — EXPLICIT SCALAR FORM.
    # Matches `d_rotvec_to_quat__drotvec` in manual_bw.py.
    rx, ry, rz = ang
    thetasq = rx*rx + ry*ry + rz*rz
    theta_reg = np.sqrt(thetasq + eps*eps)
    theta_half = 0.5 * theta_reg
    sin_h = np.sin(theta_half)
    cos_h = np.cos(theta_half)
    sinc = sin_h / theta_reg
    dsinc_dtheta = (0.5*cos_h - sinc) / theta_reg

    qg_w, qg_x, qg_y, qg_z = qrot_grad
    qg_dot_r = qg_x*rx + qg_y*ry + qg_z*rz
    # coeff = -0.5*sin_h/theta_reg * qg_w + dsinc_dtheta/theta_reg * qg_dot_r
    coeff = -0.5 * sin_h / theta_reg * qg_w + dsinc_dtheta / theta_reg * qg_dot_r
    ang_grad = np.array([
        coeff*rx + sinc*qg_x,
        coeff*ry + sinc*qg_y,
        coeff*rz + sinc*qg_z,
    ])

    # vel_next[3..5].grad += dt * ang.grad
    vel_next_grad_manual[3:6] += dt * ang_grad

    # Now vel.grad and acc.grad from vel_next = vel + dt*acc:
    vel_grad_manual = vel_next_grad_manual.copy()  # vel.grad += vel_next.grad
    acc_grad_manual = dt * vel_next_grad_manual    # acc.grad += dt * vel_next.grad

    # === Compare ===
    print("\n=== Compare manual vs kernel step_2.grad output ===")
    print(f"{'name':<14} {'manual':<48} {'kernel':<48}")
    for name, m, k in [
        ("qpos.grad", qpos_grad_manual, qpos_grad_kernel),
        ("vel.grad", vel_grad_manual, vel_grad_kernel),
        ("acc.grad", acc_grad_manual, acc_grad_kernel),
    ]:
        print(f"{name}:")
        print(f"  manual = {m}")
        print(f"  kernel = {k}")
        diff = m - k
        print(f"  diff   = {diff}")
        print(f"  max|d| = {float(np.abs(diff).max()):.3e}")


if __name__ == "__main__":
    main()
