"""Minimal repro — N-step FK chain: reverse-mode gradient diverges from FD.

# What this script shows

A single-substep FK update written as

    quat[t+1] = quat_mul(quat[t], rotvec_to_quat(omega * dt))
    pos[t+1]  = pos[t] + transform_by_quat(arm_local, quat[t])

is iterated for N substeps, then we ask Quadrants reverse-mode for
`d (sum pos[N]) / d omega`. We compare the analytical gradient against
a Richardson (O(h^4)) finite-difference reference.

# Observed behavior

  arm_local = [0, 0, 0]   (no arm offset → no chain through transform_by_quat):
    abs_diff = 0 for all N.

  arm_local = [0.2, 0, 0]:
    N=2    abs_diff ~ 4e-12
    N=4    abs_diff ~ 1e-11
    N=8    abs_diff ~ 1e-10
    N=16   abs_diff ~ 4e-10

  arm_local = [1.0, 0, 0]:
    abs_diff scales ~linearly with |arm_local| (≈5x larger throughout).

So:
  - The single-substep gradient matches FD to FP64 floor.
  - As substeps accumulate, the analytical-vs-FD gap grows.
  - The gap is proportional to |arm_local| (i.e. it comes from the
    transform_by_quat path, not the rotation update on its own).

# What we have tried

  - Replaced the Quadrants reverse with a hand-written numpy reverse of
    `transform_by_quat`: byte-exact match with Quadrants reverse at
    every substep.
  - Rewrote the manual reverse with explicit scalar mul-then-add (no
    qd.Vector arithmetic in the reverse formula): byte-exact identical.
  - Rewrote the forward with explicit intermediate temps to control
    expression structure: byte-exact identical output.

None of these changes moved the analytical-vs-FD gap.

Run:
    python notes/quadrants_repros/case_n_step_fk_chain_drift.py
"""

import numpy as np
import quadrants as qd


qd.init(arch=qd.cpu)


DT = 0.01
EPS = 1e-8


# ---------- forward primitives (copied verbatim from genesis/utils/geom.py) ----------


@qd.func
def quat_mul(a, b):
    return qd.Vector(
        [
            a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
        ],
        dt=qd.f64,
    )


@qd.func
def rotvec_to_quat(rotvec):
    rx = rotvec[0]
    ry = rotvec[1]
    rz = rotvec[2]
    thetasq = rx * rx + ry * ry + rz * rz
    theta_reg = qd.sqrt(thetasq + EPS * EPS)
    cos_h = qd.cos(theta_reg / 2.0)
    sin_h = qd.sin(theta_reg / 2.0)
    sinc = sin_h / theta_reg
    return qd.Vector([cos_h, sinc * rx, sinc * ry, sinc * rz], dt=qd.f64)


@qd.func
def transform_by_quat(v, quat):
    q_w = quat[0]
    q_x = quat[1]
    q_y = quat[2]
    q_z = quat[3]
    v0 = v[0]
    v1 = v[1]
    v2 = v[2]
    q_xx = q_x * q_x
    q_xy = q_x * q_y
    q_xz = q_x * q_z
    q_wx = q_x * q_w
    q_yy = q_y * q_y
    q_yz = q_y * q_z
    q_wy = q_y * q_w
    q_zz = q_z * q_z
    q_wz = q_z * q_w
    q_ww = q_w * q_w
    return qd.Vector(
        [
            v0 * (q_xx + q_ww - q_yy - q_zz) + v1 * (2.0 * q_xy - 2.0 * q_wz) + v2 * (2.0 * q_xz + 2.0 * q_wy),
            v0 * (2.0 * q_xy + 2.0 * q_wz) + v1 * (q_yy + q_ww - q_xx - q_zz) + v2 * (2.0 * q_yz - 2.0 * q_wx),
            v0 * (2.0 * q_xz - 2.0 * q_wy) + v1 * (2.0 * q_yz + 2.0 * q_wx) + v2 * (q_zz + q_ww - q_xx - q_yy),
        ],
        dt=qd.f64,
    )


# ---------- one substep: rotate, then accumulate arm offset into world pos ----------


@qd.kernel
def step(
    quat_in: qd.template(),
    quat_out: qd.template(),
    pos_in: qd.template(),
    pos_out: qd.template(),
    arm_local: qd.template(),
    omega: qd.template(),
):
    """Single FK substep, mirroring Genesis's forward-kinematics inner loop.

    quat_out = quat_in * rotvec_to_quat(omega * dt)
    pos_out  = pos_in + transform_by_quat(arm_local, quat_in)
    """
    # Rotation update (quaternion right-multiplication by infinitesimal rotation)
    ang = qd.Vector([omega[0] * DT, omega[1] * DT, omega[2] * DT], dt=qd.f64)
    qrot = rotvec_to_quat(ang)
    q_new = quat_mul(quat_in, qrot)
    for j in qd.static(range(4)):
        quat_out[j] = q_new[j]

    # Position update: world-frame arm offset added to running pos
    arm_world = transform_by_quat(arm_local, quat_in)
    for j in qd.static(range(3)):
        pos_out[j] = pos_in[j] + arm_world[j]


# ---------- driver: roll N substeps forward, then reverse, then FD ----------


def run_chain(N: int, arm_local_np, omega_np):
    """Forward N steps + backward through full chain. Returns d_loss/d_omega."""
    # Per-step buffers (we keep them separate so backward chains properly)
    quats = [qd.field(dtype=qd.f64, shape=(4,), needs_grad=True) for _ in range(N + 1)]
    poss = [qd.field(dtype=qd.f64, shape=(3,), needs_grad=True) for _ in range(N + 1)]
    arm = qd.field(dtype=qd.f64, shape=(3,), needs_grad=True)
    omega = qd.field(dtype=qd.f64, shape=(3,), needs_grad=True)

    # Seed forward state
    quats[0].from_numpy(np.array([1.0, 0.0, 0.0, 0.0]))  # identity
    poss[0].from_numpy(np.zeros(3))
    arm.from_numpy(arm_local_np)
    omega.from_numpy(omega_np)
    for t in range(1, N + 1):
        quats[t].from_numpy(np.zeros(4))
        poss[t].from_numpy(np.zeros(3))

    # Forward
    for t in range(N):
        step(quats[t], quats[t + 1], poss[t], poss[t + 1], arm, omega)

    # Seed adjoint: loss = sum(pos[N])
    for t in range(N + 1):
        quats[t].grad.from_numpy(np.zeros(4))
        poss[t].grad.from_numpy(np.zeros(3))
    arm.grad.from_numpy(np.zeros(3))
    omega.grad.from_numpy(np.zeros(3))
    poss[N].grad.from_numpy(np.ones(3))

    # Backward (manual reverse traversal, mirroring how Genesis unrolls substeps)
    for t in reversed(range(N)):
        step.grad(quats[t], quats[t + 1], poss[t], poss[t + 1], arm, omega)

    return np.array([float(omega.grad[i]) for i in range(3)])


def forward_loss(N: int, arm_local_np, omega_np):
    """Forward-only loss (for finite-difference reference)."""
    quats = [qd.field(dtype=qd.f64, shape=(4,)) for _ in range(N + 1)]
    poss = [qd.field(dtype=qd.f64, shape=(3,)) for _ in range(N + 1)]
    arm = qd.field(dtype=qd.f64, shape=(3,))
    omega = qd.field(dtype=qd.f64, shape=(3,))

    quats[0].from_numpy(np.array([1.0, 0.0, 0.0, 0.0]))
    poss[0].from_numpy(np.zeros(3))
    arm.from_numpy(arm_local_np)
    omega.from_numpy(omega_np)
    for t in range(1, N + 1):
        quats[t].from_numpy(np.zeros(4))
        poss[t].from_numpy(np.zeros(3))

    for t in range(N):
        step(quats[t], quats[t + 1], poss[t], poss[t + 1], arm, omega)

    return float(poss[N][0]) + float(poss[N][1]) + float(poss[N][2])


def richardson_fd(N: int, arm_local_np, omega_np, h_coarse=1e-5):
    """O(h^4) Richardson finite difference of d_loss / d_omega[k]."""
    h_fine = h_coarse / 2.0
    g = np.zeros(3)
    for k in range(3):
        def L(h):
            op = omega_np.copy(); op[k] += h
            om = omega_np.copy(); om[k] -= h
            return forward_loss(N, arm_local_np, op) - forward_loss(N, arm_local_np, om)
        D_coarse = L(h_coarse) / (2 * h_coarse)
        D_fine = L(h_fine) / (2 * h_fine)
        g[k] = (16 * D_fine - D_coarse) / 15.0  # 4th-order Richardson
    return g


def sweep(arm_local_np, label: str):
    print(f"\n=== arm_local = {arm_local_np.tolist()}  ({label}) ===")
    omega_np = np.array([1.7, -0.9, 0.5])  # constant angular velocity (nontrivial axis)

    Ns = [1, 2, 4, 8, 16]
    print(f"{'N':>4}  {'max|ana|':>12}  {'max|FD|':>12}  {'max|diff|':>12}  {'max rel':>10}")
    print("-" * 60)
    for N in Ns:
        ana = run_chain(N, arm_local_np, omega_np)
        fd = richardson_fd(N, arm_local_np, omega_np)
        diff = ana - fd
        rel = np.where(np.abs(fd) > 1e-14, np.abs(diff) / (np.abs(fd) + 1e-30), 0.0)
        print(
            f"{N:>4}  "
            f"{np.abs(ana).max():>12.3e}  "
            f"{np.abs(fd).max():>12.3e}  "
            f"{np.abs(diff).max():>12.3e}  "
            f"{rel.max():>10.3e}"
        )


def main():
    print("Minimal N-step FK chain — reverse-mode gradient vs Richardson FD.")
    print(f"DT = {DT}, FP64.  Forward primitives copied from genesis/utils/geom.py.")
    print()
    print("Reporting max|ana|, max|FD|, max|ana - FD|, max relative diff")
    print("across the 3 components of d (sum pos[N]) / d omega.")

    sweep(np.array([0.0, 0.0, 0.0]), label="no transform_by_quat chain")
    sweep(np.array([0.2, 0.0, 0.0]), label="arm offset 0.2")
    sweep(np.array([1.0, 0.0, 0.0]), label="arm offset 1.0 (5x)")


if __name__ == "__main__":
    main()
