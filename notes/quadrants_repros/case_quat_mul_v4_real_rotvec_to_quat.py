"""Variant 4: add real qd_rotvec_to_quat (cos/sin/sqrt) into the chain."""

import numpy as np
import quadrants as qd


qd.init(arch=qd.cpu)


DT = 0.01
EPS = 1e-8


@qd.func
def my_rotvec_to_quat(rotvec, eps):
    """Verbatim copy of geom.qd_rotvec_to_quat."""
    rx = rotvec[0]
    ry = rotvec[1]
    rz = rotvec[2]
    thetasq = rx * rx + ry * ry + rz * rz
    theta_reg = qd.sqrt(thetasq + eps * eps)
    cos_h = qd.cos(theta_reg / 2.0)
    sin_h = qd.sin(theta_reg / 2.0)
    sinc = sin_h / theta_reg
    return qd.Vector([cos_h, sinc * rx, sinc * ry, sinc * rz], dt=qd.f64)


@qd.func
def my_quat_mul(a, b):
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
def my_transform_quat_by_quat(v, u):
    return my_quat_mul(u, v)


@qd.func
def step1_vel_next(vel, acc, vel_next):
    for i_d in qd.ndrange(3):
        vel_next[i_d] = vel[i_d] + DT * acc[i_d]


@qd.func
def step2_rotation(qpos, vel_next, qpos_next):
    for i_l in qd.ndrange(1):
        rot0 = qd.Vector([qpos[0], qpos[1], qpos[2], qpos[3]], dt=qd.f64)
        ang = qd.Vector([vel_next[0], vel_next[1], vel_next[2]], dt=qd.f64) * DT
        qrot_v = my_rotvec_to_quat(ang, EPS)
        rot = my_transform_quat_by_quat(qrot_v, rot0)
        for j in qd.static(range(4)):
            qpos_next[j] = rot[j]


@qd.kernel
def kernel_v4(
    vel: qd.template(),
    acc: qd.template(),
    qpos: qd.template(),
    vel_next: qd.template(),
    qpos_next: qd.template(),
):
    step1_vel_next(vel, acc, vel_next)
    step2_rotation(qpos, vel_next, qpos_next)


def quat_mul_jac_a(a, b):
    bw, bx, by, bz = b
    return np.array([
        [bw, -bx, -by, -bz],
        [bx,  bw,  bz, -by],
        [by, -bz,  bw,  bx],
        [bz,  by, -bx,  bw],
    ])


def rotvec_to_quat_np(ang, eps):
    rx, ry, rz = ang
    thetasq = rx*rx + ry*ry + rz*rz
    theta_reg = np.sqrt(thetasq + eps*eps)
    return np.array([np.cos(theta_reg/2), np.sin(theta_reg/2)*rx/theta_reg,
                     np.sin(theta_reg/2)*ry/theta_reg, np.sin(theta_reg/2)*rz/theta_reg])


def main():
    vel = qd.field(dtype=qd.f64, shape=(3,), needs_grad=True)
    acc = qd.field(dtype=qd.f64, shape=(3,), needs_grad=True)
    qpos = qd.field(dtype=qd.f64, shape=(4,), needs_grad=True)
    vel_next = qd.field(dtype=qd.f64, shape=(3,), needs_grad=True)
    qpos_next = qd.field(dtype=qd.f64, shape=(4,), needs_grad=True)

    q_pre_np = np.array([1.0, 0.0, 0.0, 0.0])
    vel_np = np.zeros(3)
    acc_np = np.array([5.374, 0.6111, -2.591])
    vel_next_np = vel_np + DT * acc_np
    ang_np = vel_next_np * DT
    qrot_np = rotvec_to_quat_np(ang_np, EPS)
    q_next_grad_seed = np.array([1.5999e-1, 4.2922e-5, -8.6127e-5, -8.8639e-6])

    def seed():
        vel.from_numpy(vel_np)
        acc.from_numpy(acc_np)
        qpos.from_numpy(q_pre_np)
        vel_next.from_numpy(np.zeros(3))
        qpos_next.from_numpy(np.zeros(4))
        qpos_next.grad.from_numpy(q_next_grad_seed)
        qpos.grad.from_numpy(np.zeros(4))
        vel.grad.from_numpy(np.zeros(3))
        acc.grad.from_numpy(np.zeros(3))
        vel_next.grad.from_numpy(np.zeros(3))

    seed()
    kernel_v4(vel, acc, qpos, vel_next, qpos_next)
    kernel_v4.grad(vel, acc, qpos, vel_next, qpos_next)

    q_pre_grad_ana = np.array([float(qpos.grad[i]) for i in range(4)])
    J_a = quat_mul_jac_a(q_pre_np, qrot_np)
    q_pre_grad_np = J_a.T @ q_next_grad_seed

    print(f"qrot = {qrot_np}")
    print(f"q_pre.grad (kernel) = {q_pre_grad_ana}")
    print(f"q_pre.grad (numpy)  = {q_pre_grad_np}")
    diff = q_pre_grad_ana - q_pre_grad_np
    print(f"max|diff| = {float(np.abs(diff).max()):.3e}")
    if float(np.abs(diff).max()) > 1e-10:
        print("===> SILENT DROP REPRODUCED in v4 (real rotvec_to_quat) <===")
    else:
        print("===> chain matches; v4 not the trigger <===")


if __name__ == "__main__":
    main()
