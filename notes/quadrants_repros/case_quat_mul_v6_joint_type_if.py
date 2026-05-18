"""Variant 6: add `if joint_type == FREE` conditional branch."""

import numpy as np
import quadrants as qd


qd.init(arch=qd.cpu)


DT = 0.01
EPS = 1e-8


@qd.func
def my_rotvec_to_quat(rotvec, eps):
    rx = rotvec[0]
    ry = rotvec[1]
    rz = rotvec[2]
    thetasq = rx*rx + ry*ry + rz*rz
    theta_reg = qd.sqrt(thetasq + eps*eps)
    cos_h = qd.cos(theta_reg / 2.0)
    sin_h = qd.sin(theta_reg / 2.0)
    sinc = sin_h / theta_reg
    return qd.Vector([cos_h, sinc*rx, sinc*ry, sinc*rz], dt=qd.f64)


@qd.func
def my_quat_mul(a, b):
    return qd.Vector(
        [
            a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
            a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
            a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
            a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
        ],
        dt=qd.f64,
    )


@qd.func
def my_transform_quat_by_quat(v, u):
    return my_quat_mul(u, v)


@qd.func
def step1_vel_next(vel, acc, vel_next):
    for i_d, i_b in qd.ndrange(3, 1):
        vel_next[i_d, i_b] = vel[i_d, i_b] + DT * acc[i_d, i_b]


@qd.func
def step2_rotation(qpos, vel_next, qpos_next, q_starts, joint_types):
    for i_l, i_b in qd.ndrange(1, 1):
        joint_type = joint_types[i_l]
        if joint_type == 0:  # mock JOINT_TYPE.FREE
            q_start = q_starts[i_l]
            rot0 = qd.Vector(
                [
                    qpos[q_start + 0, i_b],
                    qpos[q_start + 1, i_b],
                    qpos[q_start + 2, i_b],
                    qpos[q_start + 3, i_b],
                ],
                dt=qd.f64,
            )
            ang = qd.Vector([vel_next[0, i_b], vel_next[1, i_b], vel_next[2, i_b]], dt=qd.f64) * DT
            qrot_v = my_rotvec_to_quat(ang, EPS)
            rot = my_transform_quat_by_quat(qrot_v, rot0)
            for j in qd.static(range(4)):
                qpos_next[q_start + j, i_b] = rot[j]


@qd.kernel
def kernel_v6(
    vel: qd.template(),
    acc: qd.template(),
    qpos: qd.template(),
    vel_next: qd.template(),
    qpos_next: qd.template(),
    q_starts: qd.template(),
    joint_types: qd.template(),
):
    step1_vel_next(vel, acc, vel_next)
    step2_rotation(qpos, vel_next, qpos_next, q_starts, joint_types)


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
    Q_START = 3
    vel = qd.field(dtype=qd.f64, shape=(3, 1), needs_grad=True)
    acc = qd.field(dtype=qd.f64, shape=(3, 1), needs_grad=True)
    qpos = qd.field(dtype=qd.f64, shape=(7, 1), needs_grad=True)
    vel_next = qd.field(dtype=qd.f64, shape=(3, 1), needs_grad=True)
    qpos_next = qd.field(dtype=qd.f64, shape=(7, 1), needs_grad=True)
    q_starts = qd.field(dtype=qd.i32, shape=(1,))
    joint_types = qd.field(dtype=qd.i32, shape=(1,))

    q_pre_full_np = np.zeros((7, 1))
    q_pre_full_np[Q_START:Q_START + 4, 0] = np.array([1.0, 0.0, 0.0, 0.0])
    vel_np = np.zeros((3, 1))
    acc_np = np.array([[5.374], [0.6111], [-2.591]])
    vel_next_np = vel_np + DT * acc_np
    ang_np = vel_next_np[:, 0] * DT
    qrot_np = rotvec_to_quat_np(ang_np, EPS)
    q_next_grad_seed_full = np.zeros((7, 1))
    q_next_grad_seed_full[Q_START:Q_START + 4, 0] = np.array([1.5999e-1, 4.2922e-5, -8.6127e-5, -8.8639e-6])

    def seed():
        vel.from_numpy(vel_np)
        acc.from_numpy(acc_np)
        qpos.from_numpy(q_pre_full_np)
        vel_next.from_numpy(np.zeros((3, 1)))
        qpos_next.from_numpy(np.zeros((7, 1)))
        qpos_next.grad.from_numpy(q_next_grad_seed_full)
        qpos.grad.from_numpy(np.zeros((7, 1)))
        vel.grad.from_numpy(np.zeros((3, 1)))
        acc.grad.from_numpy(np.zeros((3, 1)))
        vel_next.grad.from_numpy(np.zeros((3, 1)))
        q_starts.from_numpy(np.array([Q_START], dtype=np.int32))
        joint_types.from_numpy(np.array([0], dtype=np.int32))  # FREE

    seed()
    kernel_v6(vel, acc, qpos, vel_next, qpos_next, q_starts, joint_types)
    kernel_v6.grad(vel, acc, qpos, vel_next, qpos_next, q_starts, joint_types)

    q_pre_grad_ana_full = np.array([float(qpos.grad[i, 0]) for i in range(7)])
    q_pre_grad_ana = q_pre_grad_ana_full[Q_START:Q_START + 4]

    J_a = quat_mul_jac_a(q_pre_full_np[Q_START:Q_START + 4, 0], qrot_np)
    q_pre_grad_np = J_a.T @ q_next_grad_seed_full[Q_START:Q_START + 4, 0]

    print(f"q_pre.grad (kernel) = {q_pre_grad_ana}")
    print(f"q_pre.grad (numpy)  = {q_pre_grad_np}")
    diff = q_pre_grad_ana - q_pre_grad_np
    print(f"max|diff| = {float(np.abs(diff).max()):.3e}")
    if float(np.abs(diff).max()) > 1e-10:
        print("===> SILENT DROP REPRODUCED in v6 (joint_type if) <===")
    else:
        print("===> chain matches; v6 not the trigger <===")


if __name__ == "__main__":
    main()
