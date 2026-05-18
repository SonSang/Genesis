"""Variant 3 of step_2 silent drop repro: cross-ndrange chain.

Forward:
  ndrange 1 (n_dofs):     vel_next[i] = vel[i] + dt * acc[i]
  ndrange 2 (n_links):    qrot_v = (1, vel_next[0]·dt/2, vel_next[1]·dt/2, vel_next[2]·dt/2)  # simplified
                          q_next = quat_mul(q_pre, qrot_v)

This tests whether having two separate ndranges with the second reading
the first's output triggers the silent drop. (rotvec_to_quat is simplified
to a linear approximation so the chain rule stays trivial.)
"""

import numpy as np
import quadrants as qd


qd.init(arch=qd.cpu)


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


DT = 0.01


@qd.func
def step1_vel_next(vel, acc, vel_next):
    for i_d in qd.ndrange(3):
        vel_next[i_d] = vel[i_d] + DT * acc[i_d]


@qd.func
def step2_rotation(qpos, vel_next, qpos_next):
    for i_l in qd.ndrange(1):
        rot0 = qd.Vector([qpos[0], qpos[1], qpos[2], qpos[3]], dt=qd.f64)
        qrot_v = qd.Vector(
            [1.0, vel_next[0] * (DT * 0.5), vel_next[1] * (DT * 0.5), vel_next[2] * (DT * 0.5)],
            dt=qd.f64,
        )
        rot = my_transform_quat_by_quat(qrot_v, rot0)
        for j in qd.static(range(4)):
            qpos_next[j] = rot[j]


@qd.kernel
def kernel_v3(
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


def main():
    vel = qd.field(dtype=qd.f64, shape=(3,), needs_grad=True)
    acc = qd.field(dtype=qd.f64, shape=(3,), needs_grad=True)
    qpos = qd.field(dtype=qd.f64, shape=(4,), needs_grad=True)
    vel_next = qd.field(dtype=qd.f64, shape=(3,), needs_grad=True)
    qpos_next = qd.field(dtype=qd.f64, shape=(4,), needs_grad=True)

    dt = 0.01
    q_pre_np = np.array([1.0, 0.0, 0.0, 0.0])
    vel_np = np.zeros(3)
    acc_np = np.array([5.374, 0.6111, -2.591])  # so vel_next = dt*acc ≈ [5.4e-2, 6.1e-3, -2.6e-2]
    vel_next_np = vel_np + dt * acc_np
    qrot_np = np.array([1.0, vel_next_np[0] * dt / 2, vel_next_np[1] * dt / 2, vel_next_np[2] * dt / 2])
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
    kernel_v3(vel, acc, qpos, vel_next, qpos_next)
    kernel_v3.grad(vel, acc, qpos, vel_next, qpos_next)

    q_pre_grad_ana = np.array([float(qpos.grad[i]) for i in range(4)])

    # Numpy chain
    J_a = quat_mul_jac_a(q_pre_np, qrot_np)
    q_pre_grad_np = J_a.T @ q_next_grad_seed

    print(f"qrot (simplified linear) = {qrot_np}")
    print(f"q_pre.grad (kernel) = {q_pre_grad_ana}")
    print(f"q_pre.grad (numpy)  = {q_pre_grad_np}")
    diff = q_pre_grad_ana - q_pre_grad_np
    print(f"max|diff| = {float(np.abs(diff).max()):.3e}")
    if float(np.abs(diff).max()) > 1e-10:
        print("===> SILENT DROP REPRODUCED in v3 (cross-ndrange) <===")
    else:
        print("===> chain matches; v3 not the trigger <===")


if __name__ == "__main__":
    main()
