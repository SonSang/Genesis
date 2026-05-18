"""Variant 2 of step_2 silent drop repro.

Adds two ingredients on top of `case_quat_mul_in_step2_context.py`:
  1. outer `qd.ndrange(n_links, B)` loop (mirroring func_integrate)
  2. dynamic indexing `qpos[q_start + rot_offset + j]` with q_start a Python int
     (still a compile-time constant; v3 will promote it to an array read).

If still no silent drop, v3 will use a *real* dynamic q_start loaded from
an entities_info-like array.
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


@qd.kernel
def kernel_v2(
    qpos: qd.template(),       # shape (n_qs=7,) — pos(3) + quat(4)
    qrot: qd.template(),       # shape (4,)
    qpos_next: qd.template(),  # shape (n_qs=7,)
):
    """Outer ndrange + dynamic-but-static-constant q_start indexing."""
    for i_l, i_b in qd.ndrange(1, 1):
        q_start = qd.static(3)  # rot offset (static constant)
        rot0 = qd.Vector(
            [
                qpos[q_start + 0],
                qpos[q_start + 1],
                qpos[q_start + 2],
                qpos[q_start + 3],
            ],
            dt=qd.f64,
        )
        qrot_v = qd.Vector([qrot[0], qrot[1], qrot[2], qrot[3]], dt=qd.f64)
        rot = my_transform_quat_by_quat(qrot_v, rot0)
        for j in qd.static(range(4)):
            qpos_next[q_start + j] = rot[j]


def quat_mul_jac_a(a, b):
    bw, bx, by, bz = b
    return np.array([
        [bw, -bx, -by, -bz],
        [bx,  bw,  bz, -by],
        [by, -bz,  bw,  bx],
        [bz,  by, -bx,  bw],
    ])


def main():
    qpos = qd.field(dtype=qd.f64, shape=(7,), needs_grad=True)
    qrot = qd.field(dtype=qd.f64, shape=(4,), needs_grad=True)
    qpos_next = qd.field(dtype=qd.f64, shape=(7,), needs_grad=True)

    q_pre_np = np.zeros(7)
    q_pre_np[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    qrot_np = np.array([0.9999999955, 2.687e-4, 3.055e-5, -1.296e-4])
    q_next_grad_seed = np.zeros(7)
    q_next_grad_seed[3:7] = np.array([1.5999e-1, 4.2922e-5, -8.6127e-5, -8.8639e-6])

    def seed():
        qpos.from_numpy(q_pre_np)
        qrot.from_numpy(qrot_np)
        qpos_next.from_numpy(np.zeros(7))
        qpos_next.grad.from_numpy(q_next_grad_seed)
        qpos.grad.from_numpy(np.zeros(7))
        qrot.grad.from_numpy(np.zeros(4))

    seed()
    kernel_v2(qpos, qrot, qpos_next)
    kernel_v2.grad(qpos, qrot, qpos_next)

    q_pre_grad_ana = np.array([float(qpos.grad[i]) for i in range(7)])

    J_a = quat_mul_jac_a(q_pre_np[3:7], qrot_np)
    q_pre_grad_np_rot = J_a.T @ q_next_grad_seed[3:7]
    q_pre_grad_np = np.zeros(7)
    q_pre_grad_np[3:7] = q_pre_grad_np_rot

    print("q_pre.grad rotation part (kernel) =", q_pre_grad_ana[3:7])
    print("q_pre.grad rotation part (numpy)  =", q_pre_grad_np[3:7])
    diff = q_pre_grad_ana - q_pre_grad_np
    print(f"max|diff| (rotation part) = {float(np.abs(diff[3:7]).max()):.3e}")
    if float(np.abs(diff[3:7]).max()) > 1e-10:
        print("===> SILENT DROP REPRODUCED <===")
    else:
        print("===> chain matches; no silent drop in v2 <===")


if __name__ == "__main__":
    main()
