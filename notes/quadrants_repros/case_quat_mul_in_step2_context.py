"""Minimal repro of the silent drop seen in step_2.grad's quat integrator
reverse on J4 N=2.

Forward — mirrors the rotation block of `func_integrate`:
    rot0   = qd.Vector([qpos[0..3]])
    qrot_v = qd.Vector([qrot[0..3]])
    rot    = quat_mul(rot0, qrot_v)        # via qd_transform_quat_by_quat(qrot_v, rot0)
    for j: qpos_next[j] = rot[j]            # element-wise write

Seed `qpos_next.grad` with magnitudes matching the J4 N=2 case and
compute `qpos.grad` via Quadrants AD. Compare to numpy chain:
    qpos.grad = J_a(rot0, qrot_v)^T @ qpos_next.grad
where J_a[i, j] = ∂(quat_mul(a, b))[i] / ∂a[j] (depends on b only).

If the Quadrants result equals the numpy chain → the silent drop is
NOT triggered in isolation, and the bug lives in the larger
`func_integrate` context (other for-loops, ndranges, etc.).
If it differs the same way as the J4 N=2 trace
(only first term of J_a^T[1] · q_next.grad survives) → reproduced.
"""

import numpy as np
import quadrants as qd


qd.init(arch=qd.cpu)


@qd.func
def my_quat_mul(a, b):
    """Verbatim copy of qd_quat_mul (Hamilton convention)."""
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
    """Verbatim copy of qd_transform_quat_by_quat — equiv to quat_mul(u, v)."""
    return my_quat_mul(u, v)


@qd.kernel
def kernel_step2_rot_block(
    qpos: qd.template(),
    qrot: qd.template(),
    qpos_next: qd.template(),
):
    """Mirror of the rotation block inside func_integrate."""
    rot0 = qd.Vector([qpos[0], qpos[1], qpos[2], qpos[3]], dt=qd.f64)
    qrot_v = qd.Vector([qrot[0], qrot[1], qrot[2], qrot[3]], dt=qd.f64)
    rot = my_transform_quat_by_quat(qrot_v, rot0)
    for j in qd.static(range(4)):
        qpos_next[j] = rot[j]


def quat_mul_jac_a(a, b):
    """∂(quat_mul(a, b))/∂a — 4x4 (function of b only)."""
    bw, bx, by, bz = b
    return np.array([
        [bw, -bx, -by, -bz],
        [bx,  bw,  bz, -by],
        [by, -bz,  bw,  bx],
        [bz,  by, -bx,  bw],
    ])


def main():
    qpos = qd.field(dtype=qd.f64, shape=(4,), needs_grad=True)
    qrot = qd.field(dtype=qd.f64, shape=(4,), needs_grad=True)
    qpos_next = qd.field(dtype=qd.f64, shape=(4,), needs_grad=True)

    q_pre_np = np.array([1.0, 0.0, 0.0, 0.0])
    # qrot near identity, mimicking J4 N=2 step 0
    qrot_np = np.array([0.9999999955, 2.687e-4, 3.055e-5, -1.296e-4])
    q_next_grad_seed = np.array([1.5999e-1, 4.2922e-5, -8.6127e-5, -8.8639e-6])

    def seed():
        qpos.from_numpy(q_pre_np)
        qrot.from_numpy(qrot_np)
        qpos_next.from_numpy(np.zeros(4))
        qpos_next.grad.from_numpy(q_next_grad_seed)
        qpos.grad.from_numpy(np.zeros(4))
        qrot.grad.from_numpy(np.zeros(4))

    # Analytical (Quadrants AD)
    seed()
    kernel_step2_rot_block(qpos, qrot, qpos_next)
    kernel_step2_rot_block.grad(qpos, qrot, qpos_next)

    q_pre_grad_ana = np.array([float(qpos.grad[i]) for i in range(4)])
    qrot_grad_ana = np.array([float(qrot.grad[i]) for i in range(4)])
    q_next_out = np.array([float(qpos_next[i]) for i in range(4)])

    # Numpy chain
    J_a = quat_mul_jac_a(q_pre_np, qrot_np)
    q_pre_grad_np = J_a.T @ q_next_grad_seed
    q_next_np_ref = np.array([
        q_pre_np[0]*qrot_np[0] - q_pre_np[1]*qrot_np[1] - q_pre_np[2]*qrot_np[2] - q_pre_np[3]*qrot_np[3],
        q_pre_np[0]*qrot_np[1] + q_pre_np[1]*qrot_np[0] + q_pre_np[2]*qrot_np[3] - q_pre_np[3]*qrot_np[2],
        q_pre_np[0]*qrot_np[2] - q_pre_np[1]*qrot_np[3] + q_pre_np[2]*qrot_np[0] + q_pre_np[3]*qrot_np[1],
        q_pre_np[0]*qrot_np[3] + q_pre_np[1]*qrot_np[2] - q_pre_np[2]*qrot_np[1] + q_pre_np[3]*qrot_np[0],
    ])

    print(f"q_pre   = {q_pre_np}")
    print(f"qrot    = {qrot_np}")
    print(f"q_next forward kernel = {q_next_out}")
    print(f"q_next forward numpy  = {q_next_np_ref}")
    print(f"forward match: max|diff| = {float(np.abs(q_next_out - q_next_np_ref).max()):.3e}")
    print()
    print(f"q_next.grad seed = {q_next_grad_seed}")
    print()
    print("q_pre.grad row 1 (= qx contribution) breakdown:")
    print(f"  J_a^T[1] = {J_a.T[1]}")
    for k in range(4):
        c = J_a.T[1, k] * q_next_grad_seed[k]
        print(f"  contrib from q_next.grad[{k}]={q_next_grad_seed[k]:>11.4e}: {J_a.T[1, k]:>11.4e} * {q_next_grad_seed[k]:>11.4e} = {c:>11.4e}")
    print(f"  numpy sum            = {q_pre_grad_np[1]:.4e}")
    print(f"  Quadrants AD result  = {q_pre_grad_ana[1]:.4e}")
    print()
    print(f"q_pre.grad (kernel) = {q_pre_grad_ana}")
    print(f"q_pre.grad (numpy)  = {q_pre_grad_np}")
    diff = q_pre_grad_ana - q_pre_grad_np
    print(f"  max|diff| = {float(np.abs(diff).max()):.3e}")
    if float(np.abs(diff).max()) > 1e-10:
        print("  ===> SILENT DROP REPRODUCED <===")
    else:
        print("  ===> chain rule matches; silent drop not triggered in this isolated repro <===")


if __name__ == "__main__":
    main()
