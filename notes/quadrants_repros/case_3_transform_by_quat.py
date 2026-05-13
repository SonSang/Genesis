"""Case 3 — Standalone repro of `qd_transform_by_quat` backward.

Isolates the function copied verbatim from
`genesis/utils/geom.py::qd_transform_by_quat` (line 294), then compares
the analytical reverse-mode gradient against finite-difference and the
hand-computed Jacobian.

Forward (raw quaternion product, no normalize):
    out[0] = v0·(qw² + qx² - qy² - qz²) + v1·(2qxy - 2qwz) + v2·(2qxz + 2qwy)
    out[1] = v0·(2qxy + 2qwz) + v1·(qw² - qx² + qy² - qz²) + v2·(2qyz - 2qwx)
    out[2] = v0·(2qxz - 2qwy) + v1·(2qyz + 2qwx) + v2·(qw² - qx² - qy² + qz²)

For v = [0.2, 0, 0] and seed `out.grad = [1, 1, 1]`:
    d(sum out)/d(qy) = 0.2·(-2qy)  +  0.4qx  +  (-0.4qw)
                     = -0.4qy + 0.4qx - 0.4qw

At quat = [0.995, 0, 0.1, 0]:
    = -0.4·0.1 + 0 - 0.4·0.995
    = -0.04 - 0.398
    = -0.438

If `qpos.grad[2]` returned by Quadrants reverse-mode matches -0.438, the
function is fine.

If it returns only -0.398 (i.e., the second term — qw-related), the
qy-from-output[0] chain (the `q_yy = q_y * q_y` term) is silently
dropped. That would be the new silent-drop pattern at the heart of the
J4/J5 multi-step xfail.
"""

import numpy as np

import quadrants as qd


qd.init(arch=qd.cpu)


@qd.kernel
def transform_by_quat(v: qd.template(), quat: qd.template(), out: qd.template()):
    """Verbatim copy of `genesis/utils/geom.py::qd_transform_by_quat`."""
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

    out[0] = v0 * (q_xx + q_ww - q_yy - q_zz) + v1 * (2.0 * q_xy - 2.0 * q_wz) + v2 * (2.0 * q_xz + 2.0 * q_wy)
    out[1] = v0 * (2.0 * q_xy + 2.0 * q_wz) + v1 * (q_yy + q_ww - q_xx - q_zz) + v2 * (2.0 * q_yz - 2.0 * q_wx)
    out[2] = v0 * (2.0 * q_xz - 2.0 * q_wy) + v1 * (2.0 * q_yz + 2.0 * q_wx) + v2 * (q_zz + q_ww - q_xx - q_yy)


def main():
    v = qd.field(dtype=qd.f64, shape=(3,), needs_grad=False)
    quat = qd.field(dtype=qd.f64, shape=(4,), needs_grad=True)
    out = qd.field(dtype=qd.f64, shape=(3,), needs_grad=True)

    # Baseline near identity but with qy=0.1 (non-tiny so the qy² chain
    # is clearly distinguishable from the qw·qy chain).
    quat_np = np.array([np.sqrt(1 - 0.01), 0.0, 0.1, 0.0])  # unit quat, qy=0.1
    v_np = np.array([0.2, 0.0, 0.0])

    def seed():
        v.from_numpy(v_np)
        quat.from_numpy(quat_np)
        out.from_numpy(np.zeros(3))
        out.grad.from_numpy(np.ones(3))  # seed sum
        quat.grad.from_numpy(np.zeros(4))

    # Analytical (Quadrants AD)
    seed()
    transform_by_quat(v, quat, out)
    transform_by_quat.grad(v, quat, out)
    ana = np.array([float(quat.grad[i]) for i in range(4)])
    out_np = np.array([float(out[i]) for i in range(3)])

    print("baseline quat =", quat_np)
    print("baseline v    =", v_np)
    print("forward out   =", out_np)
    print()

    # Finite-difference reference: d(sum out) / d(quat[k])
    eps = 1e-6
    fd = np.zeros(4)
    for k in range(4):
        seed()
        quat_p = quat_np.copy()
        quat_p[k] += eps
        quat.from_numpy(quat_p)
        transform_by_quat(v, quat, out)
        sp = float(out[0]) + float(out[1]) + float(out[2])

        seed()
        quat_m = quat_np.copy()
        quat_m[k] -= eps
        quat.from_numpy(quat_m)
        transform_by_quat(v, quat, out)
        sm = float(out[0]) + float(out[1]) + float(out[2])

        fd[k] = (sp - sm) / (2 * eps)

    # Hand-computed Jacobian for v = [0.2, 0, 0], sum-output seed:
    #   sum_out = v0·(qw² + qx² - qy² - qz²) + (other v components zero)
    #           + v0·(2qxy + 2qwz)
    #           + v0·(2qxz - 2qwy)
    # d/dqw = v0·2qw + v0·2qz + v0·(-2qy) = 0.4·(qw + qz - qy)
    # d/dqx = v0·2qx + v0·2qy + v0·2qz   = 0.4·(qx + qy + qz)
    # d/dqy = v0·(-2qy) + v0·2qx + v0·(-2qw) = 0.4·(-qy + qx - qw)
    # d/dqz = v0·(-2qz) + v0·2qw + v0·2qx = 0.4·(-qz + qw + qx)
    qw, qx, qy, qz = quat_np
    hand = 0.4 * np.array([qw + qz - qy, qx + qy + qz, -qy + qx - qw, -qz + qw + qx])

    print(f"{'k':>3}  {'analytical':>14}  {'finite-diff':>14}  {'hand-derived':>14}  {'ana==FD?':>10}")
    print("-" * 75)
    for k in range(4):
        ok = np.isclose(ana[k], fd[k], rtol=1e-4)
        print(f"{k:>3}  {ana[k]:>14.6e}  {fd[k]:>14.6e}  {hand[k]:>14.6e}  {'OK' if ok else 'MISMATCH':>10}")


if __name__ == "__main__":
    main()
