"""Case 4 — Cross-index same-buffer read/write in a single launch silently
drops the self-product backward chain.

Pattern (mimics Genesis's `kernel_update_cartesian_space_one_link` arm-link
launch, which inside a single kernel call:
  - reads  `links_state.quat[parent_idx]`   (chassis quat, index 0)
  - writes `links_state.quat[i_l]`          (arm quat, index 1)
  - reads from the *same field* through `qd_transform_by_quat`, which
    contains a self-product `q_yy = q_y * q_y` then uses it in
    `out[0] = v0 * (q_xx + q_ww - q_yy - q_zz) + ...`):

    @qd.kernel
    def step(q, v, out):
        # cross-index write — mirrors Genesis arm-link quat cache write
        q[1, 0] = q[0, 0]
        q[1, 1] = q[0, 1]
        q[1, 2] = q[0, 2]
        q[1, 3] = q[0, 3]

        # transform_by_quat body, inlined verbatim from geom.py
        qw = q[0, 0]; qx = q[0, 1]; qy = q[0, 2]; qz = q[0, 3]
        v0 = v[0]; v1 = v[1]; v2 = v[2]
        q_xx = qx * qx
        q_yy = qy * qy
        ...
        out[0] = v0 * (q_xx + q_ww - q_yy - q_zz) + ...

Seed: out.grad = [1, 1, 1]
Expected (same as case 3 standalone, since transform body is identical):
    q[0].grad[2] (qy) = -0.438   at quat = [0.995, 0, 0.1, 0]

If silent drop: q[0].grad[2] only has -0.398 (the qw·qy chain), with the
qy² → qy chain missing.

If standalone (case 3, no cross-index write) gives the full -0.438 and
this case gives only -0.398, the *cross-index write* within the same
launch is the trigger.

If both give -0.438, the trigger is something else (Genesis-internal cache
slot machinery, joint update, etc.). Time to look elsewhere.
"""

import numpy as np

import quadrants as qd


qd.init(arch=qd.cpu)


@qd.kernel
def step(q: qd.template(), v: qd.template(), out: qd.template()):
    # Cross-index write to same buffer — mirrors Genesis writing
    # links_state.quat[arm_link_idx] inside the same launch that reads
    # links_state.quat[chassis_idx].
    q[1, 0] = q[0, 0]
    q[1, 1] = q[0, 1]
    q[1, 2] = q[0, 2]
    q[1, 3] = q[0, 3]

    # qd_transform_by_quat body, verbatim from geom.py:294, with q[0] as
    # the source quat (read from same buffer as the write above).
    q_w = q[0, 0]
    q_x = q[0, 1]
    q_y = q[0, 2]
    q_z = q[0, 3]
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
    # q has 2 indices, same as Genesis's [chassis, arm] in a 2-link entity.
    q = qd.field(dtype=qd.f64, shape=(2, 4), needs_grad=True)
    v = qd.field(dtype=qd.f64, shape=(3,), needs_grad=False)
    out = qd.field(dtype=qd.f64, shape=(3,), needs_grad=True)

    quat_np = np.array([np.sqrt(1 - 0.01), 0.0, 0.1, 0.0])  # same baseline as case 3
    v_np = np.array([0.2, 0.0, 0.0])

    def seed():
        # q[0] = quat_np, q[1] = anything (will be overwritten)
        q_init = np.zeros((2, 4))
        q_init[0] = quat_np
        q.from_numpy(q_init)
        v.from_numpy(v_np)
        out.from_numpy(np.zeros(3))
        out.grad.from_numpy(np.ones(3))
        q.grad.from_numpy(np.zeros((2, 4)))

    seed()
    step(q, v, out)
    step.grad(q, v, out)
    ana = np.array([float(q.grad[0, i]) for i in range(4)])

    out_np = np.array([float(out[i]) for i in range(3)])

    # Hand-derived expected (identical to case 3, computed at quat=[0.995, 0, 0.1, 0]):
    qw, qx, qy, qz = quat_np
    hand = 0.4 * np.array([qw + qz - qy, qx + qy + qz, -qy + qx - qw, -qz + qw + qx])

    print("baseline quat (q[0]) =", quat_np)
    print("forward out          =", out_np)
    print()
    print(f"{'k':>3}  {'analytical':>14}  {'hand-derived':>14}  {'diff':>14}  {'match?':>10}")
    print("-" * 70)
    for k in range(4):
        diff = ana[k] - hand[k]
        ok = abs(diff) < 1e-6
        print(f"{k:>3}  {ana[k]:>14.6e}  {hand[k]:>14.6e}  {diff:>14.3e}  {'OK' if ok else 'MISMATCH':>10}")


if __name__ == "__main__":
    main()
