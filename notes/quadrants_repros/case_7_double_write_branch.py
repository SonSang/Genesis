"""Case 7 — Double-write to same buffer index + dynamic if-branch around
the transform.

Mirrors Genesis `func_forward_kinematics_entity_one_link` (line 695-704):
    pos = W(pos_bw, I_l0, links_info.pos[I_l], BW)       # 1st write
    if parent_idx != -1:                                  # dynamic branch
        ...
        pos_ = parent_pos + transform_by_quat(...)
        pos = W(pos_bw, I_l0, pos_, BW)                  # 2nd write (overwrite)

Hypothesis: the double-write to the same buffer index, possibly inside a
dynamic branch, causes Quadrants AD reverse to drop part of the chain
through `transform_by_quat`'s self-product (q_yy = q_y * q_y) term.

Forward seed: out.grad = [1, 1, 1] (sum-of-pos[1]).
Expected (same as case 3/4/5/6): q[0].grad[2] = -0.438.
"""

import numpy as np
import quadrants as qd


qd.init(arch=qd.cpu)


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


@qd.kernel
def step(
    q: qd.template(),
    pos: qd.template(),
    pos_bw: qd.template(),
    v: qd.template(),
    out: qd.template(),
    parent_idx_minus1: qd.template(),
):
    # 1st write — initial pos[1] = arm_local (mirrors Genesis line 695 W(pos_bw, I_l0, links_info.pos, BW))
    pos_bw[1, 0, 0] = v[0]
    pos_bw[1, 0, 1] = v[1]
    pos_bw[1, 0, 2] = v[2]

    # Dynamic branch: parent_idx != -1
    if parent_idx_minus1[0] != -1:
        parent_pos = qd.Vector([pos[0, 0], pos[0, 1], pos[0, 2]], dt=qd.f64)
        parent_quat = qd.Vector([q[0, 0], q[0, 1], q[0, 2], q[0, 3]], dt=qd.f64)
        arm_local = qd.Vector([v[0], v[1], v[2]], dt=qd.f64)

        pos_new = parent_pos + transform_by_quat(arm_local, parent_quat)

        # 2nd write — overwrite
        pos_bw[1, 0, 0] = pos_new[0]
        pos_bw[1, 0, 1] = pos_new[1]
        pos_bw[1, 0, 2] = pos_new[2]

    # final read
    pos[1, 0] = pos_bw[1, 0, 0]
    pos[1, 1] = pos_bw[1, 0, 1]
    pos[1, 2] = pos_bw[1, 0, 2]

    out[0] = pos[1, 0]
    out[1] = pos[1, 1]
    out[2] = pos[1, 2]


def main():
    q = qd.field(dtype=qd.f64, shape=(2, 4), needs_grad=True)
    pos = qd.field(dtype=qd.f64, shape=(2, 3), needs_grad=True)
    pos_bw = qd.field(dtype=qd.f64, shape=(2, 1, 3), needs_grad=True)
    v = qd.field(dtype=qd.f64, shape=(3,), needs_grad=False)
    out = qd.field(dtype=qd.f64, shape=(3,), needs_grad=True)
    parent_idx_minus1 = qd.field(dtype=qd.i32, shape=(1,), needs_grad=False)

    quat_np = np.array([np.sqrt(1 - 0.01), 0.0, 0.1, 0.0])
    pos_np = np.zeros((2, 3))
    pos_np[0] = [0.05, -0.03, 0.02]
    v_np = np.array([0.2, 0.0, 0.0])

    def seed():
        q_init = np.zeros((2, 4))
        q_init[0] = quat_np
        q.from_numpy(q_init)
        pos.from_numpy(pos_np)
        pos_bw.from_numpy(np.zeros((2, 1, 3)))
        v.from_numpy(v_np)
        out.from_numpy(np.zeros(3))
        out.grad.from_numpy(np.ones(3))
        q.grad.from_numpy(np.zeros((2, 4)))
        pos.grad.from_numpy(np.zeros((2, 3)))
        pos_bw.grad.from_numpy(np.zeros((2, 1, 3)))
        parent_idx_minus1.from_numpy(np.array([0], dtype=np.int32))  # i.e., parent exists

    seed()
    step(q, pos, pos_bw, v, out, parent_idx_minus1)
    step.grad(q, pos, pos_bw, v, out, parent_idx_minus1)
    ana_q = np.array([float(q.grad[0, i]) for i in range(4)])
    ana_pos = np.array([float(pos.grad[0, i]) for i in range(3)])

    qw, qx, qy, qz = quat_np
    hand_q = 0.4 * np.array([qw + qz - qy, qx + qy + qz, -qy + qx - qw, -qz + qw + qx])
    hand_pos = np.array([1.0, 1.0, 1.0])

    print(f"{'k':>3}  {'q.grad[0]':>14}  {'hand':>14}  {'diff':>14}  {'match?':>10}")
    for k in range(4):
        diff = ana_q[k] - hand_q[k]
        ok = abs(diff) < 1e-6
        print(f"{k:>3}  {ana_q[k]:>14.6e}  {hand_q[k]:>14.6e}  {diff:>14.3e}  {'OK' if ok else 'MISMATCH':>10}")
    print()
    for k in range(3):
        diff = ana_pos[k] - hand_pos[k]
        ok = abs(diff) < 1e-6
        print(
            f"pos[0][{k}]  {ana_pos[k]:>14.6e}  {hand_pos[k]:>14.6e}  {diff:>14.3e}  {'OK' if ok else 'MISMATCH':>10}"
        )


if __name__ == "__main__":
    main()
