"""Case 5 — Add parent_pos to the transform output, mirroring Genesis's
`pos_ = parent_pos + qd_transform_by_quat(arm_local, parent_quat)`.

Hypothesis: maybe the chain `out = parent_pos + R·v` (combining a read
from same-buffer different-index with a transform_by_quat result)
silently drops part of the transform's backward chain.

Forward (added to case 4):
    out[i] = parent_pos[i] + transform_by_quat(v, q[0])[i]

where parent_pos = pos[0] (read from same buffer 'pos' as we also write
pos[1]).

Same seed and baseline as case 3/4. Expected unchanged (parent_pos
contribution is independent of q).
"""

import numpy as np
import quadrants as qd


qd.init(arch=qd.cpu)


@qd.kernel
def step(q: qd.template(), pos: qd.template(), v: qd.template(), out: qd.template()):
    # arm-link cache writes — same as case 4
    q[1, 0] = q[0, 0]
    q[1, 1] = q[0, 1]
    q[1, 2] = q[0, 2]
    q[1, 3] = q[0, 3]

    # Read parent_pos from same buffer 'pos' that we also write pos[1] to.
    parent_pos_0 = pos[0, 0]
    parent_pos_1 = pos[0, 1]
    parent_pos_2 = pos[0, 2]

    # qd_transform_by_quat body
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

    t0 = v0 * (q_xx + q_ww - q_yy - q_zz) + v1 * (2.0 * q_xy - 2.0 * q_wz) + v2 * (2.0 * q_xz + 2.0 * q_wy)
    t1 = v0 * (2.0 * q_xy + 2.0 * q_wz) + v1 * (q_yy + q_ww - q_xx - q_zz) + v2 * (2.0 * q_yz - 2.0 * q_wx)
    t2 = v0 * (2.0 * q_xz - 2.0 * q_wy) + v1 * (2.0 * q_yz + 2.0 * q_wx) + v2 * (q_zz + q_ww - q_xx - q_yy)

    # write pos[1] (= parent_pos + transform_output) — same buffer as parent_pos read
    pos[1, 0] = parent_pos_0 + t0
    pos[1, 1] = parent_pos_1 + t1
    pos[1, 2] = parent_pos_2 + t2

    # Also expose to out so we can seed backward easily (seed sum of pos[1]).
    out[0] = pos[1, 0]
    out[1] = pos[1, 1]
    out[2] = pos[1, 2]


def main():
    q = qd.field(dtype=qd.f64, shape=(2, 4), needs_grad=True)
    pos = qd.field(dtype=qd.f64, shape=(2, 3), needs_grad=True)
    v = qd.field(dtype=qd.f64, shape=(3,), needs_grad=False)
    out = qd.field(dtype=qd.f64, shape=(3,), needs_grad=True)

    quat_np = np.array([np.sqrt(1 - 0.01), 0.0, 0.1, 0.0])
    pos_np = np.zeros((2, 3))
    pos_np[0] = [0.05, -0.03, 0.02]  # nontrivial parent_pos so chain is observable
    v_np = np.array([0.2, 0.0, 0.0])

    def seed():
        q_init = np.zeros((2, 4))
        q_init[0] = quat_np
        q.from_numpy(q_init)
        pos.from_numpy(pos_np)
        v.from_numpy(v_np)
        out.from_numpy(np.zeros(3))
        out.grad.from_numpy(np.ones(3))
        q.grad.from_numpy(np.zeros((2, 4)))
        pos.grad.from_numpy(np.zeros((2, 3)))

    seed()
    step(q, pos, v, out)
    step.grad(q, pos, v, out)
    ana_q = np.array([float(q.grad[0, i]) for i in range(4)])
    ana_pos = np.array([float(pos.grad[0, i]) for i in range(3)])
    out_np = np.array([float(out[i]) for i in range(3)])

    qw, qx, qy, qz = quat_np
    # transform_by_quat output gradient = out.grad = [1, 1, 1].
    # So q.grad receives the same contributions as case 3.
    hand_q = 0.4 * np.array([qw + qz - qy, qx + qy + qz, -qy + qx - qw, -qz + qw + qx])
    # pos[0].grad = out.grad (identity through addition) = [1, 1, 1].
    hand_pos = np.array([1.0, 1.0, 1.0])

    print("baseline q[0]      =", quat_np)
    print("baseline pos[0]    =", pos_np[0])
    print("forward out        =", out_np)
    print()
    print("--- q[0].grad ---")
    print(f"{'k':>3}  {'analytical':>14}  {'hand':>14}  {'diff':>14}  {'match?':>10}")
    print("-" * 70)
    for k in range(4):
        diff = ana_q[k] - hand_q[k]
        ok = abs(diff) < 1e-6
        print(f"{k:>3}  {ana_q[k]:>14.6e}  {hand_q[k]:>14.6e}  {diff:>14.3e}  {'OK' if ok else 'MISMATCH':>10}")
    print()
    print("--- pos[0].grad ---")
    print(f"{'k':>3}  {'analytical':>14}  {'hand':>14}  {'diff':>14}  {'match?':>10}")
    print("-" * 70)
    for k in range(3):
        diff = ana_pos[k] - hand_pos[k]
        ok = abs(diff) < 1e-6
        print(f"{k:>3}  {ana_pos[k]:>14.6e}  {hand_pos[k]:>14.6e}  {diff:>14.3e}  {'OK' if ok else 'MISMATCH':>10}")


if __name__ == "__main__":
    main()
