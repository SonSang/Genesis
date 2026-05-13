"""Case 8 — Same forward pattern but with Vec3 fields (not scalar fields).

Genesis's `links_state.pos` and `links_info.pos` are `qd.Vector.field(3, ...)`
(vector-valued fields). Cases 3-7 used `qd.field(qd.f64, shape=(2, 3))`
which is a *scalar* field accessed as `pos[i, j]`. The difference might
matter for Quadrants AD when a self-product (q_yy = q_y * q_y) is inside
a `qd.func` called on a *Vec3 field indexed-read*.

Forward (mirrors Genesis line 698-700):
    parent_pos  = links_pos[0]                              # Vec3 indexed read
    parent_quat = links_quat[0]                             # Vec4 indexed read
    arm_local   = info_pos[1]                               # Vec3 indexed read
    out_pos     = parent_pos + transform_by_quat(arm_local, parent_quat)
    links_pos[1] = out_pos                                  # cross-index Vec3 write

Seed: out.grad = [1, 1, 1] (sum of links_pos[1] components).
Expected: links_quat[0].grad[2] = -0.438 (same as cases 3-7).
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
    links_pos: qd.template(),
    links_quat: qd.template(),
    info_pos: qd.template(),
    out: qd.template(),
):
    parent_pos = links_pos[0]
    parent_quat = links_quat[0]
    arm_local = info_pos[1]
    out_pos = parent_pos + transform_by_quat(arm_local, parent_quat)
    links_pos[1] = out_pos

    out[0] = links_pos[1][0]
    out[1] = links_pos[1][1]
    out[2] = links_pos[1][2]


def main():
    # Vec3 / Vec4 fields — mirrors Genesis links_state.pos / links_state.quat
    links_pos = qd.Vector.field(3, dtype=qd.f64, shape=(2,), needs_grad=True)
    links_quat = qd.Vector.field(4, dtype=qd.f64, shape=(2,), needs_grad=True)
    info_pos = qd.Vector.field(3, dtype=qd.f64, shape=(2,), needs_grad=False)
    out = qd.field(dtype=qd.f64, shape=(3,), needs_grad=True)

    quat_np = np.array([np.sqrt(1 - 0.01), 0.0, 0.1, 0.0])
    info_pos_np = np.zeros((2, 3))
    info_pos_np[1] = [0.2, 0.0, 0.0]  # arm's link_info.pos

    def seed():
        links_pos_np = np.zeros((2, 3))
        links_pos_np[0] = [0.05, -0.03, 0.02]
        links_pos.from_numpy(links_pos_np)
        links_quat_np = np.zeros((2, 4))
        links_quat_np[0] = quat_np
        links_quat.from_numpy(links_quat_np)
        info_pos.from_numpy(info_pos_np)
        out.from_numpy(np.zeros(3))
        out.grad.from_numpy(np.ones(3))
        links_pos.grad.from_numpy(np.zeros((2, 3)))
        links_quat.grad.from_numpy(np.zeros((2, 4)))

    seed()
    step(links_pos, links_quat, info_pos, out)
    step.grad(links_pos, links_quat, info_pos, out)
    ana_quat = np.array([float(links_quat.grad[0][k]) for k in range(4)])
    ana_pos = np.array([float(links_pos.grad[0][k]) for k in range(3)])

    qw, qx, qy, qz = quat_np
    hand_quat = 0.4 * np.array([qw + qz - qy, qx + qy + qz, -qy + qx - qw, -qz + qw + qx])
    hand_pos = np.array([1.0, 1.0, 1.0])

    print(f"{'k':>3}  {'links_quat[0]':>14}  {'hand':>14}  {'diff':>14}  {'match?':>10}")
    for k in range(4):
        diff = ana_quat[k] - hand_quat[k]
        ok = abs(diff) < 1e-6
        print(f"{k:>3}  {ana_quat[k]:>14.6e}  {hand_quat[k]:>14.6e}  {diff:>14.3e}  {'OK' if ok else 'MISMATCH':>10}")
    print()
    for k in range(3):
        diff = ana_pos[k] - hand_pos[k]
        ok = abs(diff) < 1e-6
        print(
            f"links_pos[0][{k}]  {ana_pos[k]:>14.6e}  {hand_pos[k]:>14.6e}  {diff:>14.3e}  {'OK' if ok else 'MISMATCH':>10}"
        )


if __name__ == "__main__":
    main()
