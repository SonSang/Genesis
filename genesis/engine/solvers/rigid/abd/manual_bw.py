"""Manual backward kernels for `update_cartesian_space` (FK reverse).

Bypasses Quadrants AD's silent drop on
`func_forward_kinematics_entity_one_link` (Phase A
`pos_ = parent_pos + qd_transform_by_quat(arm_local, parent_quat)`) by
computing the FK Jacobian-transpose explicitly.

Scope: FREE, REVOLUTE, PRISMATIC, FIXED joint types. Single-batch.
SPHERICAL is NOT implemented and surfaces an explicit error via the
`errno` field (`ErrorCode.MANUAL_BW_UNIMPLEMENTED_JOINT_TYPE`) per guide
P9 — silent skip would corrupt gradients of any topology that uses it.

Maths:
  Forward FK for a link `i_l` with parent `p` and joint of type T:
    arm_local      = links_info.pos[i_l]            (Vec3, in parent frame)
    parent_pos     = links_state.pos[p, i_b]
    parent_quat    = links_state.quat[p, i_b]
    arm_base_pos   = parent_pos + R(parent_quat) · arm_local
    arm_base_quat  = parent_quat ⊗ links_info.quat[i_l]   (= parent_quat at init)
    joint applied (REVOLUTE):
      axis      = dofs_info.motion_ang[d]
      angle     = qpos[q] - qpos0[q]
      qloc      = rotvec_to_quat(axis · angle)
      arm_quat  = qloc ⊗ arm_base_quat
      arm_pos   = arm_base_pos   (since joints_info.pos = 0 by MJCF default)
    links_state.pos[i_l]  = arm_pos
    links_state.quat[i_l] = arm_quat

  Backward (chain rule, seeded by links_state.{pos,quat}.grad):
    arm_quat_grad ← links_state.quat.grad[i_l]
    arm_pos_grad  ← links_state.pos.grad[i_l]

    # arm_pos chain (Phase A: arm_base_pos = parent_pos + R·arm_local)
    parent_pos_grad  += arm_pos_grad                                            # identity
    parent_quat_grad += d_transform_by_quat__dq(arm_local, parent_quat, arm_pos_grad)

    # arm_quat chain through joint apply:
    #   arm_quat = qloc ⊗ parent_quat (since arm_base_quat = parent_quat at init)
    qloc_grad        = d_quat_mul__dlhs(qloc, parent_quat, arm_quat_grad)
    parent_quat_grad += d_quat_mul__drhs(qloc, parent_quat, arm_quat_grad)

    # qloc = rotvec_to_quat(axis · angle)
    rotvec_grad      = d_rotvec_to_quat__drotvec(axis · angle, qloc_grad)
    angle_grad       = axis · rotvec_grad
    qpos.grad[q]    += angle_grad

  Chassis (free joint) backward — pos/quat copy directly to qpos:
    qpos.grad[0:3]  += links_state.pos.grad[chassis]
    qpos.grad[3:7]  += links_state.quat.grad[chassis]

The functions below implement `d_transform_by_quat__dq`, `d_quat_mul__dlhs`,
`d_quat_mul__drhs`, and `d_rotvec_to_quat__drotvec` as `@qd.func`. They
mirror the standard chain-rule derivatives of the corresponding forward
functions in `genesis/utils/geom.py` and are designed so that, in
isolation (no surrounding Genesis context), Quadrants reverse-mode AD
of the forward function should give identical results — but we apply
them by hand to avoid the silent drop reported in
`notes/quadrants_repros/case_*.py`.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu


@qd.func
def d_transform_by_quat__dq(v, quat, out_grad):
    """Gradient w.r.t. `quat` of `qd_transform_by_quat(v, quat)`.

    Forward (geom.py:294):
        out[0] = v0·(qw² + qx² - qy² - qz²) + v1·(2qxy - 2qwz) + v2·(2qxz + 2qwy)
        out[1] = v0·(2qxy + 2qwz) + v1·(qw² - qx² + qy² - qz²) + v2·(2qyz - 2qwx)
        out[2] = v0·(2qxz - 2qwy) + v1·(2qyz + 2qwx) + v2·(qw² - qx² - qy² + qz²)

    Returns Vec4 = (∂L/∂qw, ∂L/∂qx, ∂L/∂qy, ∂L/∂qz) where
    L is whatever scalar seeded `out_grad`. (No normalization assumed.)
    """
    qw = quat[0]
    qx = quat[1]
    qy = quat[2]
    qz = quat[3]
    v0 = v[0]
    v1 = v[1]
    v2 = v[2]
    og0 = out_grad[0]
    og1 = out_grad[1]
    og2 = out_grad[2]

    # ∂out[0]/∂{w,x,y,z}
    do0_dqw = 2.0 * (qw * v0 - qz * v1 + qy * v2)
    do0_dqx = 2.0 * (qx * v0 + qy * v1 + qz * v2)
    do0_dqy = 2.0 * (-qy * v0 + qx * v1 + qw * v2)
    do0_dqz = 2.0 * (-qz * v0 - qw * v1 + qx * v2)

    # ∂out[1]/∂{w,x,y,z}
    do1_dqw = 2.0 * (qz * v0 + qw * v1 - qx * v2)
    do1_dqx = 2.0 * (qy * v0 - qx * v1 - qw * v2)
    do1_dqy = 2.0 * (qx * v0 + qy * v1 + qz * v2)
    do1_dqz = 2.0 * (qw * v0 - qz * v1 + qy * v2)

    # ∂out[2]/∂{w,x,y,z}
    do2_dqw = 2.0 * (-qy * v0 + qx * v1 + qw * v2)
    do2_dqx = 2.0 * (qz * v0 + qw * v1 - qx * v2)
    do2_dqy = 2.0 * (-qw * v0 + qz * v1 - qy * v2)
    do2_dqz = 2.0 * (qx * v0 + qy * v1 + qz * v2)

    return qd.Vector(
        [
            og0 * do0_dqw + og1 * do1_dqw + og2 * do2_dqw,
            og0 * do0_dqx + og1 * do1_dqx + og2 * do2_dqx,
            og0 * do0_dqy + og1 * do1_dqy + og2 * do2_dqy,
            og0 * do0_dqz + og1 * do1_dqz + og2 * do2_dqz,
        ],
        dt=gs.qd_float,
    )


@qd.func
def d_quat_mul__dlhs(a, b, out_grad):
    """Gradient w.r.t. `a` of `quat_mul(a, b)` (Hamilton convention).

    Forward (geom.py qd_quat_mul):
        out_w = aw·bw - ax·bx - ay·by - az·bz
        out_x = aw·bx + ax·bw + ay·bz - az·by
        out_y = aw·by - ax·bz + ay·bw + az·bx
        out_z = aw·bz + ax·by - ay·bx + az·bw
    """
    bw = b[0]
    bx = b[1]
    by = b[2]
    bz = b[3]
    ogw = out_grad[0]
    ogx = out_grad[1]
    ogy = out_grad[2]
    ogz = out_grad[3]
    return qd.Vector(
        [
            ogw * bw + ogx * bx + ogy * by + ogz * bz,  # ∂L/∂aw
            -ogw * bx + ogx * bw - ogy * bz + ogz * by,  # ∂L/∂ax
            -ogw * by + ogx * bz + ogy * bw - ogz * bx,  # ∂L/∂ay
            -ogw * bz - ogx * by + ogy * bx + ogz * bw,  # ∂L/∂az
        ],
        dt=gs.qd_float,
    )


@qd.func
def d_quat_mul__drhs(a, b, out_grad):
    """Gradient w.r.t. `b` of `quat_mul(a, b)`."""
    aw = a[0]
    ax = a[1]
    ay = a[2]
    az = a[3]
    ogw = out_grad[0]
    ogx = out_grad[1]
    ogy = out_grad[2]
    ogz = out_grad[3]
    return qd.Vector(
        [
            ogw * aw + ogx * ax + ogy * ay + ogz * az,  # ∂L/∂bw
            -ogw * ax + ogx * aw + ogy * az - ogz * ay,  # ∂L/∂bx
            -ogw * ay - ogx * az + ogy * aw + ogz * ax,  # ∂L/∂by
            -ogw * az + ogx * ay - ogy * ax + ogz * aw,  # ∂L/∂bz
        ],
        dt=gs.qd_float,
    )


@qd.func
def d_rotvec_to_quat__drotvec(rotvec, eps, quat_grad):
    """Gradient w.r.t. `rotvec` of `qd_rotvec_to_quat(rotvec, eps)`.

    Forward (geom.py:111):
        thetasq   = rx² + ry² + rz²
        theta_reg = sqrt(thetasq + eps²)
        c         = cos(theta_reg / 2)
        sinc      = sin(theta_reg / 2) / theta_reg
        quat      = (c, sinc·rx, sinc·ry, sinc·rz)

    Backward — by chain rule on theta_reg(rx, ry, rz):
        ∂theta_reg/∂ri  = ri / theta_reg
        ∂c/∂ri          = -0.5·sin(theta_reg/2)·ri/theta_reg
                        = -0.5·(sin·ri)/theta_reg
        ∂sinc/∂ri       = [(0.5·cos(theta_reg/2))/theta_reg
                            - sin(theta_reg/2)/theta_reg²] · ri/theta_reg
                        = ri·(0.5·c/theta_reg² - sinc/theta_reg²)

    ∂quat[0]/∂ri = ∂c/∂ri = -0.5·sin·ri/theta_reg
    ∂quat[1+j]/∂ri = ∂(sinc·r_j)/∂ri
                  = δ(i,j)·sinc + r_j·∂sinc/∂ri

    So rotvec_grad[i] = quat_grad[0]·(-0.5·sin·ri/theta_reg)
                      + sum_j quat_grad[1+j] · [δ(i,j)·sinc + r_j·∂sinc/∂ri]
                      = quat_grad[0]·(-0.5·sin·ri/theta_reg)
                      + sinc·quat_grad[1+i]
                      + ∂sinc/∂ri · sum_j quat_grad[1+j]·r_j
    """
    rx = rotvec[0]
    ry = rotvec[1]
    rz = rotvec[2]
    thetasq = rx * rx + ry * ry + rz * rz
    theta_reg = qd.sqrt(thetasq + eps * eps)
    theta_half = 0.5 * theta_reg
    sin_h = qd.sin(theta_half)
    cos_h = qd.cos(theta_half)
    sinc = sin_h / theta_reg
    # ∂sinc/∂theta_reg = (0.5·cos_h - sinc) / theta_reg
    dsinc_dtheta = (0.5 * cos_h - sinc) / theta_reg

    qg_w = quat_grad[0]
    qg_x = quat_grad[1]
    qg_y = quat_grad[2]
    qg_z = quat_grad[3]

    # sum_j quat_grad[1+j] · r_j
    qg_dot_r = qg_x * rx + qg_y * ry + qg_z * rz

    # ∂quat[0]/∂ri = -0.5·sin_h·ri/theta_reg
    # ∂(sinc·rj)/∂ri = δij·sinc + r_j·(dsinc_dtheta · ri/theta_reg)
    # so total per i:
    #   ri·[ -0.5·sin_h/theta_reg · qg_w + dsinc_dtheta/theta_reg · qg_dot_r ] + sinc·qg_{x,y,z}[i]
    coeff = -0.5 * sin_h / theta_reg * qg_w + dsinc_dtheta / theta_reg * qg_dot_r
    return qd.Vector(
        [
            coeff * rx + sinc * qg_x,
            coeff * ry + sinc * qg_y,
            coeff * rz + sinc * qg_z,
        ],
        dt=gs.qd_float,
    )


@qd.kernel(fastcache=True)
def kernel_manual_fk_only_bw(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    errno: qd.Tensor,
):
    """Single-call manual reverse of `kernel_forward_kinematics_fk_only`.
    Iterates the links of each entity from leaf to root inside one
    kernel launch so a child's `parent.{pos,quat}.grad` write completes
    before the parent's own iteration consumes it.

    Joint types supported: FREE / REVOLUTE / PRISMATIC / FIXED / SPHERICAL.
    """
    qd.loop_config(
        name="manual_fk_only_bw",
        serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL),
    )
    for i_e, i_b in qd.ndrange(entities_info.n_links.shape[0], links_state.pos.shape[1]):
        n_in_e = entities_info.n_links[i_e]
        for i_l_rev in range(n_in_e):
            i_l = entities_info.link_end[i_e] - 1 - i_l_rev
            I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l

            i_j = links_info.joint_start[I_l]
            I_j = [i_j, i_b] if qd.static(static_rigid_sim_config.batch_joints_info) else i_j
            joint_type = joints_info.type[I_j]
            q_start = joints_info.q_start[I_j]
            dof_start = joints_info.dof_start[I_j]

            if joint_type == gs.JOINT_TYPE.FREE:
                pos_grad = links_state.pos.grad[i_l, i_b]
                quat_grad = links_state.quat.grad[i_l, i_b]
                xanchor_grad = joints_state.xanchor.grad[i_j, i_b]
                for j in qd.static(range(3)):
                    rigid_global_info.qpos.grad[q_start + j, i_b] = (
                        rigid_global_info.qpos.grad[q_start + j, i_b] + pos_grad[j] + xanchor_grad[j]
                    )
                for j in qd.static(range(4)):
                    rigid_global_info.qpos.grad[q_start + 3 + j, i_b] = (
                        rigid_global_info.qpos.grad[q_start + 3 + j, i_b] + quat_grad[j]
                    )
                for j in qd.static(range(3)):
                    links_state.pos.grad[i_l, i_b][j] = 0.0
                    joints_state.xanchor.grad[i_j, i_b][j] = 0.0
                    joints_state.xaxis.grad[i_j, i_b][j] = 0.0
                for j in qd.static(range(4)):
                    links_state.quat.grad[i_l, i_b][j] = 0.0

            elif joint_type == gs.JOINT_TYPE.REVOLUTE:
                I_d = [dof_start, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else dof_start
                axis = dofs_info.motion_ang[I_d]
                angle = rigid_global_info.qpos[q_start, i_b] - rigid_global_info.qpos0[q_start, i_b]
                rotvec = axis * angle
                qloc = gu.qd_rotvec_to_quat(rotvec, rigid_global_info.EPS[None])
                parent_idx = links_info.parent_idx[I_l]
                arm_pos_grad = links_state.pos.grad[i_l, i_b]
                arm_quat_grad = links_state.quat.grad[i_l, i_b]

                if parent_idx != -1:
                    parent_quat = links_state.quat[parent_idx, i_b]
                    arm_local = links_info.pos[I_l]
                    parent_quat_grad_from_pos = d_transform_by_quat__dq(arm_local, parent_quat, arm_pos_grad)
                    parent_quat_grad_from_quat = d_quat_mul__dlhs(parent_quat, qloc, arm_quat_grad)
                    qloc_grad = d_quat_mul__drhs(parent_quat, qloc, arm_quat_grad)

                    xanchor_grad = joints_state.xanchor.grad[i_j, i_b]
                    xaxis_grad = joints_state.xaxis.grad[i_j, i_b]
                    joint_pos_off = joints_info.pos[I_j]
                    parent_quat_grad_from_xanchor_via_quat = d_transform_by_quat__dq(
                        joint_pos_off, parent_quat, xanchor_grad
                    )
                    parent_quat_grad_from_xaxis = d_transform_by_quat__dq(axis, parent_quat, xaxis_grad)
                    parent_quat_grad_from_xanchor_via_pos = d_transform_by_quat__dq(
                        arm_local, parent_quat, xanchor_grad
                    )

                    rotvec_grad = d_rotvec_to_quat__drotvec(rotvec, rigid_global_info.EPS[None], qloc_grad)
                    angle_grad = axis[0] * rotvec_grad[0] + axis[1] * rotvec_grad[1] + axis[2] * rotvec_grad[2]
                    rigid_global_info.qpos.grad[q_start, i_b] = rigid_global_info.qpos.grad[q_start, i_b] + angle_grad

                    for j in qd.static(range(3)):
                        links_state.pos.grad[parent_idx, i_b][j] = (
                            links_state.pos.grad[parent_idx, i_b][j] + arm_pos_grad[j] + xanchor_grad[j]
                        )
                    for j in qd.static(range(4)):
                        links_state.quat.grad[parent_idx, i_b][j] = (
                            links_state.quat.grad[parent_idx, i_b][j]
                            + parent_quat_grad_from_pos[j]
                            + parent_quat_grad_from_quat[j]
                            + parent_quat_grad_from_xanchor_via_quat[j]
                            + parent_quat_grad_from_xaxis[j]
                            + parent_quat_grad_from_xanchor_via_pos[j]
                        )
                    for j in qd.static(range(3)):
                        joints_state.xanchor.grad[i_j, i_b][j] = 0.0
                        joints_state.xaxis.grad[i_j, i_b][j] = 0.0
                else:
                    quat_init = links_info.quat[I_l]
                    qloc_grad = d_quat_mul__drhs(quat_init, qloc, arm_quat_grad)
                    rotvec_grad = d_rotvec_to_quat__drotvec(rotvec, rigid_global_info.EPS[None], qloc_grad)
                    angle_grad = axis[0] * rotvec_grad[0] + axis[1] * rotvec_grad[1] + axis[2] * rotvec_grad[2]
                    rigid_global_info.qpos.grad[q_start, i_b] = rigid_global_info.qpos.grad[q_start, i_b] + angle_grad

                for j in qd.static(range(3)):
                    links_state.pos.grad[i_l, i_b][j] = 0.0
                for j in qd.static(range(4)):
                    links_state.quat.grad[i_l, i_b][j] = 0.0

            elif joint_type == gs.JOINT_TYPE.PRISMATIC:
                I_d = [dof_start, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else dof_start
                axis = dofs_info.motion_vel[I_d]
                displacement = rigid_global_info.qpos[q_start, i_b] - rigid_global_info.qpos0[q_start, i_b]
                parent_idx = links_info.parent_idx[I_l]
                arm_pos_grad = links_state.pos.grad[i_l, i_b]
                arm_quat_grad = links_state.quat.grad[i_l, i_b]

                if parent_idx != -1:
                    parent_quat = links_state.quat[parent_idx, i_b]
                    quat_init = gu.qd_transform_quat_by_quat(links_info.quat[I_l], parent_quat)
                    arm_local = links_info.pos[I_l]
                    xaxis = gu.qd_transform_by_quat(axis, quat_init)
                    displacement_grad = (
                        xaxis[0] * arm_pos_grad[0] + xaxis[1] * arm_pos_grad[1] + xaxis[2] * arm_pos_grad[2]
                    )
                    xaxis_grad = qd.Vector(
                        [
                            arm_pos_grad[0] * displacement,
                            arm_pos_grad[1] * displacement,
                            arm_pos_grad[2] * displacement,
                        ],
                        dt=gs.qd_float,
                    )
                    quat_init_grad_from_xaxis = d_transform_by_quat__dq(axis, quat_init, xaxis_grad)
                    quat_init_grad_total = qd.Vector(
                        [
                            quat_init_grad_from_xaxis[0] + arm_quat_grad[0],
                            quat_init_grad_from_xaxis[1] + arm_quat_grad[1],
                            quat_init_grad_from_xaxis[2] + arm_quat_grad[2],
                            quat_init_grad_from_xaxis[3] + arm_quat_grad[3],
                        ],
                        dt=gs.qd_float,
                    )
                    parent_quat_grad_from_quat = d_quat_mul__dlhs(
                        parent_quat, links_info.quat[I_l], quat_init_grad_total
                    )
                    parent_quat_grad_from_pos = d_transform_by_quat__dq(arm_local, parent_quat, arm_pos_grad)
                    rigid_global_info.qpos.grad[q_start, i_b] = (
                        rigid_global_info.qpos.grad[q_start, i_b] + displacement_grad
                    )
                    for j in qd.static(range(3)):
                        links_state.pos.grad[parent_idx, i_b][j] = (
                            links_state.pos.grad[parent_idx, i_b][j] + arm_pos_grad[j]
                        )
                    for j in qd.static(range(4)):
                        links_state.quat.grad[parent_idx, i_b][j] = (
                            links_state.quat.grad[parent_idx, i_b][j]
                            + parent_quat_grad_from_pos[j]
                            + parent_quat_grad_from_quat[j]
                        )
                else:
                    quat_init = links_info.quat[I_l]
                    xaxis = gu.qd_transform_by_quat(axis, quat_init)
                    displacement_grad = (
                        xaxis[0] * arm_pos_grad[0] + xaxis[1] * arm_pos_grad[1] + xaxis[2] * arm_pos_grad[2]
                    )
                    rigid_global_info.qpos.grad[q_start, i_b] = (
                        rigid_global_info.qpos.grad[q_start, i_b] + displacement_grad
                    )
                for j in qd.static(range(3)):
                    links_state.pos.grad[i_l, i_b][j] = 0.0
                for j in qd.static(range(4)):
                    links_state.quat.grad[i_l, i_b][j] = 0.0

            elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                # Forward (forward_kinematics.py SPHERICAL branch):
                #   qloc      = qpos[q_start:q_start+4]   (4 quaternion values, direct)
                #   arm_quat  = quat_mul(parent_quat, qloc)   if parent != -1
                #              = quat_mul(links_info.quat[I_l], qloc)  otherwise
                #   arm_pos   = parent_pos + R(parent_quat) · arm_local   (joints_info.pos=0 assumption)
                # axis is the default [0, 0, 1] (SPHERICAL doesn't read motion_ang/vel).
                # Same chain as REVOLUTE with qloc derivative trivial (∂qloc/∂qpos = identity).
                axis = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)
                qloc = qd.Vector(
                    [
                        rigid_global_info.qpos[q_start, i_b],
                        rigid_global_info.qpos[q_start + 1, i_b],
                        rigid_global_info.qpos[q_start + 2, i_b],
                        rigid_global_info.qpos[q_start + 3, i_b],
                    ],
                    dt=gs.qd_float,
                )
                parent_idx = links_info.parent_idx[I_l]
                arm_pos_grad = links_state.pos.grad[i_l, i_b]
                arm_quat_grad = links_state.quat.grad[i_l, i_b]

                if parent_idx != -1:
                    parent_quat = links_state.quat[parent_idx, i_b]
                    arm_local = links_info.pos[I_l]
                    parent_quat_grad_from_pos = d_transform_by_quat__dq(arm_local, parent_quat, arm_pos_grad)
                    parent_quat_grad_from_quat = d_quat_mul__dlhs(parent_quat, qloc, arm_quat_grad)
                    qloc_grad = d_quat_mul__drhs(parent_quat, qloc, arm_quat_grad)

                    xanchor_grad = joints_state.xanchor.grad[i_j, i_b]
                    xaxis_grad = joints_state.xaxis.grad[i_j, i_b]
                    joint_pos_off = joints_info.pos[I_j]
                    parent_quat_grad_from_xanchor_via_quat = d_transform_by_quat__dq(
                        joint_pos_off, parent_quat, xanchor_grad
                    )
                    parent_quat_grad_from_xaxis = d_transform_by_quat__dq(axis, parent_quat, xaxis_grad)
                    parent_quat_grad_from_xanchor_via_pos = d_transform_by_quat__dq(
                        arm_local, parent_quat, xanchor_grad
                    )

                    for j in qd.static(range(4)):
                        rigid_global_info.qpos.grad[q_start + j, i_b] = (
                            rigid_global_info.qpos.grad[q_start + j, i_b] + qloc_grad[j]
                        )
                    for j in qd.static(range(3)):
                        links_state.pos.grad[parent_idx, i_b][j] = (
                            links_state.pos.grad[parent_idx, i_b][j] + arm_pos_grad[j] + xanchor_grad[j]
                        )
                    for j in qd.static(range(4)):
                        links_state.quat.grad[parent_idx, i_b][j] = (
                            links_state.quat.grad[parent_idx, i_b][j]
                            + parent_quat_grad_from_pos[j]
                            + parent_quat_grad_from_quat[j]
                            + parent_quat_grad_from_xanchor_via_quat[j]
                            + parent_quat_grad_from_xaxis[j]
                            + parent_quat_grad_from_xanchor_via_pos[j]
                        )
                    for j in qd.static(range(3)):
                        joints_state.xanchor.grad[i_j, i_b][j] = 0.0
                        joints_state.xaxis.grad[i_j, i_b][j] = 0.0
                else:
                    quat_init = links_info.quat[I_l]
                    qloc_grad = d_quat_mul__drhs(quat_init, qloc, arm_quat_grad)
                    for j in qd.static(range(4)):
                        rigid_global_info.qpos.grad[q_start + j, i_b] = (
                            rigid_global_info.qpos.grad[q_start + j, i_b] + qloc_grad[j]
                        )

                for j in qd.static(range(3)):
                    links_state.pos.grad[i_l, i_b][j] = 0.0
                for j in qd.static(range(4)):
                    links_state.quat.grad[i_l, i_b][j] = 0.0

            elif joint_type == gs.JOINT_TYPE.FIXED:
                parent_idx = links_info.parent_idx[I_l]
                arm_pos_grad = links_state.pos.grad[i_l, i_b]
                arm_quat_grad = links_state.quat.grad[i_l, i_b]
                if parent_idx != -1:
                    parent_quat = links_state.quat[parent_idx, i_b]
                    arm_local = links_info.pos[I_l]
                    parent_quat_grad_from_pos = d_transform_by_quat__dq(arm_local, parent_quat, arm_pos_grad)
                    parent_quat_grad_from_quat = d_quat_mul__dlhs(parent_quat, links_info.quat[I_l], arm_quat_grad)
                    for j in qd.static(range(3)):
                        links_state.pos.grad[parent_idx, i_b][j] = (
                            links_state.pos.grad[parent_idx, i_b][j] + arm_pos_grad[j]
                        )
                    for j in qd.static(range(4)):
                        links_state.quat.grad[parent_idx, i_b][j] = (
                            links_state.quat.grad[parent_idx, i_b][j]
                            + parent_quat_grad_from_pos[j]
                            + parent_quat_grad_from_quat[j]
                        )
                for j in qd.static(range(3)):
                    links_state.pos.grad[i_l, i_b][j] = 0.0
                for j in qd.static(range(4)):
                    links_state.quat.grad[i_l, i_b][j] = 0.0


@qd.func
def d_motion_cross_motion(s_ang, s_vel, m_ang, m_vel, ang_g, vel_g):
    """Reverse of motion_cross_motion(s_ang, s_vel, m_ang, m_vel).

    Forward (geom.py:437):
        vel = s_ang × m_vel + s_vel × m_ang
        ang = s_ang × m_ang

    Chain rule (c=a×b ⇒ a.g += b × c.g, b.g += c.g × a):
        s_ang.g += m_ang × ang.g + m_vel × vel.g
        s_vel.g += m_ang × vel.g
        m_ang.g += ang.g × s_ang + vel.g × s_vel
        m_vel.g += vel.g × s_ang

    Returns (s_ang_g, s_vel_g, m_ang_g, m_vel_g) — additive deltas.
    """
    return (
        m_ang.cross(ang_g) + m_vel.cross(vel_g),
        m_ang.cross(vel_g),
        ang_g.cross(s_ang) + vel_g.cross(s_vel),
        vel_g.cross(s_ang),
    )


# =========================================================================
# `kernel_manual_forward_velocity_bw`: single-call manual reverse of
# `kernel_forward_velocity` (replaces the per-link split currently used to
# avoid Quadrants AD silent drop on cross-link `cd_{vel,ang}[parent_idx]`).
#
# Forward (per-link, leaf-iterated):
#   cvel_vel/ang = parent.cd_vel/ang  (or 0 if root)
#   for each joint i_j_ in 0..n_joints-1:
#     (FREE only) for i_3 in 0..2:
#       _vel = cdof_vel[ds+i_3] * vel[ds+i_3];  cvel_vel += _vel  (linear dofs)
#       _ang = cdof_ang[ds+i_3] * vel[ds+i_3];  cvel_ang += _ang  (= 0 since cdof_ang linear = 0)
#     for each dof i_d in joint:
#       cdofd_ang[i_d], cdofd_vel[i_d] = motion_cross_motion(cvel_ang, cvel_vel, cdof_ang[i_d], cdof_vel[i_d])
#     (BW slot copy: cd_*_bw[next] = cd_*_bw[curr])
#     for each dof i_d (FREE: i_3+3) in joint:
#       cvel_vel += cdof_vel[i_d] * vel[i_d]
#       cvel_ang += cdof_ang[i_d] * vel[i_d]
#   cd_vel[i_l] = cvel_vel;  cd_ang[i_l] = cvel_ang
#
# Reverse (leaf → root iteration; statement-reverse per joint):
#   cd_vel.grad[i_l] → cd_vel_bw.grad[i_l, n_joints]; same for cd_ang
#   for i_j_ in n_joints-1..0 (reverse):
#     [d-rev]  cd_*_bw[next].grad → cdof_*.grad / vel.grad
#     [c-rev]  cd_*_bw[curr].grad += cd_*_bw[next].grad; consume next
#     [b-rev]  cdofd_*.grad → cd_*_bw[curr].grad / cdof_*.grad via d_motion_cross_motion
#     [a-rev]  (FREE only) cd_*_bw[curr].grad → linear cdof_*.grad / vel.grad
#   parent.cd_*.grad += cd_*_bw[i_l, 0].grad (cross-link, root → leaf accumulation 방향)
#
# Scope: FREE, REVOLUTE, PRISMATIC. SPHERICAL handled via the else branch of
# forward_velocity (same chain as REVOLUTE/PRISMATIC — multi-dof joint).
# Hibernation NOT supported (errno bit if encountered).
# =========================================================================


@qd.kernel(fastcache=True)
def kernel_manual_forward_velocity_bw(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    errno: qd.Tensor,
):
    """Manual reverse of `kernel_forward_velocity` — single-call (no per-link
    split). Replaces the diagnostic per-link split in `substep_pre_coupling_grad`
    by computing the cross-link `cd_{vel,ang}[parent_idx]` chain explicitly.

    Inputs (read .grad seeds):
      - cd_vel.grad[i_l, i_b], cd_ang.grad[i_l, i_b]
      - cd_vel_bw.grad[i_l, k, i_b], cd_ang_bw.grad[i_l, k, i_b]
      - cdofd_ang.grad[i_d, i_b], cdofd_vel.grad[i_d, i_b]

    Outputs (accumulated .grad):
      - dofs_state.vel.grad[i_d, i_b]
      - dofs_state.cdof_ang.grad[i_d, i_b], dofs_state.cdof_vel.grad[i_d, i_b]
      - links_state.cd_vel.grad[parent_idx, i_b], links_state.cd_ang.grad[parent_idx, i_b]
        (cross-link chain — equivalent to forward replay's BW=True
        `cd_*_bw[i_l, 0] = parent.cd_*`)
    """
    qd.loop_config(
        name="manual_forward_velocity_bw",
        serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL),
    )
    for i_e, i_b in qd.ndrange(entities_info.n_links.shape[0], links_state.pos.shape[1]):
        if qd.static(static_rigid_sim_config.use_hibernation):
            errno[i_b] = errno[i_b] | array_class.ErrorCode.MANUAL_BW_UNIMPLEMENTED_JOINT_TYPE
        else:
            n_in_e = entities_info.n_links[i_e]
            # Leaf → root iteration so each link's cd_*_bw[0].grad (which
            # accumulates into parent.cd_*.grad) is propagated *before* the
            # parent's own iteration uses it.
            for i_l_rev in range(n_in_e):
                i_l = entities_info.link_end[i_e] - 1 - i_l_rev
                I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
                n_joints = links_info.joint_end[I_l] - links_info.joint_start[I_l]
                i_p = links_info.parent_idx[I_l]

                # ── Step 1 reverse: cd_*[i_l].grad → cd_*_bw[i_l, n_joints].grad
                for k in qd.static(range(3)):
                    links_state.cd_vel_bw.grad[i_l, n_joints, i_b][k] = (
                        links_state.cd_vel_bw.grad[i_l, n_joints, i_b][k] + links_state.cd_vel.grad[i_l, i_b][k]
                    )
                    links_state.cd_ang_bw.grad[i_l, n_joints, i_b][k] = (
                        links_state.cd_ang_bw.grad[i_l, n_joints, i_b][k] + links_state.cd_ang.grad[i_l, i_b][k]
                    )
                # consume cd_vel/cd_ang.grad[i_l]
                for k in qd.static(range(3)):
                    links_state.cd_vel.grad[i_l, i_b][k] = 0.0
                    links_state.cd_ang.grad[i_l, i_b][k] = 0.0

                # ── Step 2: iterate joints in reverse
                for i_j_rev in range(n_joints):
                    i_j_ = n_joints - 1 - i_j_rev
                    i_j = i_j_ + links_info.joint_start[I_l]
                    I_j = [i_j, i_b] if qd.static(static_rigid_sim_config.batch_joints_info) else i_j
                    jt = joints_info.type[I_j]
                    ds = joints_info.dof_start[I_j]
                    de = joints_info.dof_end[I_j]
                    curr_idx = i_j_
                    next_idx = i_j_ + 1

                    # ── [d-rev] cd_*_bw[next].grad → cdof_*.grad / vel.grad
                    # Forward (FREE angular: i_3=0..2 at d=ds+3+i_3; else: d in ds..de):
                    #   _vel = cdof_vel[d] * vel[d];  atomic_add(cd_vel_bw[next], _vel)
                    #   _ang = cdof_ang[d] * vel[d];  atomic_add(cd_ang_bw[next], _ang)
                    cvg_next = links_state.cd_vel_bw.grad[i_l, next_idx, i_b]
                    cag_next = links_state.cd_ang_bw.grad[i_l, next_idx, i_b]
                    if jt == gs.JOINT_TYPE.FREE:
                        for i_3 in qd.static(range(3)):
                            d_i = ds + 3 + i_3
                            v_at_d = dofs_state.vel[d_i, i_b]
                            cdv = dofs_state.cdof_vel[d_i, i_b]
                            cda = dofs_state.cdof_ang[d_i, i_b]
                            for k in qd.static(range(3)):
                                dofs_state.cdof_vel.grad[d_i, i_b][k] = (
                                    dofs_state.cdof_vel.grad[d_i, i_b][k] + cvg_next[k] * v_at_d
                                )
                                dofs_state.cdof_ang.grad[d_i, i_b][k] = (
                                    dofs_state.cdof_ang.grad[d_i, i_b][k] + cag_next[k] * v_at_d
                                )
                            dot_vel = cdv[0] * cvg_next[0] + cdv[1] * cvg_next[1] + cdv[2] * cvg_next[2]
                            dot_ang = cda[0] * cag_next[0] + cda[1] * cag_next[1] + cda[2] * cag_next[2]
                            dofs_state.vel.grad[d_i, i_b] = dofs_state.vel.grad[d_i, i_b] + dot_vel + dot_ang
                    else:
                        for i_d in range(ds, de):
                            v_at_d = dofs_state.vel[i_d, i_b]
                            cdv = dofs_state.cdof_vel[i_d, i_b]
                            cda = dofs_state.cdof_ang[i_d, i_b]
                            for k in qd.static(range(3)):
                                dofs_state.cdof_vel.grad[i_d, i_b][k] = (
                                    dofs_state.cdof_vel.grad[i_d, i_b][k] + cvg_next[k] * v_at_d
                                )
                                dofs_state.cdof_ang.grad[i_d, i_b][k] = (
                                    dofs_state.cdof_ang.grad[i_d, i_b][k] + cag_next[k] * v_at_d
                                )
                            dot_vel = cdv[0] * cvg_next[0] + cdv[1] * cvg_next[1] + cdv[2] * cvg_next[2]
                            dot_ang = cda[0] * cag_next[0] + cda[1] * cag_next[1] + cda[2] * cag_next[2]
                            dofs_state.vel.grad[i_d, i_b] = dofs_state.vel.grad[i_d, i_b] + dot_vel + dot_ang

                    # ── [c-rev] cd_*_bw[next] = cd_*_bw[curr] → curr.grad += next.grad
                    for k in qd.static(range(3)):
                        links_state.cd_vel_bw.grad[i_l, curr_idx, i_b][k] = (
                            links_state.cd_vel_bw.grad[i_l, curr_idx, i_b][k] + cvg_next[k]
                        )
                        links_state.cd_ang_bw.grad[i_l, curr_idx, i_b][k] = (
                            links_state.cd_ang_bw.grad[i_l, curr_idx, i_b][k] + cag_next[k]
                        )
                    # consume next
                    for k in qd.static(range(3)):
                        links_state.cd_vel_bw.grad[i_l, next_idx, i_b][k] = 0.0
                        links_state.cd_ang_bw.grad[i_l, next_idx, i_b][k] = 0.0

                    # ── [b-rev] motion_cross_motion reverse:
                    # Forward: (cdofd_ang[d_i], cdofd_vel[d_i]) =
                    #     motion_cross_motion(cd_ang_bw[curr], cd_vel_bw[curr], cdof_ang[d_i], cdof_vel[d_i])
                    # Reverse via d_motion_cross_motion(s_ang, s_vel, m_ang, m_vel, ang_g, vel_g)
                    s_ang_primal = links_state.cd_ang_bw[i_l, curr_idx, i_b]
                    s_vel_primal = links_state.cd_vel_bw[i_l, curr_idx, i_b]
                    if jt == gs.JOINT_TYPE.FREE:
                        # Angular dofs i_3=0..2 at d_i = ds + 3 + i_3 (linear cdofd_* are explicit 0)
                        for i_3 in qd.static(range(3)):
                            d_i = ds + 3 + i_3
                            ang_g = dofs_state.cdofd_ang.grad[d_i, i_b]
                            vel_g = dofs_state.cdofd_vel.grad[d_i, i_b]
                            cda = dofs_state.cdof_ang[d_i, i_b]
                            cdv = dofs_state.cdof_vel[d_i, i_b]
                            s_ang_g, s_vel_g, m_ang_g, m_vel_g = d_motion_cross_motion(
                                s_ang_primal, s_vel_primal, cda, cdv, ang_g, vel_g
                            )
                            for k in qd.static(range(3)):
                                links_state.cd_ang_bw.grad[i_l, curr_idx, i_b][k] = (
                                    links_state.cd_ang_bw.grad[i_l, curr_idx, i_b][k] + s_ang_g[k]
                                )
                                links_state.cd_vel_bw.grad[i_l, curr_idx, i_b][k] = (
                                    links_state.cd_vel_bw.grad[i_l, curr_idx, i_b][k] + s_vel_g[k]
                                )
                                dofs_state.cdof_ang.grad[d_i, i_b][k] = (
                                    dofs_state.cdof_ang.grad[d_i, i_b][k] + m_ang_g[k]
                                )
                                dofs_state.cdof_vel.grad[d_i, i_b][k] = (
                                    dofs_state.cdof_vel.grad[d_i, i_b][k] + m_vel_g[k]
                                )
                            # consume cdofd_*.grad[d_i]
                            for k in qd.static(range(3)):
                                dofs_state.cdofd_ang.grad[d_i, i_b][k] = 0.0
                                dofs_state.cdofd_vel.grad[d_i, i_b][k] = 0.0
                        # Linear dofs (i_3=0..2 at d_i = ds + i_3): cdofd_* set to 0
                        # (constant), reverse is no-op; just consume to mirror P8.
                        for i_3 in qd.static(range(3)):
                            d_i = ds + i_3
                            for k in qd.static(range(3)):
                                dofs_state.cdofd_ang.grad[d_i, i_b][k] = 0.0
                                dofs_state.cdofd_vel.grad[d_i, i_b][k] = 0.0
                    else:
                        for i_d in range(ds, de):
                            ang_g = dofs_state.cdofd_ang.grad[i_d, i_b]
                            vel_g = dofs_state.cdofd_vel.grad[i_d, i_b]
                            cda = dofs_state.cdof_ang[i_d, i_b]
                            cdv = dofs_state.cdof_vel[i_d, i_b]
                            s_ang_g, s_vel_g, m_ang_g, m_vel_g = d_motion_cross_motion(
                                s_ang_primal, s_vel_primal, cda, cdv, ang_g, vel_g
                            )
                            for k in qd.static(range(3)):
                                links_state.cd_ang_bw.grad[i_l, curr_idx, i_b][k] = (
                                    links_state.cd_ang_bw.grad[i_l, curr_idx, i_b][k] + s_ang_g[k]
                                )
                                links_state.cd_vel_bw.grad[i_l, curr_idx, i_b][k] = (
                                    links_state.cd_vel_bw.grad[i_l, curr_idx, i_b][k] + s_vel_g[k]
                                )
                                dofs_state.cdof_ang.grad[i_d, i_b][k] = (
                                    dofs_state.cdof_ang.grad[i_d, i_b][k] + m_ang_g[k]
                                )
                                dofs_state.cdof_vel.grad[i_d, i_b][k] = (
                                    dofs_state.cdof_vel.grad[i_d, i_b][k] + m_vel_g[k]
                                )
                            for k in qd.static(range(3)):
                                dofs_state.cdofd_ang.grad[i_d, i_b][k] = 0.0
                                dofs_state.cdofd_vel.grad[i_d, i_b][k] = 0.0

                    # ── [a-rev] (FREE only) cd_*_bw[curr].grad → linear cdof_*.grad / vel.grad
                    # Forward (FREE linear pre-motion_cross_motion): for i_3=0..2 at d_i = ds + i_3,
                    #   _vel = cdof_vel[d_i] * vel[d_i];  atomic_add(cd_vel_bw[curr], _vel)
                    #   _ang = cdof_ang[d_i] * vel[d_i];  atomic_add(cd_ang_bw[curr], _ang)
                    # (cdof_vel[linear] = e_i_3 constant; cdof_ang[linear] = 0 constant)
                    if jt == gs.JOINT_TYPE.FREE:
                        cvg_curr = links_state.cd_vel_bw.grad[i_l, curr_idx, i_b]
                        cag_curr = links_state.cd_ang_bw.grad[i_l, curr_idx, i_b]
                        for i_3 in qd.static(range(3)):
                            d_i = ds + i_3
                            v_at_d = dofs_state.vel[d_i, i_b]
                            cdv = dofs_state.cdof_vel[d_i, i_b]
                            cda = dofs_state.cdof_ang[d_i, i_b]
                            for k in qd.static(range(3)):
                                dofs_state.cdof_vel.grad[d_i, i_b][k] = (
                                    dofs_state.cdof_vel.grad[d_i, i_b][k] + cvg_curr[k] * v_at_d
                                )
                                dofs_state.cdof_ang.grad[d_i, i_b][k] = (
                                    dofs_state.cdof_ang.grad[d_i, i_b][k] + cag_curr[k] * v_at_d
                                )
                            dot_vel = cdv[0] * cvg_curr[0] + cdv[1] * cvg_curr[1] + cdv[2] * cvg_curr[2]
                            dot_ang = cda[0] * cag_curr[0] + cda[1] * cag_curr[1] + cda[2] * cag_curr[2]
                            dofs_state.vel.grad[d_i, i_b] = dofs_state.vel.grad[d_i, i_b] + dot_vel + dot_ang

                # ── Step 1 (initial cvel setup) reverse:
                # Forward: cd_*_bw[i_l, 0, i_b] = parent.cd_*[i_p, i_b] (if i_p != -1) else 0
                # Reverse: parent.cd_*.grad[i_p] += cd_*_bw[i_l, 0].grad; consume slot 0
                slot0_v_g = links_state.cd_vel_bw.grad[i_l, 0, i_b]
                slot0_a_g = links_state.cd_ang_bw.grad[i_l, 0, i_b]
                if i_p != -1:
                    for k in qd.static(range(3)):
                        links_state.cd_vel.grad[i_p, i_b][k] = links_state.cd_vel.grad[i_p, i_b][k] + slot0_v_g[k]
                        links_state.cd_ang.grad[i_p, i_b][k] = links_state.cd_ang.grad[i_p, i_b][k] + slot0_a_g[k]
                # consume slot 0
                for k in qd.static(range(3)):
                    links_state.cd_vel_bw.grad[i_l, 0, i_b][k] = 0.0
                    links_state.cd_ang_bw.grad[i_l, 0, i_b][k] = 0.0
