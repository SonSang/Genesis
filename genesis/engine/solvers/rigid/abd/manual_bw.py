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
def kernel_manual_uc_bw_one_link(
    i_l_offset: qd.i32,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_info: array_class.JointsInfo,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    errno: qd.Tensor,
):
    """Manual backward (FK Jacobian-transpose) for one link, J4/J5 minimal.

    Inputs (read .grad seeds):
      - links_state.pos.grad[i_l, i_b]
      - links_state.quat.grad[i_l, i_b]

    Inputs (read forward primal):
      - links_state.pos[parent_idx, i_b], links_state.quat[parent_idx, i_b]
      - links_info.pos[i_l]                                (parent-frame offset)
      - dofs_info.motion_ang[dof_start]                    (joint axis)
      - rigid_global_info.qpos[q_start], qpos0[q_start]    (joint angle)

    Outputs (accumulated .grad seeds):
      - links_state.pos.grad[parent_idx, i_b]
      - links_state.quat.grad[parent_idx, i_b]
      - rigid_global_info.qpos.grad[q_start..q_end]

    Joint types supported: FREE, REVOLUTE, PRISMATIC, FIXED.
    SPHERICAL is NOT supported — its qpos.grad chain will be silently
    dropped if reached. Per guide P9 (faithful replication), this kernel
    must be extended to cover SPHERICAL before it can be a complete drop-in
    replacement for `kernel_update_cartesian_space_one_link.grad`.
    """
    qd.loop_config(
        name="manual_uc_bw_one_link",
        serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL),
    )
    for i_e, i_b in qd.ndrange(entities_info.n_links.shape[0], links_state.pos.shape[1]):
        n_links_in_entity = entities_info.link_end[i_e] - entities_info.link_start[i_e]
        if i_l_offset < n_links_in_entity:
            i_l = entities_info.link_start[i_e] + i_l_offset
            I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l

            i_j = links_info.joint_start[I_l]
            I_j = [i_j, i_b] if qd.static(static_rigid_sim_config.batch_joints_info) else i_j
            joint_type = joints_info.type[I_j]
            q_start = joints_info.q_start[I_j]
            dof_start = joints_info.dof_start[I_j]

            if joint_type == gs.JOINT_TYPE.FREE:
                # chassis_pos = qpos[q_start:q_start+3]
                # chassis_quat = qpos[q_start+3:q_start+7]
                pos_grad = links_state.pos.grad[i_l, i_b]
                quat_grad = links_state.quat.grad[i_l, i_b]
                for j in qd.static(range(3)):
                    rigid_global_info.qpos.grad[q_start + j, i_b] = (
                        rigid_global_info.qpos.grad[q_start + j, i_b] + pos_grad[j]
                    )
                for j in qd.static(range(4)):
                    rigid_global_info.qpos.grad[q_start + 3 + j, i_b] = (
                        rigid_global_info.qpos.grad[q_start + 3 + j, i_b] + quat_grad[j]
                    )
                # Per guide P8: zero consumed input .grad to mirror auto-AD
                # consume-after-use convention. Otherwise subsequent kernel
                # calls reading the same field will double-count.
                for j in qd.static(range(3)):
                    links_state.pos.grad[i_l, i_b][j] = 0.0
                for j in qd.static(range(4)):
                    links_state.quat.grad[i_l, i_b][j] = 0.0

            elif joint_type == gs.JOINT_TYPE.REVOLUTE:
                # Forward:
                #   If has parent:
                #     pos_init  = parent_pos + R(parent_quat) · links_info.pos[I_l]
                #     quat_init = quat_mul(qloc, parent_quat)  via qd_transform_quat_by_quat
                #   else:
                #     pos_init  = links_info.pos[I_l]    (constant)
                #     quat_init = links_info.quat[I_l]   (constant)
                #   qloc      = rotvec_to_quat(axis · angle)
                #   link_quat = quat_mul(qloc, quat_init)
                #   link_pos  = pos_init   (assuming joints_info.pos = 0, true for default MJCF)
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
                    # arm_pos = parent_pos + R(parent_quat) · arm_local
                    parent_quat_grad_from_pos = d_transform_by_quat__dq(arm_local, parent_quat, arm_pos_grad)

                    # arm_quat = qloc ⊗ parent_quat (lhs=qloc, rhs=parent_quat)
                    qloc_grad = d_quat_mul__dlhs(qloc, parent_quat, arm_quat_grad)
                    parent_quat_grad_from_quat = d_quat_mul__drhs(qloc, parent_quat, arm_quat_grad)

                    # qloc = rotvec_to_quat(axis · angle)
                    rotvec_grad = d_rotvec_to_quat__drotvec(rotvec, rigid_global_info.EPS[None], qloc_grad)
                    angle_grad = axis[0] * rotvec_grad[0] + axis[1] * rotvec_grad[1] + axis[2] * rotvec_grad[2]
                    rigid_global_info.qpos.grad[q_start, i_b] = rigid_global_info.qpos.grad[q_start, i_b] + angle_grad

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
                    # No parent: pos_init/quat_init are link-level constants.
                    # arm_pos.grad doesn't chain anywhere (constant).
                    # arm_quat = qloc ⊗ quat_init  ⇒  qloc_grad = d_quat_mul__dlhs.
                    quat_init = links_info.quat[I_l]
                    qloc_grad = d_quat_mul__dlhs(qloc, quat_init, arm_quat_grad)
                    rotvec_grad = d_rotvec_to_quat__drotvec(rotvec, rigid_global_info.EPS[None], qloc_grad)
                    angle_grad = axis[0] * rotvec_grad[0] + axis[1] * rotvec_grad[1] + axis[2] * rotvec_grad[2]
                    rigid_global_info.qpos.grad[q_start, i_b] = rigid_global_info.qpos.grad[q_start, i_b] + angle_grad

                # Per guide P8: zero consumed input .grad for this link.
                for j in qd.static(range(3)):
                    links_state.pos.grad[i_l, i_b][j] = 0.0
                for j in qd.static(range(4)):
                    links_state.quat.grad[i_l, i_b][j] = 0.0

            elif joint_type == gs.JOINT_TYPE.PRISMATIC:
                # Forward:
                #   parent_idx = links_info.parent_idx[I_l]
                #   If has parent:
                #     pos_init  = parent_pos + R(parent_quat) · links_info.pos[I_l]
                #     quat_init = parent_quat ⊗ links_info.quat[I_l]
                #   else:
                #     pos_init  = links_info.pos[I_l]    (constant)
                #     quat_init = links_info.quat[I_l]   (constant)
                #   xaxis = R(quat_init) · motion_vel
                #   displacement = qpos[q_start] - qpos0[q_start]
                #   link_pos  = pos_init + xaxis · displacement
                #   link_quat = quat_init  (unchanged by prismatic)
                I_d = [dof_start, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else dof_start
                axis = dofs_info.motion_vel[I_d]
                displacement = rigid_global_info.qpos[q_start, i_b] - rigid_global_info.qpos0[q_start, i_b]
                parent_idx = links_info.parent_idx[I_l]
                arm_pos_grad = links_state.pos.grad[i_l, i_b]
                arm_quat_grad = links_state.quat.grad[i_l, i_b]

                if parent_idx != -1:
                    parent_quat = links_state.quat[parent_idx, i_b]
                    # quat_init = parent_quat ⊗ links_info.quat[I_l]
                    quat_init = gu.qd_transform_quat_by_quat(links_info.quat[I_l], parent_quat)
                    arm_local = links_info.pos[I_l]
                    # xaxis = R(quat_init) · motion_vel
                    # link_pos = parent_pos + R(parent_quat) · arm_local + xaxis · displacement
                    #
                    # Reverse:
                    #   parent_pos.grad += arm_pos_grad
                    #   parent_quat.grad += d_transform_by_quat__dq(arm_local, parent_quat, arm_pos_grad)
                    #   xaxis_grad = arm_pos_grad · displacement
                    #   displacement_grad = (R(quat_init) · motion_vel) · arm_pos_grad
                    #                      = qd_transform_by_quat(motion_vel, quat_init) · arm_pos_grad
                    #   quat_init_grad += d_transform_by_quat__dq(motion_vel, quat_init, xaxis_grad)
                    #   quat_init_grad += arm_quat_grad
                    #   parent_quat.grad += d_quat_mul__drhs(links_info.quat[I_l], parent_quat, quat_init_grad)
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
                    parent_quat_grad_from_quat = d_quat_mul__drhs(
                        links_info.quat[I_l], parent_quat, quat_init_grad_total
                    )
                    parent_quat_grad_from_pos = d_transform_by_quat__dq(arm_local, parent_quat, arm_pos_grad)
                    # Accumulate into qpos.grad and parent's links_state.grad
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
                    # No parent: pos_init and quat_init are constants.
                    # link_pos = links_info.pos[I_l] + R(links_info.quat[I_l]) · motion_vel · displacement
                    # link_quat = links_info.quat[I_l]  (constant)
                    # Only displacement chains back.
                    quat_init = links_info.quat[I_l]
                    xaxis = gu.qd_transform_by_quat(axis, quat_init)
                    displacement_grad = (
                        xaxis[0] * arm_pos_grad[0] + xaxis[1] * arm_pos_grad[1] + xaxis[2] * arm_pos_grad[2]
                    )
                    rigid_global_info.qpos.grad[q_start, i_b] = (
                        rigid_global_info.qpos.grad[q_start, i_b] + displacement_grad
                    )
                # Per guide P8: zero consumed input .grad for this link.
                for j in qd.static(range(3)):
                    links_state.pos.grad[i_l, i_b][j] = 0.0
                for j in qd.static(range(4)):
                    links_state.quat.grad[i_l, i_b][j] = 0.0
            elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                # Per guide P9 (faithful replication): unimplemented joint
                # types must NOT silently skip. Raise via errno so the
                # solver-level Python check catches it after the kernel.
                errno[i_b] = errno[i_b] | array_class.ErrorCode.MANUAL_BW_UNIMPLEMENTED_JOINT_TYPE
            elif joint_type == gs.JOINT_TYPE.FIXED:
                # Forward (fixed joint applies no transform):
                #   pos  = parent_pos + R(parent_quat) · links_info.pos[I_l]
                #   quat = parent_quat ⊗ links_info.quat[I_l]
                #   (no qpos)
                parent_idx = links_info.parent_idx[I_l]
                arm_pos_grad = links_state.pos.grad[i_l, i_b]
                arm_quat_grad = links_state.quat.grad[i_l, i_b]
                if parent_idx != -1:
                    parent_quat = links_state.quat[parent_idx, i_b]
                    arm_local = links_info.pos[I_l]
                    parent_quat_grad_from_pos = d_transform_by_quat__dq(arm_local, parent_quat, arm_pos_grad)
                    parent_quat_grad_from_quat = d_quat_mul__drhs(links_info.quat[I_l], parent_quat, arm_quat_grad)
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
                # else: no parent + fixed = world-anchored constant; nothing chains back.
                # Per guide P8: zero consumed input .grad for this link.
                for j in qd.static(range(3)):
                    links_state.pos.grad[i_l, i_b][j] = 0.0
                for j in qd.static(range(4)):
                    links_state.quat.grad[i_l, i_b][j] = 0.0
            # All joint types covered: FREE / REVOLUTE / PRISMATIC / FIXED
            # produce the correct backward chain. SPHERICAL flips an errno
            # bit (caller must check + raise).


# =========================================================================
# Manual reverse for `func_update_force` (Step 5 sub-3).
#
# Forward (per-link, `genesis/.../forward_dynamics.py:1023`):
#   f1_ang, f1_vel = inertial_mul(cinr_pos, cinr_inertial, cinr_mass, cdd_v, cdd_a)
#   f2_ang, f2_vel = inertial_mul(cinr_pos, cinr_inertial, cinr_mass, cd_v, cd_a)
#   f3_ang, f3_vel = motion_cross_force(cd_ang, cd_vel, f2_ang, f2_vel)
#   cfrc_vel[i_l] = f1_vel + f3_vel + cfrc_applied_vel[i_l] + cfrc_coupling_vel[i_l]
#   cfrc_ang[i_l] = f1_ang + f3_ang + cfrc_applied_ang[i_l] + cfrc_coupling_ang[i_l]
# Then tree aggregation (leaf → root):
#   for i_l_ in range(n_links_in_entity):
#       i_l = link_end - 1 - i_l_
#       if parent_idx[i_l] != -1:
#           cfrc_vel[parent] += cfrc_vel[i_l]
#           cfrc_ang[parent] += cfrc_ang[i_l]
# Then clear coupling forces:
#   cfrc_coupling_{ang,vel} = 0
#
# Reverse (processed in statement-reverse order):
# 1. Clear coupling reverse — no-op for upstream grad (forward set to 0).
# 2. Tree aggregation reverse (root → leaf): cfrc[i_l].grad += cfrc[parent].grad
# 3. Per-link reverse of f1+f2+f3+applied+coupling chain.
#
# Scope: single-entity-batch (no hibernation). batch_links_info NOT supported (J4 has it False).
# =========================================================================


@qd.func
def d_motion_cross_force(m_ang, m_vel, f_ang, f_vel, ang_g, vel_g):
    """Reverse of motion_cross_force(m_ang, m_vel, f_ang, f_vel).

    Forward (geom.py:430):
        vel = m_ang × f_vel
        ang = m_ang × f_ang + m_vel × f_vel

    Chain rule (using c=a×b ⇒ a.g += b × c.g, b.g += c.g × a):
        m_ang.g += f_vel × vel.g  +  f_ang × ang.g
        m_vel.g += f_vel × ang.g
        f_ang.g += ang.g × m_ang
        f_vel.g += vel.g × m_ang  +  ang.g × m_vel

    Returns (m_ang_g, m_vel_g, f_ang_g, f_vel_g) — additive deltas to inputs.
    """
    return (
        f_vel.cross(vel_g) + f_ang.cross(ang_g),
        f_vel.cross(ang_g),
        ang_g.cross(m_ang),
        vel_g.cross(m_ang) + ang_g.cross(m_vel),
    )


@qd.func
def d_inertial_mul(pos, I_mat, mass, vel, ang, ang_g, vel_g):
    """Reverse of inertial_mul(pos, I, mass, vel, ang).

    Forward (geom.py:423):
        _ang = I @ ang + pos × vel
        _vel = mass * vel - pos × ang

    Chain rule:
        I.g     += outer(_ang.g, ang)
        ang.g   += I.T @ _ang.g + pos × _vel.g
        pos.g   += vel × _ang.g - (ang × _vel.g)
        vel.g   += mass * _vel.g + _ang.g × pos
        mass.g  += vel · _vel.g
    """
    pos_g = vel.cross(ang_g) - ang.cross(vel_g)
    I_g = ang_g.outer_product(ang)  # 3x3 matrix
    ang_out_g = I_mat.transpose() @ ang_g + pos.cross(vel_g)
    vel_out_g = mass * vel_g + ang_g.cross(pos)
    mass_g = vel.dot(vel_g)
    return pos_g, I_g, mass_g, vel_out_g, ang_out_g


@qd.kernel(fastcache=True)
def kernel_manual_update_force_bw(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    errno: qd.Tensor,
):
    """Manual backward replacement for `kernel_split_update_force.grad`.

    Bypasses the auto-AD reverse pass which (when run inside the monolithic
    `kernel_forward_dynamics_without_qacc.grad`) silently drops chain
    contributions to cinr_pos.grad / cd_v.grad / cd_a.grad — see
    `notes/diffrigid_handoff_j4_n2_fwd_velocity_suspect.md` and
    `/tmp/diag_fine_zero.py` (cinr_pos zero-out → max rel reduction by 2.66
    of baseline 6.97 on J4 N=2).

    Reads from links_state grads as inputs, writes the chain rule outputs
    into the input field grads (cd_v/cd_a/cdd_v/cdd_a/cinr_*/cfrc_applied_*/
    cfrc_coupling_*.grad). Per guide P8: consumes cfrc_vel/cfrc_ang.grad
    (zeros them after).

    Hibernation NOT supported — would require iteration over awake_entities/
    awake_links; raise errno bit if encountered.
    """
    qd.loop_config(
        name="manual_update_force_bw",
        serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL),
    )
    for i_b in range(links_state.pos.shape[1]):
        if qd.static(static_rigid_sim_config.use_hibernation):
            # P9: hibernation not implemented in this manual reverse. Flip errno.
            errno[i_b] = errno[i_b] | array_class.ErrorCode.MANUAL_BW_UNIMPLEMENTED_JOINT_TYPE
        else:
            # =================== Step 1: tree-aggregation reverse ===================
            # Forward: for i_l_ in range(n_links_in_entity):
            #            i_l = link_end - 1 - i_l_       (leaf → root)
            #            if parent != -1: cfrc[parent] += cfrc[i_l]
            # Reverse: iterate the SAME order so child's parent.grad
            # contribution is added to child BEFORE child is itself used as a
            # parent of its descendants. (Equivalent to iterating in root→leaf
            # order — since 'leaf to root' forward means 'root assigned last',
            # statement-reverse means 'root first'.)
            # We iterate i_l_ from 0 to n_links-1, which gives leaf→root order
            # (same as forward) — since each statement is independent in its
            # write (cfrc[parent] += cfrc[i_l]), reverse order of statements
            # is i_l_ from n-1 down to 0. That's root → leaf.
            # In statement-reverse: cfrc[i_l].grad += cfrc[parent].grad. For
            # a tree with root at i_l_root, we need to propagate root's grad
            # to its children, then each child propagates to its grandchildren.
            # ROOT → LEAF iteration achieves this. For J4 (chassis=0 root,
            # arm=1 child of chassis), iterating from arm(i_l_=0, i_l=1) up
            # to chassis(i_l_=1, i_l=0) in REVERSE means: i_l_=1 first (chassis,
            # parent=-1, skip), then i_l_=0 (arm, parent=0, cfrc[1].grad += cfrc[0].grad).
            for i_e in range(entities_info.n_links.shape[0]):
                n_in_e = entities_info.n_links[i_e]
                # Reverse the forward loop order: forward goes i_l_ = 0..n-1,
                # reverse goes i_l_ = n-1..0. Statement is "cfrc[parent] += cfrc[i_l]"
                # → reverse statement "cfrc[i_l].grad += cfrc[parent].grad".
                for i_l_rev in range(n_in_e):
                    # forward index i_l_ = n-1 - i_l_rev  ⇒ i_l = link_end - 1 - i_l_ = link_start + i_l_rev
                    i_l = entities_info.link_start[i_e] + i_l_rev
                    I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
                    i_p = links_info.parent_idx[I_l]
                    if i_p != -1:
                        for k in qd.static(range(3)):
                            links_state.cfrc_vel.grad[i_l, i_b][k] = (
                                links_state.cfrc_vel.grad[i_l, i_b][k] + links_state.cfrc_vel.grad[i_p, i_b][k]
                            )
                            links_state.cfrc_ang.grad[i_l, i_b][k] = (
                                links_state.cfrc_ang.grad[i_l, i_b][k] + links_state.cfrc_ang.grad[i_p, i_b][k]
                            )

            # =================== Step 2: per-link first-loop reverse ===================
            for i_e in range(entities_info.n_links.shape[0]):
                for i_l_ in range(entities_info.n_links[i_e]):
                    i_l = entities_info.link_start[i_e] + i_l_
                    I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l

                    cfrc_v_g = links_state.cfrc_vel.grad[i_l, i_b]
                    cfrc_a_g = links_state.cfrc_ang.grad[i_l, i_b]

                    # Forward additions distributing cfrc.grad evenly:
                    # f1_v_g = f3_v_g = cfrc_applied_v_g = cfrc_coupling_v_g = cfrc_v_g
                    # (same for ang)
                    # Accumulate into applied / coupling
                    for k in qd.static(range(3)):
                        links_state.cfrc_applied_vel.grad[i_l, i_b][k] = (
                            links_state.cfrc_applied_vel.grad[i_l, i_b][k] + cfrc_v_g[k]
                        )
                        links_state.cfrc_applied_ang.grad[i_l, i_b][k] = (
                            links_state.cfrc_applied_ang.grad[i_l, i_b][k] + cfrc_a_g[k]
                        )
                        links_state.cfrc_coupling_vel.grad[i_l, i_b][k] = (
                            links_state.cfrc_coupling_vel.grad[i_l, i_b][k] + cfrc_v_g[k]
                        )
                        links_state.cfrc_coupling_ang.grad[i_l, i_b][k] = (
                            links_state.cfrc_coupling_ang.grad[i_l, i_b][k] + cfrc_a_g[k]
                        )

                    # Read forward primal
                    cinr_pos = links_state.cinr_pos[i_l, i_b]
                    cinr_I = links_state.cinr_inertial[i_l, i_b]
                    cinr_m = links_state.cinr_mass[i_l, i_b]
                    cd_v = links_state.cd_vel[i_l, i_b]
                    cd_a = links_state.cd_ang[i_l, i_b]
                    cdd_v = links_state.cdd_vel[i_l, i_b]
                    cdd_a = links_state.cdd_ang[i_l, i_b]

                    # Reverse f3 = motion_cross_force(cd_a, cd_v, f2_ang, f2_vel)
                    # We need f2 = inertial_mul(cinr_pos, cinr_I, cinr_m, cd_v, cd_a)
                    f2_a = cinr_I @ cd_a + cinr_pos.cross(cd_v)
                    f2_v = cinr_m * cd_v - cinr_pos.cross(cd_a)
                    (m_ang_g, m_vel_g, f2_a_g, f2_v_g) = d_motion_cross_force(
                        cd_a, cd_v, f2_a, f2_v, cfrc_a_g, cfrc_v_g
                    )
                    # m_ang corresponds to cd_a; m_vel to cd_v
                    for k in qd.static(range(3)):
                        links_state.cd_ang.grad[i_l, i_b][k] = links_state.cd_ang.grad[i_l, i_b][k] + m_ang_g[k]
                        links_state.cd_vel.grad[i_l, i_b][k] = links_state.cd_vel.grad[i_l, i_b][k] + m_vel_g[k]

                    # Reverse f2 = inertial_mul(cinr_pos, cinr_I, cinr_m, cd_v, cd_a)
                    # Note: ang_g for this is f2_a_g, vel_g is f2_v_g
                    # Returns (pos_g, I_g, mass_g, vel_out_g, ang_out_g)
                    f2_pos_g, f2_I_g, f2_m_g, f2_cd_v_step_g, f2_cd_a_step_g = d_inertial_mul(
                        cinr_pos, cinr_I, cinr_m, cd_v, cd_a, f2_a_g, f2_v_g
                    )
                    for k in qd.static(range(3)):
                        links_state.cinr_pos.grad[i_l, i_b][k] = links_state.cinr_pos.grad[i_l, i_b][k] + f2_pos_g[k]
                        links_state.cd_vel.grad[i_l, i_b][k] = links_state.cd_vel.grad[i_l, i_b][k] + f2_cd_v_step_g[k]
                        links_state.cd_ang.grad[i_l, i_b][k] = links_state.cd_ang.grad[i_l, i_b][k] + f2_cd_a_step_g[k]
                    for r in qd.static(range(3)):
                        for c in qd.static(range(3)):
                            links_state.cinr_inertial.grad[i_l, i_b][r, c] = (
                                links_state.cinr_inertial.grad[i_l, i_b][r, c] + f2_I_g[r, c]
                            )
                    links_state.cinr_mass.grad[i_l, i_b] = links_state.cinr_mass.grad[i_l, i_b] + f2_m_g

                    # Reverse f1 = inertial_mul(cinr_pos, cinr_I, cinr_m, cdd_v, cdd_a)
                    # ang_g = cfrc_a_g, vel_g = cfrc_v_g (from f1_*_g)
                    f1_pos_g, f1_I_g, f1_m_g, f1_cdd_v_g, f1_cdd_a_g = d_inertial_mul(
                        cinr_pos, cinr_I, cinr_m, cdd_v, cdd_a, cfrc_a_g, cfrc_v_g
                    )
                    for k in qd.static(range(3)):
                        links_state.cinr_pos.grad[i_l, i_b][k] = links_state.cinr_pos.grad[i_l, i_b][k] + f1_pos_g[k]
                        links_state.cdd_vel.grad[i_l, i_b][k] = links_state.cdd_vel.grad[i_l, i_b][k] + f1_cdd_v_g[k]
                        links_state.cdd_ang.grad[i_l, i_b][k] = links_state.cdd_ang.grad[i_l, i_b][k] + f1_cdd_a_g[k]
                    for r in qd.static(range(3)):
                        for c in qd.static(range(3)):
                            links_state.cinr_inertial.grad[i_l, i_b][r, c] = (
                                links_state.cinr_inertial.grad[i_l, i_b][r, c] + f1_I_g[r, c]
                            )
                    links_state.cinr_mass.grad[i_l, i_b] = links_state.cinr_mass.grad[i_l, i_b] + f1_m_g

                    # Per guide P8: zero consumed cfrc.grad
                    for k in qd.static(range(3)):
                        links_state.cfrc_vel.grad[i_l, i_b][k] = 0.0
                        links_state.cfrc_ang.grad[i_l, i_b][k] = 0.0


# =========================================================================
# Manual reverses for compute_mass_matrix sub-blocks (Step 5 sub-6).
#
# Hypothesis (from stage dump): the auto-AD reverse of kernel_mm_assemble or
# kernel_mm_crb_aggregate may silently drop or otherwise miscompute the
# mass_mat → f → cdof and crb tree chain rules. Replace them with explicit
# manual chain rule kernels to see whether J4 N=2 rel err changes.
# =========================================================================


@qd.kernel(fastcache=True)
def kernel_manual_mm_assemble_bw(
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    errno: qd.Tensor,
):
    """Manual reverse for func_mm_assemble.

    Forward (per entity, per (i_d, j_d) ∈ [dof_start, dof_end)²):
      val = f_ang[i_d].dot(cdof_ang[j_d]) + f_vel[i_d].dot(cdof_vel[j_d])
      mass_mat[i_d, j_d] = val * mass_parent_mask[i_d, j_d]
    Then symmetric copy for upper triangle:
      for i_d:
        for j_d in (i_d+1, dof_end):
          mass_mat[i_d, j_d] = mass_mat[j_d, i_d]   ← OVERWRITES the dot-product result

    Reverse (statement-reverse order):
      1) Symmetric copy reverse:
         mass_mat[j_d, i_d].grad += mass_mat[i_d, j_d].grad
         mass_mat[i_d, j_d].grad = 0     (P8 — the dot-product output was overwritten by symm copy)
      2) f.dot(cdof) loop reverse:
         val_grad = mass_mat[i_d, j_d].grad * mass_parent_mask[i_d, j_d]
         f_ang[i_d].grad   += val_grad * cdof_ang[j_d]
         cdof_ang[j_d].grad += val_grad * f_ang[i_d]
         f_vel[i_d].grad   += val_grad * cdof_vel[j_d]
         cdof_vel[j_d].grad += val_grad * f_vel[i_d]
      3) mass_mat[i_d, j_d].grad = 0 (P8, consumed)
    """
    qd.loop_config(
        name="manual_mm_assemble_bw",
        serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL),
    )
    for i_e, i_b in qd.ndrange(entities_info.n_links.shape[0], dofs_state.f_ang.shape[1]):
        if qd.static(static_rigid_sim_config.use_hibernation):
            errno[i_b] = errno[i_b] | array_class.ErrorCode.MANUAL_BW_UNIMPLEMENTED_JOINT_TYPE
        else:
            d_start = entities_info.dof_start[i_e]
            d_end = entities_info.dof_end[i_e]

            # =========== Step 1: reverse symmetric copy =================
            # Forward iterated i_d ∈ [d_start, d_end), j_d ∈ (i_d+1, d_end).
            # Reverse iterates same statements in reverse order.
            for i_d in range(d_start, d_end):
                for j_d in range(i_d + 1, d_end):
                    # forward: mass_mat[i_d, j_d] = mass_mat[j_d, i_d]
                    # reverse: mass_mat[j_d, i_d].grad += mass_mat[i_d, j_d].grad
                    rigid_global_info.mass_mat.grad[j_d, i_d, i_b] = (
                        rigid_global_info.mass_mat.grad[j_d, i_d, i_b] + rigid_global_info.mass_mat.grad[i_d, j_d, i_b]
                    )
                    rigid_global_info.mass_mat.grad[i_d, j_d, i_b] = 0.0

            # =========== Step 2: reverse f.dot(cdof) chain ===============
            for i_d, j_d in qd.ndrange((d_start, d_end), (d_start, d_end)):
                val_grad = rigid_global_info.mass_mat.grad[i_d, j_d, i_b] * rigid_global_info.mass_parent_mask[i_d, j_d]
                f_ang_i = dofs_state.f_ang[i_d, i_b]
                f_vel_i = dofs_state.f_vel[i_d, i_b]
                cdof_ang_j = dofs_state.cdof_ang[j_d, i_b]
                cdof_vel_j = dofs_state.cdof_vel[j_d, i_b]
                for k in qd.static(range(3)):
                    dofs_state.f_ang.grad[i_d, i_b][k] = dofs_state.f_ang.grad[i_d, i_b][k] + val_grad * cdof_ang_j[k]
                    dofs_state.cdof_ang.grad[j_d, i_b][k] = (
                        dofs_state.cdof_ang.grad[j_d, i_b][k] + val_grad * f_ang_i[k]
                    )
                    dofs_state.f_vel.grad[i_d, i_b][k] = dofs_state.f_vel.grad[i_d, i_b][k] + val_grad * cdof_vel_j[k]
                    dofs_state.cdof_vel.grad[j_d, i_b][k] = (
                        dofs_state.cdof_vel.grad[j_d, i_b][k] + val_grad * f_vel_i[k]
                    )

            # =========== Step 3: P8 — zero consumed mass_mat.grad ==============
            for i_d, j_d in qd.ndrange((d_start, d_end), (d_start, d_end)):
                rigid_global_info.mass_mat.grad[i_d, j_d, i_b] = 0.0


@qd.kernel(fastcache=True)
def kernel_manual_mm_crb_aggregate_bw(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    errno: qd.Tensor,
):
    """Manual reverse for func_mm_crb_aggregate.

    Forward (per entity, iterating i from 0 to n_links-1):
      i_l = link_end - 1 - i              (leaf → root)
      if parent != -1:
        crb_inertial[parent] += crb_inertial[i_l]
        crb_mass[parent]     += crb_mass[i_l]
        crb_pos[parent]      += crb_pos[i_l]
        crb_quat[parent]     += crb_quat[i_l]

    Reverse: each `X[parent] += X[i_l]` statement reverses to
    `X[i_l].grad += X[parent].grad`. Iterate root → leaf (= REVERSE forward
    order) so a parent's accumulated grad propagates to its children before
    those children are themselves used as parents.

    Iteration order: i in [n_links-1, n_links-2, ..., 0] in forward →
    i in [0, 1, ..., n_links-1] in reverse → i_l from link_start up to
    link_end-1.
    """
    qd.loop_config(
        name="manual_mm_crb_aggregate_bw",
        serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL),
    )
    for i_e, i_b in qd.ndrange(entities_info.n_links.shape[0], links_state.pos.shape[1]):
        if qd.static(static_rigid_sim_config.use_hibernation):
            errno[i_b] = errno[i_b] | array_class.ErrorCode.MANUAL_BW_UNIMPLEMENTED_JOINT_TYPE
        else:
            n_in_e = entities_info.n_links[i_e]
            for i_l_offset in range(n_in_e):
                i_l = entities_info.link_start[i_e] + i_l_offset
                I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
                i_p = links_info.parent_idx[I_l]
                if i_p != -1:
                    for k in qd.static(range(3)):
                        links_state.crb_pos.grad[i_l, i_b][k] = (
                            links_state.crb_pos.grad[i_l, i_b][k] + links_state.crb_pos.grad[i_p, i_b][k]
                        )
                    for k in qd.static(range(4)):
                        links_state.crb_quat.grad[i_l, i_b][k] = (
                            links_state.crb_quat.grad[i_l, i_b][k] + links_state.crb_quat.grad[i_p, i_b][k]
                        )
                    for r in qd.static(range(3)):
                        for c in qd.static(range(3)):
                            links_state.crb_inertial.grad[i_l, i_b][r, c] = (
                                links_state.crb_inertial.grad[i_l, i_b][r, c]
                                + links_state.crb_inertial.grad[i_p, i_b][r, c]
                            )
                    links_state.crb_mass.grad[i_l, i_b] = (
                        links_state.crb_mass.grad[i_l, i_b] + links_state.crb_mass.grad[i_p, i_b]
                    )
