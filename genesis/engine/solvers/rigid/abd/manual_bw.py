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
    joints_state: array_class.JointsState,
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
                # Also (FREE branch in forward_kinematics_entity_one_link line 720-728):
                #   xanchor[free_joint] = qpos[q_start:q_start+3]
                #   xaxis[free_joint] = [0, 0, 1]  (const)
                # Reverse: qpos.grad[0:3] also gets xanchor.grad contribution.
                pos_grad = links_state.pos.grad[i_l, i_b]
                quat_grad = links_state.quat.grad[i_l, i_b]
                xanchor_grad = joints_state.xanchor.grad[i_j, i_b]
                for j in qd.static(range(3)):
                    rigid_global_info.qpos.grad[q_start + j, i_b] = (
                        rigid_global_info.qpos.grad[q_start + j, i_b]
                        + pos_grad[j]
                        + xanchor_grad[j]  # xanchor[free_joint] = qpos[0:3]
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
                    joints_state.xanchor.grad[i_j, i_b][j] = 0.0
                    joints_state.xaxis.grad[i_j, i_b][j] = 0.0  # xaxis = const, no chain
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

                    # arm_quat = parent_quat ⊗ qloc (lhs=parent_quat, rhs=qloc).
                    # Forward is `qd_transform_quat_by_quat(qloc, parent_quat)` =
                    # `qd_quat_mul(parent_quat, qloc)` (see geom.py:281-290).
                    parent_quat_grad_from_quat = d_quat_mul__dlhs(parent_quat, qloc, arm_quat_grad)
                    qloc_grad = d_quat_mul__drhs(parent_quat, qloc, arm_quat_grad)

                    # NEW: xanchor / xaxis reverse chain.
                    # Forward (forward_kinematics_entity_one_link line 720-742, REVOLUTE/PRISMATIC):
                    #   pos_curr = parent_pos + R(parent_quat) · arm_local   (= arm_base_pos)
                    #   quat_curr = parent_quat                              (= arm_base_quat, identity links_info.quat)
                    #   xanchor[i_j] = R(quat_curr) · joints_info.pos[i_j] + pos_curr
                    #   xaxis[i_j]   = R(quat_curr) · axis
                    # Reverse:
                    #   pos_curr.grad   += xanchor.grad
                    #   quat_curr.grad  += d_transform_by_quat__dq(joints_info.pos, quat_curr, xanchor.grad)
                    #                    + d_transform_by_quat__dq(axis,            quat_curr, xaxis.grad)
                    # Then pos_curr.grad / quat_curr.grad chain to parent_pos.grad / parent_quat.grad
                    # the same way arm_pos.grad does (parent_pos += pos_curr.grad,
                    # parent_quat += d_transform_by_quat__dq(arm_local, parent_quat, pos_curr.grad)).
                    xanchor_grad = joints_state.xanchor.grad[i_j, i_b]
                    xaxis_grad = joints_state.xaxis.grad[i_j, i_b]
                    joint_pos_off = joints_info.pos[I_j]
                    parent_quat_grad_from_xanchor_via_quat = d_transform_by_quat__dq(
                        joint_pos_off, parent_quat, xanchor_grad
                    )
                    parent_quat_grad_from_xaxis = d_transform_by_quat__dq(axis, parent_quat, xaxis_grad)
                    # pos_curr.grad = xanchor.grad → chain to parent_pos / parent_quat via arm_local
                    parent_quat_grad_from_xanchor_via_pos = d_transform_by_quat__dq(
                        arm_local, parent_quat, xanchor_grad
                    )

                    # qloc = rotvec_to_quat(axis · angle)
                    rotvec_grad = d_rotvec_to_quat__drotvec(rotvec, rigid_global_info.EPS[None], qloc_grad)
                    angle_grad = axis[0] * rotvec_grad[0] + axis[1] * rotvec_grad[1] + axis[2] * rotvec_grad[2]
                    rigid_global_info.qpos.grad[q_start, i_b] = rigid_global_info.qpos.grad[q_start, i_b] + angle_grad

                    for j in qd.static(range(3)):
                        links_state.pos.grad[parent_idx, i_b][j] = (
                            links_state.pos.grad[parent_idx, i_b][j]
                            + arm_pos_grad[j]
                            + xanchor_grad[j]  # pos_curr.grad chain
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
                    # P8: consume xanchor / xaxis .grad
                    for j in qd.static(range(3)):
                        joints_state.xanchor.grad[i_j, i_b][j] = 0.0
                        joints_state.xaxis.grad[i_j, i_b][j] = 0.0
                else:
                    # No parent: pos_init/quat_init are link-level constants.
                    # arm_pos.grad doesn't chain anywhere (constant).
                    # arm_quat = quat_init ⊗ qloc (forward = quat_mul(quat_init, qloc))
                    # ⇒ qloc_grad = d_quat_mul__drhs(quat_init, qloc, og).
                    quat_init = links_info.quat[I_l]
                    qloc_grad = d_quat_mul__drhs(quat_init, qloc, arm_quat_grad)
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
                    # quat_init = parent_quat ⊗ links_info.quat[I_l]
                    # ⇒ parent_quat.grad += d_quat_mul__dlhs(parent_quat, links_info.quat[I_l], og)
                    parent_quat_grad_from_quat = d_quat_mul__dlhs(
                        parent_quat, links_info.quat[I_l], quat_init_grad_total
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
                    # arm_quat = parent_quat ⊗ links_info.quat[I_l]
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
                # else: no parent + fixed = world-anchored constant; nothing chains back.
                # Per guide P8: zero consumed input .grad for this link.
                for j in qd.static(range(3)):
                    links_state.pos.grad[i_l, i_b][j] = 0.0
                for j in qd.static(range(4)):
                    links_state.quat.grad[i_l, i_b][j] = 0.0
            # All joint types covered: FREE / REVOLUTE / PRISMATIC / FIXED
            # produce the correct backward chain. SPHERICAL flips an errno
            # bit (caller must check + raise).


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
    Same per-link body as `kernel_manual_uc_bw_one_link` but iterates the
    links of each entity from leaf to root inside one kernel launch so a
    child's `parent.{pos,quat}.grad` write completes before the parent's
    own iteration consumes it.

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
# Manual reverse for `func_j_pos_quat_propagation_entity` (Phase 5 of
# `func_COM_links_entity` extracted).
#
# Forward (per entity, per link i_l with n_dofs > 0):
#   i_p = parent_idx[i_l]
#   p_pos, p_quat = (pos[i_p], quat[i_p]) if i_p != -1 else (0, identity)
#   if joint_type == FREE or (is_fixed and i_p == -1):
#       j_pos[i_l]  = pos[i_l]
#       j_quat[i_l] = quat[i_l]
#   else:
#       (j_pos_bw[i_l, 0], j_quat_bw[i_l, 0]) = transform_pq_by_tq(
#           links_info.pos[I_l], links_info.quat[I_l], p_pos, p_quat
#       )
#       for i_j_ in range(n_joints):
#           (j_pos_bw[i_l, next], j_quat_bw[i_l, next]) = transform_pq_by_tq(
#               joints_info.pos[i_j], identity, j_pos_bw[i_l, curr], j_quat_bw[i_l, curr]
#           )
#       j_pos[i_l]  = j_pos_bw[i_l, n_joints]
#       j_quat[i_l] = j_quat_bw[i_l, n_joints]
#
# Bypasses Quadrants AD silent drop on cross-link parent.pos / parent.quat
# read into local `p_pos` / `p_quat` (J4/J5 standalone FD shows pos/quat.grad
# rel error ~17-86% via auto-AD vs FP64 floor for other inputs).
#
# Joint type branches:
#   - FREE / (is_fixed and root): identity copy reverse
#   - else (REVOLUTE/PRISMATIC/SPHERICAL/FIXED-with-parent): joint chain reverse
# Both branches handled — no errno needed.
# =========================================================================


@qd.kernel(fastcache=True)
def kernel_manual_COM_links_phase5_bw(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_info: array_class.JointsInfo,
    entities_info: array_class.EntitiesInfo,
    static_rigid_sim_config: qd.template(),
    errno: qd.Tensor,
):
    """Manual reverse of `func_j_pos_quat_propagation_entity` (Phase 5).

    Inputs (read .grad seeds):
      - links_state.j_pos.grad[i_l, i_b], links_state.j_quat.grad[i_l, i_b]
      - links_state.j_pos_bw.grad[i_l, k, i_b], links_state.j_quat_bw.grad[i_l, k, i_b]
        for k in 0..n_joints[i_l]

    Outputs (accumulated .grad):
      - links_state.pos.grad[i_l, i_b], links_state.quat.grad[i_l, i_b]  (FREE / root-fixed branch)
      - links_state.pos.grad[i_p, i_b], links_state.quat.grad[i_p, i_b]  (joint chain branch, cross-link)

    P8 (consume convention): after each statement-reverse, consume the
    upstream .grad by zeroing it.

    Hibernation NOT supported — flips errno bit. SPHERICAL is supported here
    (Phase 5 forward treats it the same as other joint chains).
    """
    qd.loop_config(
        name="manual_COM_links_phase5_bw",
        serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL),
    )
    for i_e, i_b in qd.ndrange(entities_info.n_links.shape[0], links_state.pos.shape[1]):
        if qd.static(static_rigid_sim_config.use_hibernation):
            errno[i_b] = errno[i_b] | array_class.ErrorCode.MANUAL_BW_UNIMPLEMENTED_JOINT_TYPE
        else:
            for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
                I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l

                if links_info.n_dofs[I_l] > 0:
                    i_p = links_info.parent_idx[I_l]
                    _i_j = links_info.joint_start[I_l]
                    _I_j = [_i_j, i_b] if qd.static(static_rigid_sim_config.batch_joints_info) else _i_j
                    joint_type = joints_info.type[_I_j]

                    if joint_type == gs.JOINT_TYPE.FREE or (links_info.is_fixed[I_l] and i_p == -1):
                        # Forward: j_pos[i_l] = pos[i_l], j_quat[i_l] = quat[i_l].
                        # Reverse: pos.grad += j_pos.grad, quat.grad += j_quat.grad.
                        for k in qd.static(range(3)):
                            links_state.pos.grad[i_l, i_b][k] = (
                                links_state.pos.grad[i_l, i_b][k] + links_state.j_pos.grad[i_l, i_b][k]
                            )
                        for k in qd.static(range(4)):
                            links_state.quat.grad[i_l, i_b][k] = (
                                links_state.quat.grad[i_l, i_b][k] + links_state.j_quat.grad[i_l, i_b][k]
                            )
                        # P8: consume j_pos / j_quat .grad
                        for k in qd.static(range(3)):
                            links_state.j_pos.grad[i_l, i_b][k] = 0.0
                        for k in qd.static(range(4)):
                            links_state.j_quat.grad[i_l, i_b][k] = 0.0
                    else:
                        # Joint chain branch.
                        n_joints = links_info.joint_end[I_l] - links_info.joint_start[I_l]

                        # KEY OBSERVATION: j_quat_bw[i_l, k, i_b] is the SAME
                        # for all k = 0..n_joints because each joint chain
                        # step multiplies by `identity_quat` on the right
                        # (forward: new_quat = qd_quat_mul(j_quat_bw[k],
                        # identity) = j_quat_bw[k]). So we can self-compute
                        # the primal without reading the stale phase-5
                        # forward output (which lets us skip phase 5 forward
                        # entirely and avoid Quadrants AD ad-stack
                        # mismatch).
                        #
                        #   j_quat_bw[k] = qd_quat_mul(p_quat, links_info.quat[I_l])
                        #   where p_quat is parent's quat (or identity if root)
                        p_quat_for_jbw = gu.qd_identity_quat()
                        if i_p != -1:
                            p_quat_for_jbw = links_state.quat[i_p, i_b]
                        j_quat_bw_const = gu.qd_quat_mul(p_quat_for_jbw, links_info.quat[I_l])

                        # ─── Reverse statement N: j_pos[i_l] = j_pos_bw[i_l, n_joints],
                        # j_quat[i_l] = j_quat_bw[i_l, n_joints].
                        for k in qd.static(range(3)):
                            links_state.j_pos_bw.grad[i_l, n_joints, i_b][k] = (
                                links_state.j_pos_bw.grad[i_l, n_joints, i_b][k] + links_state.j_pos.grad[i_l, i_b][k]
                            )
                        for k in qd.static(range(4)):
                            links_state.j_quat_bw.grad[i_l, n_joints, i_b][k] = (
                                links_state.j_quat_bw.grad[i_l, n_joints, i_b][k] + links_state.j_quat.grad[i_l, i_b][k]
                            )
                        # P8 consume j_pos / j_quat
                        for k in qd.static(range(3)):
                            links_state.j_pos.grad[i_l, i_b][k] = 0.0
                        for k in qd.static(range(4)):
                            links_state.j_quat.grad[i_l, i_b][k] = 0.0

                        # ─── Reverse joint chain in REVERSE forward order
                        # (i_j_ from n_joints-1 down to 0). Forward statement:
                        #   (j_pos_bw[next], j_quat_bw[next]) = transform_pq_by_tq(
                        #       joints_info.pos[i_j], identity, j_pos_bw[curr], j_quat_bw[curr]
                        #   )
                        # = (j_pos_bw[curr] + R(j_quat_bw[curr]) · joints_info.pos[i_j],
                        #    qd_quat_mul(j_quat_bw[curr], identity) = j_quat_bw[curr])
                        for i_j_rev in range(n_joints):
                            i_j_ = n_joints - 1 - i_j_rev
                            i_j = i_j_ + links_info.joint_start[I_l]
                            I_j = [i_j, i_b] if qd.static(static_rigid_sim_config.batch_joints_info) else i_j
                            curr_idx = i_j_
                            next_idx = i_j_ + 1

                            # Use self-computed primal (see KEY OBSERVATION above) —
                            # avoids dependency on stale links_state.j_quat_bw.
                            j_quat_bw_curr = j_quat_bw_const
                            j_pos_bw_next_grad = links_state.j_pos_bw.grad[i_l, next_idx, i_b]
                            j_quat_bw_next_grad = links_state.j_quat_bw.grad[i_l, next_idx, i_b]
                            joint_pos_off = joints_info.pos[I_j]

                            # j_quat_bw[curr].grad += d_transform_by_quat__dq(
                            #     joint_pos_off, j_quat_bw_curr, j_pos_bw_next.grad
                            # )  (from new_pos = R(j_quat_bw_curr) · joint_pos_off chain)
                            quat_grad_from_pos = d_transform_by_quat__dq(
                                joint_pos_off, j_quat_bw_curr, j_pos_bw_next_grad
                            )

                            for k in qd.static(range(3)):
                                # j_pos_bw[curr].grad += j_pos_bw[next].grad (identity part)
                                links_state.j_pos_bw.grad[i_l, curr_idx, i_b][k] = (
                                    links_state.j_pos_bw.grad[i_l, curr_idx, i_b][k] + j_pos_bw_next_grad[k]
                                )
                            for k in qd.static(range(4)):
                                # j_quat_bw[curr].grad += (
                                #     d_transform_by_quat__dq(...) + j_quat_bw[next].grad
                                # )
                                links_state.j_quat_bw.grad[i_l, curr_idx, i_b][k] = (
                                    links_state.j_quat_bw.grad[i_l, curr_idx, i_b][k]
                                    + quat_grad_from_pos[k]
                                    + j_quat_bw_next_grad[k]
                                )
                            # P8: consume j_pos_bw[next] / j_quat_bw[next] .grad
                            for k in qd.static(range(3)):
                                links_state.j_pos_bw.grad[i_l, next_idx, i_b][k] = 0.0
                            for k in qd.static(range(4)):
                                links_state.j_quat_bw.grad[i_l, next_idx, i_b][k] = 0.0

                        # ─── Reverse statement A: (j_pos_bw[0], j_quat_bw[0]) =
                        #   transform_pq_by_tq(l_info.pos, l_info.quat, p_pos, p_quat)
                        # = (p_pos + R(p_quat) · l_info.pos, qd_quat_mul(p_quat, l_info.quat))
                        l_info_pos = links_info.pos[I_l]
                        l_info_quat = links_info.quat[I_l]
                        j_pos_bw_0_grad = links_state.j_pos_bw.grad[i_l, 0, i_b]
                        j_quat_bw_0_grad = links_state.j_quat_bw.grad[i_l, 0, i_b]

                        if i_p != -1:
                            p_quat = links_state.quat[i_p, i_b]
                            # From j_pos_bw[0]: p_pos.grad += j_pos_bw_0.grad
                            #                   p_quat.grad += d_transform_by_quat__dq(l_info.pos, p_quat, j_pos_bw_0.grad)
                            p_quat_grad_from_pos = d_transform_by_quat__dq(l_info_pos, p_quat, j_pos_bw_0_grad)
                            # From j_quat_bw[0]: p_quat.grad += d_quat_mul__dlhs(p_quat, l_info.quat, j_quat_bw_0.grad)
                            p_quat_grad_from_quat = d_quat_mul__dlhs(p_quat, l_info_quat, j_quat_bw_0_grad)

                            for k in qd.static(range(3)):
                                links_state.pos.grad[i_p, i_b][k] = (
                                    links_state.pos.grad[i_p, i_b][k] + j_pos_bw_0_grad[k]
                                )
                            for k in qd.static(range(4)):
                                links_state.quat.grad[i_p, i_b][k] = (
                                    links_state.quat.grad[i_p, i_b][k]
                                    + p_quat_grad_from_pos[k]
                                    + p_quat_grad_from_quat[k]
                                )
                        # else: p_pos / p_quat are constants (zero / identity); no chain.

                        # P8: consume j_pos_bw[0] / j_quat_bw[0] .grad
                        for k in qd.static(range(3)):
                            links_state.j_pos_bw.grad[i_l, 0, i_b][k] = 0.0
                        for k in qd.static(range(4)):
                            links_state.j_quat_bw.grad[i_l, 0, i_b][k] = 0.0


# =========================================================================
# Helpers for the full manual `kernel_manual_COM_links_bw` (all phases).
# =========================================================================


@qd.func
def d_qd_quat_to_R__dquat(quat, R_grad):
    """Reverse of `R = qd_quat_to_R(quat, eps)` (assumes unit quat input).

    qd_quat_to_R first normalizes the quat (q_n = quat / norm(quat)) then
    applies the non-normalized R-formula on q_n. The chain rule must
    account for both:
      d(R[i,j])/d(quat[α]) = d(R[i,j])/d(q_n[β]) · d(q_n[β])/d(quat[α])

    For unit input quat (norm = 1):
      d(q_n[β])/d(quat[α]) = δ_{αβ} - quat[α] · quat[β]
                            (projection orthogonal to quat)

    So quat_grad = raw - quat · (raw · quat),
    where raw is the non-normalized chain `sum_j d_transform_by_quat__dq(
    e_j, quat, R_grad[:, j])` (since qd_transform_by_quat is the
    non-normalized R-formula).
    """
    col0_g = qd.Vector([R_grad[0, 0], R_grad[1, 0], R_grad[2, 0]], dt=gs.qd_float)
    col1_g = qd.Vector([R_grad[0, 1], R_grad[1, 1], R_grad[2, 1]], dt=gs.qd_float)
    col2_g = qd.Vector([R_grad[0, 2], R_grad[1, 2], R_grad[2, 2]], dt=gs.qd_float)
    e_x = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
    e_y = qd.Vector([0.0, 1.0, 0.0], dt=gs.qd_float)
    e_z = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)
    raw = (
        d_transform_by_quat__dq(e_x, quat, col0_g)
        + d_transform_by_quat__dq(e_y, quat, col1_g)
        + d_transform_by_quat__dq(e_z, quat, col2_g)
    )
    raw_dot_quat = raw[0] * quat[0] + raw[1] * quat[1] + raw[2] * quat[2] + raw[3] * quat[3]
    return qd.Vector(
        [
            raw[0] - quat[0] * raw_dot_quat,
            raw[1] - quat[1] * raw_dot_quat,
            raw[2] - quat[2] * raw_dot_quat,
            raw[3] - quat[3] * raw_dot_quat,
        ],
        dt=gs.qd_float,
    )


@qd.func
def d_qd_transform_inertia_by_trans_quat(
    inertial_i,
    i_mass,
    trans,
    quat,
    eps,
    out_i_grad,
    out_trans_grad,
    out_quat_grad,
    out_mass_grad,
):
    """Reverse of `qd_transform_inertia_by_trans_quat` (geom.py:373).

    Forward:
        hhT[i,j] = (||trans||² · I - trans ⊗ trans)[i,j]   # depends on trans
        R = qd_quat_to_R(quat, eps)                        # depends on quat
        out_i      = R @ inertial_i @ R.T + hhT * i_mass
        out_trans  = trans * i_mass
        out_quat   = quat
        out_mass   = i_mass

    Returns (i_mass_grad, trans_grad, quat_grad). inertial_i is treated as
    constant (link-level config). Caller may ignore inertial_i_grad.
    """
    R = gu.qd_quat_to_R(quat, eps)

    # ─── i_mass.grad: 3 sources
    # (1) out_mass = i_mass  →  +out_mass_grad
    # (2) out_trans = trans · i_mass  →  +trans · out_trans_grad
    # (3) out_i has term hhT · i_mass  →  +<hhT, out_i_grad>
    tx = trans[0]
    ty = trans[1]
    tz = trans[2]
    txx = tx * tx
    tyy = ty * ty
    tzz = tz * tz
    txy = tx * ty
    txz = tx * tz
    tyz = ty * tz
    hhT_dot_outi = (
        (tyy + tzz) * out_i_grad[0, 0]
        + (-txy) * out_i_grad[0, 1]
        + (-txz) * out_i_grad[0, 2]
        + (-txy) * out_i_grad[1, 0]
        + (txx + tzz) * out_i_grad[1, 1]
        + (-tyz) * out_i_grad[1, 2]
        + (-txz) * out_i_grad[2, 0]
        + (-tyz) * out_i_grad[2, 1]
        + (txx + tyy) * out_i_grad[2, 2]
    )
    i_mass_grad = (
        out_mass_grad + tx * out_trans_grad[0] + ty * out_trans_grad[1] + tz * out_trans_grad[2] + hhT_dot_outi
    )

    # ─── trans.grad: 2 sources
    # (1) out_trans = trans · i_mass  →  + out_trans_grad · i_mass (component-wise scale)
    # (2) hhT chain: trans.grad[k] += i_mass · (2 trans[k] · tr(out_i_grad)
    #                                         - ((out_i_grad + out_i_grad.T) @ trans)[k])
    trace_oig = out_i_grad[0, 0] + out_i_grad[1, 1] + out_i_grad[2, 2]
    # Compute ((out_i_grad + out_i_grad.T) @ trans) component-wise
    sym_at_trans_0 = (
        2.0 * out_i_grad[0, 0] * tx
        + (out_i_grad[0, 1] + out_i_grad[1, 0]) * ty
        + (out_i_grad[0, 2] + out_i_grad[2, 0]) * tz
    )
    sym_at_trans_1 = (
        (out_i_grad[1, 0] + out_i_grad[0, 1]) * tx
        + 2.0 * out_i_grad[1, 1] * ty
        + (out_i_grad[1, 2] + out_i_grad[2, 1]) * tz
    )
    sym_at_trans_2 = (
        (out_i_grad[2, 0] + out_i_grad[0, 2]) * tx
        + (out_i_grad[2, 1] + out_i_grad[1, 2]) * ty
        + 2.0 * out_i_grad[2, 2] * tz
    )
    trans_grad = qd.Vector(
        [
            i_mass * out_trans_grad[0] + i_mass * (2.0 * tx * trace_oig - sym_at_trans_0),
            i_mass * out_trans_grad[1] + i_mass * (2.0 * ty * trace_oig - sym_at_trans_1),
            i_mass * out_trans_grad[2] + i_mass * (2.0 * tz * trace_oig - sym_at_trans_2),
        ],
        dt=gs.qd_float,
    )

    # ─── quat.grad: 2 sources
    # (1) out_quat = quat  →  + out_quat_grad
    # (2) R chain via out_i = R @ inertial_i @ R.T:
    #     R_grad = (out_i_grad + out_i_grad.T) @ R @ inertial_i
    #     quat_grad_from_R = d_qd_quat_to_R__dquat(quat, R_grad)
    sym_oig = out_i_grad + out_i_grad.transpose()
    R_inertial = R @ inertial_i
    R_grad = sym_oig @ R_inertial
    quat_grad_from_R = d_qd_quat_to_R__dquat(quat, R_grad)
    quat_grad = qd.Vector(
        [
            out_quat_grad[0] + quat_grad_from_R[0],
            out_quat_grad[1] + quat_grad_from_R[1],
            out_quat_grad[2] + quat_grad_from_R[2],
            out_quat_grad[3] + quat_grad_from_R[3],
        ],
        dt=gs.qd_float,
    )

    return i_mass_grad, trans_grad, quat_grad


@qd.func
def d_qd_transform_pos_quat_by_trans_quat(
    pos,
    quat,
    t_trans,
    t_quat,
    new_pos_grad,
    new_quat_grad,
):
    """Reverse of `qd_transform_pos_quat_by_trans_quat(pos, quat, t_trans, t_quat)`.

    Forward (geom.py:366):
        new_pos  = t_trans + R(t_quat) · pos
        new_quat = qd_quat_mul(t_quat, quat)

    Returns (pos_grad, quat_grad, t_trans_grad, t_quat_grad).
    """
    # t_trans contribution: identity
    t_trans_grad = new_pos_grad
    # pos contribution from new_pos: R(t_quat).T · new_pos_grad
    pos_grad = gu.qd_inv_transform_by_quat(new_pos_grad, t_quat)
    # t_quat contribution from new_pos: d_transform_by_quat__dq(pos, t_quat, new_pos_grad)
    t_quat_grad_from_pos = d_transform_by_quat__dq(pos, t_quat, new_pos_grad)
    # t_quat contribution from new_quat: d_quat_mul__dlhs(t_quat, quat, new_quat_grad)
    t_quat_grad_from_quat = d_quat_mul__dlhs(t_quat, quat, new_quat_grad)
    # quat contribution from new_quat: d_quat_mul__drhs(t_quat, quat, new_quat_grad)
    quat_grad = d_quat_mul__drhs(t_quat, quat, new_quat_grad)
    t_quat_grad = qd.Vector(
        [
            t_quat_grad_from_pos[0] + t_quat_grad_from_quat[0],
            t_quat_grad_from_pos[1] + t_quat_grad_from_quat[1],
            t_quat_grad_from_pos[2] + t_quat_grad_from_quat[2],
            t_quat_grad_from_pos[3] + t_quat_grad_from_quat[3],
        ],
        dt=gs.qd_float,
    )
    return pos_grad, quat_grad, t_trans_grad, t_quat_grad


# =========================================================================
# `kernel_manual_COM_links_bw`: FULL manual reverse for `func_COM_links`.
#
# Drop-in replacement for `kernel_COM_links.grad`. Avoids Quadrants AD silent
# drop on cross-link parent.pos / parent.quat chain (j_pos_bw chain) AND
# any similar issue with atomic accumulations (mass_sum / root_COM_bw) or
# cross-link reads (root_COM propagation Phase 3).
#
# Forward to reverse mapping (statement-reverse order, Phase 6 → 1):
#   - Phase 6: cdofvel + cdof + offset_pos chain → vel/xaxis/xanchor/root_COM/quat grads
#   - Phase 5: j_pos/j_quat chain → pos[i_l]/quat[i_l] or pos[i_p]/quat[i_p]
#   - Phase 4: cinr + i_pos chain → i_pos_bw/i_quat/root_COM/mass(_shift) grads
#   - Phase 3: root_COM propagation reverse → root_COM[i_r].grad += root_COM[i_l].grad
#   - Phase 2: root_COM = root_COM_bw / mass_sum reverse → root_COM_bw/mass_sum grads
#   - Phase 1: i_pos_bw/i_quat compute + mass_sum/root_COM_bw atomic accum reverse
#              → pos/quat/i_pos_shift/mass_shift grads
#
# Joint type support: FREE, REVOLUTE, PRISMATIC, FIXED (n_dofs=0), SPHERICAL.
# Hibernation NOT supported — flips errno bit.
# =========================================================================


@qd.kernel(fastcache=True)
def kernel_manual_COM_links_bw(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    errno: qd.Tensor,
):
    """Full manual reverse for `func_COM_links_entity` — replaces
    `kernel_COM_links.grad`.

    Inputs (read .grad seeds — all outputs of `kernel_COM_links` forward):
      i_pos_bw, i_quat, mass_sum, root_COM_bw, root_COM, i_pos,
      cinr_inertial, cinr_pos, cinr_quat, cinr_mass,
      j_pos_bw, j_quat_bw, j_pos, j_quat,
      cdof_ang, cdof_vel, cdofvel_ang, cdofvel_vel

    Outputs (accumulated .grad):
      links_state.pos.grad, links_state.quat.grad,
      links_state.mass_shift.grad, links_state.i_pos_shift.grad,
      dofs_state.vel.grad,
      joints_state.xanchor.grad, joints_state.xaxis.grad

    P8 (consume) is applied to all output grad seeds at the end so the
    next backward call sees clean state.
    """
    EPS = rigid_global_info.EPS[None]
    qd.loop_config(
        name="manual_COM_links_bw",
        serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL),
    )
    for i_e, i_b in qd.ndrange(entities_info.n_links.shape[0], links_state.pos.shape[1]):
        if qd.static(static_rigid_sim_config.use_hibernation):
            errno[i_b] = errno[i_b] | array_class.ErrorCode.MANUAL_BW_UNIMPLEMENTED_JOINT_TYPE
        else:
            # ─────────────────────────────────────────────────────────────
            # PHASE 6 reverse
            # Forward (per i_l with n_dofs > 0, per joint i_j):
            #   offset_pos = root_COM[i_l] - xanchor[i_j]
            #   joint_type-specific cdof_ang / cdof_vel
            #   cdofvel_ang[d] = cdof_ang[d] * vel[d]
            #   cdofvel_vel[d] = cdof_vel[d] * vel[d]
            # ─────────────────────────────────────────────────────────────
            for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
                I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
                if links_info.n_dofs[I_l] > 0:
                    for i_j in range(links_info.joint_start[I_l], links_info.joint_end[I_l]):
                        I_j = [i_j, i_b] if qd.static(static_rigid_sim_config.batch_joints_info) else i_j
                        joint_type = joints_info.type[I_j]
                        dof_start = joints_info.dof_start[I_j]
                        dof_end = joints_info.dof_end[I_j]

                        offset_pos = links_state.root_COM[i_l, i_b] - joints_state.xanchor[i_j, i_b]

                        # ─ (c) reverse: cdofvel_*.grad → cdof_*.grad + vel.grad
                        for i_d in range(dof_start, dof_end):
                            cdofvel_ang_g = dofs_state.cdofvel_ang.grad[i_d, i_b]
                            cdofvel_vel_g = dofs_state.cdofvel_vel.grad[i_d, i_b]
                            vel_at_d = dofs_state.vel[i_d, i_b]
                            for k in qd.static(range(3)):
                                dofs_state.cdof_ang.grad[i_d, i_b][k] = (
                                    dofs_state.cdof_ang.grad[i_d, i_b][k] + cdofvel_ang_g[k] * vel_at_d
                                )
                                dofs_state.cdof_vel.grad[i_d, i_b][k] = (
                                    dofs_state.cdof_vel.grad[i_d, i_b][k] + cdofvel_vel_g[k] * vel_at_d
                                )
                            # vel.grad += dot products
                            dot_ang = (
                                dofs_state.cdof_ang[i_d, i_b][0] * cdofvel_ang_g[0]
                                + dofs_state.cdof_ang[i_d, i_b][1] * cdofvel_ang_g[1]
                                + dofs_state.cdof_ang[i_d, i_b][2] * cdofvel_ang_g[2]
                            )
                            dot_vel = (
                                dofs_state.cdof_vel[i_d, i_b][0] * cdofvel_vel_g[0]
                                + dofs_state.cdof_vel[i_d, i_b][1] * cdofvel_vel_g[1]
                                + dofs_state.cdof_vel[i_d, i_b][2] * cdofvel_vel_g[2]
                            )
                            dofs_state.vel.grad[i_d, i_b] = dofs_state.vel.grad[i_d, i_b] + dot_ang + dot_vel
                            # P8 consume cdofvel.grad
                            for k in qd.static(range(3)):
                                dofs_state.cdofvel_ang.grad[i_d, i_b][k] = 0.0
                                dofs_state.cdofvel_vel.grad[i_d, i_b][k] = 0.0

                        # ─ (b) reverse: cdof_*.grad → xaxis.grad / xmat_T.grad / offset_pos.grad
                        offset_pos_grad = qd.Vector.zero(gs.qd_float, 3)
                        # quat_grad_acc collects contributions from R(quat[i_l]) chain (FREE/SPHERICAL angular)
                        link_quat_grad_acc = qd.Vector.zero(gs.qd_float, 4)

                        if joint_type == gs.JOINT_TYPE.REVOLUTE:
                            cdof_ang_g = dofs_state.cdof_ang.grad[dof_start, i_b]
                            cdof_vel_g = dofs_state.cdof_vel.grad[dof_start, i_b]
                            xaxis_primal = joints_state.xaxis[i_j, i_b]
                            # cdof_ang = xaxis  →  xaxis.grad += cdof_ang.grad
                            # cdof_vel = xaxis × offset_pos →
                            #   xaxis.grad += offset_pos × cdof_vel.grad
                            #   offset_pos.grad += cdof_vel.grad × xaxis
                            xaxis_grad_contrib = cdof_ang_g + offset_pos.cross(cdof_vel_g)
                            for k in qd.static(range(3)):
                                joints_state.xaxis.grad[i_j, i_b][k] = (
                                    joints_state.xaxis.grad[i_j, i_b][k] + xaxis_grad_contrib[k]
                                )
                            offset_pos_grad = offset_pos_grad + cdof_vel_g.cross(xaxis_primal)
                            # P8 consume cdof_*.grad
                            for k in qd.static(range(3)):
                                dofs_state.cdof_ang.grad[dof_start, i_b][k] = 0.0
                                dofs_state.cdof_vel.grad[dof_start, i_b][k] = 0.0
                        elif joint_type == gs.JOINT_TYPE.PRISMATIC:
                            cdof_vel_g = dofs_state.cdof_vel.grad[dof_start, i_b]
                            # cdof_ang = 0 (no chain). cdof_vel = xaxis → xaxis.grad += cdof_vel.grad
                            for k in qd.static(range(3)):
                                joints_state.xaxis.grad[i_j, i_b][k] = (
                                    joints_state.xaxis.grad[i_j, i_b][k] + cdof_vel_g[k]
                                )
                                dofs_state.cdof_ang.grad[dof_start, i_b][k] = 0.0
                                dofs_state.cdof_vel.grad[dof_start, i_b][k] = 0.0
                        elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                            # 3 dofs: cdof_ang[d_i] = R(quat[i_l])[:, i] = qd_transform_by_quat(e_i, quat[i_l])
                            #         cdof_vel[d_i] = xmat_T_row × offset_pos
                            link_quat = links_state.quat[i_l, i_b]
                            R_at_il = gu.qd_quat_to_R(link_quat, EPS)
                            R_grad_matrix = qd.Matrix.zero(gs.qd_float, 3, 3)
                            for i in qd.static(range(3)):
                                cdof_ang_g = dofs_state.cdof_ang.grad[dof_start + i, i_b]
                                cdof_vel_g = dofs_state.cdof_vel.grad[dof_start + i, i_b]
                                # xmat_T[i, :] = R.T[i, :] = R[:, i]   ← column i of R
                                xmat_T_row_primal = qd.Vector(
                                    [R_at_il[0, i], R_at_il[1, i], R_at_il[2, i]], dt=gs.qd_float
                                )
                                # cdof_ang[d_i] = xmat_T[i, :]  →  R[:, i].grad += cdof_ang.grad
                                # cdof_vel[d_i] = xmat_T_row × offset_pos →
                                #   xmat_T_row.grad += offset_pos × cdof_vel_g
                                #   offset_pos.grad += cdof_vel_g × xmat_T_row
                                row_grad_total = cdof_ang_g + offset_pos.cross(cdof_vel_g)
                                offset_pos_grad = offset_pos_grad + cdof_vel_g.cross(xmat_T_row_primal)
                                # Stash R_grad_matrix[:, i] += row_grad_total
                                for k in qd.static(range(3)):
                                    R_grad_matrix[k, i] = R_grad_matrix[k, i] + row_grad_total[k]
                                # P8 consume
                                for k in qd.static(range(3)):
                                    dofs_state.cdof_ang.grad[dof_start + i, i_b][k] = 0.0
                                    dofs_state.cdof_vel.grad[dof_start + i, i_b][k] = 0.0
                            # quat.grad += d_R(quat) chain
                            link_quat_grad_acc = link_quat_grad_acc + d_qd_quat_to_R__dquat(link_quat, R_grad_matrix)
                        elif joint_type == gs.JOINT_TYPE.FREE:
                            # Linear i=0..2: cdof_ang = 0, cdof_vel = e_i (constants, no chain).
                            # Just P8 consume.
                            for i in qd.static(range(3)):
                                for k in qd.static(range(3)):
                                    dofs_state.cdof_ang.grad[dof_start + i, i_b][k] = 0.0
                                    dofs_state.cdof_vel.grad[dof_start + i, i_b][k] = 0.0
                            # Angular i=0..2: same SPHERICAL pattern offset by 3 dofs.
                            link_quat = links_state.quat[i_l, i_b]
                            R_at_il = gu.qd_quat_to_R(link_quat, EPS)
                            R_grad_matrix = qd.Matrix.zero(gs.qd_float, 3, 3)
                            for i in qd.static(range(3)):
                                cdof_ang_g = dofs_state.cdof_ang.grad[dof_start + 3 + i, i_b]
                                cdof_vel_g = dofs_state.cdof_vel.grad[dof_start + 3 + i, i_b]
                                xmat_T_row_primal = qd.Vector(
                                    [R_at_il[0, i], R_at_il[1, i], R_at_il[2, i]], dt=gs.qd_float
                                )
                                row_grad_total = cdof_ang_g + offset_pos.cross(cdof_vel_g)
                                offset_pos_grad = offset_pos_grad + cdof_vel_g.cross(xmat_T_row_primal)
                                for k in qd.static(range(3)):
                                    R_grad_matrix[k, i] = R_grad_matrix[k, i] + row_grad_total[k]
                                for k in qd.static(range(3)):
                                    dofs_state.cdof_ang.grad[dof_start + 3 + i, i_b][k] = 0.0
                                    dofs_state.cdof_vel.grad[dof_start + 3 + i, i_b][k] = 0.0
                            link_quat_grad_acc = link_quat_grad_acc + d_qd_quat_to_R__dquat(link_quat, R_grad_matrix)

                        # Accumulate the R(quat) chain contribution into links_state.quat.grad
                        for k in qd.static(range(4)):
                            links_state.quat.grad[i_l, i_b][k] = (
                                links_state.quat.grad[i_l, i_b][k] + link_quat_grad_acc[k]
                            )

                        # ─ (a) reverse: offset_pos = root_COM[i_l] - xanchor[i_j]
                        for k in qd.static(range(3)):
                            links_state.root_COM.grad[i_l, i_b][k] = (
                                links_state.root_COM.grad[i_l, i_b][k] + offset_pos_grad[k]
                            )
                            joints_state.xanchor.grad[i_j, i_b][k] = (
                                joints_state.xanchor.grad[i_j, i_b][k] - offset_pos_grad[k]
                            )

            # ─────────────────────────────────────────────────────────────
            # PHASE 5 reverse — (j_pos / j_quat propagation)
            # Same as kernel_manual_COM_links_phase5_bw body — see notes there.
            # ─────────────────────────────────────────────────────────────
            for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
                I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
                if links_info.n_dofs[I_l] > 0:
                    i_p = links_info.parent_idx[I_l]
                    _i_j_5 = links_info.joint_start[I_l]
                    _I_j_5 = [_i_j_5, i_b] if qd.static(static_rigid_sim_config.batch_joints_info) else _i_j_5
                    joint_type_5 = joints_info.type[_I_j_5]

                    if joint_type_5 == gs.JOINT_TYPE.FREE or (links_info.is_fixed[I_l] and i_p == -1):
                        for k in qd.static(range(3)):
                            links_state.pos.grad[i_l, i_b][k] = (
                                links_state.pos.grad[i_l, i_b][k] + links_state.j_pos.grad[i_l, i_b][k]
                            )
                        for k in qd.static(range(4)):
                            links_state.quat.grad[i_l, i_b][k] = (
                                links_state.quat.grad[i_l, i_b][k] + links_state.j_quat.grad[i_l, i_b][k]
                            )
                        for k in qd.static(range(3)):
                            links_state.j_pos.grad[i_l, i_b][k] = 0.0
                        for k in qd.static(range(4)):
                            links_state.j_quat.grad[i_l, i_b][k] = 0.0
                    else:
                        n_joints_5 = links_info.joint_end[I_l] - links_info.joint_start[I_l]
                        # Self-compute j_quat_bw primal (same for all k — identity rhs mul).
                        p_quat_for_jbw = gu.qd_identity_quat()
                        if i_p != -1:
                            p_quat_for_jbw = links_state.quat[i_p, i_b]
                        j_quat_bw_const = gu.qd_quat_mul(p_quat_for_jbw, links_info.quat[I_l])

                        for k in qd.static(range(3)):
                            links_state.j_pos_bw.grad[i_l, n_joints_5, i_b][k] = (
                                links_state.j_pos_bw.grad[i_l, n_joints_5, i_b][k] + links_state.j_pos.grad[i_l, i_b][k]
                            )
                        for k in qd.static(range(4)):
                            links_state.j_quat_bw.grad[i_l, n_joints_5, i_b][k] = (
                                links_state.j_quat_bw.grad[i_l, n_joints_5, i_b][k]
                                + links_state.j_quat.grad[i_l, i_b][k]
                            )
                        for k in qd.static(range(3)):
                            links_state.j_pos.grad[i_l, i_b][k] = 0.0
                        for k in qd.static(range(4)):
                            links_state.j_quat.grad[i_l, i_b][k] = 0.0

                        for i_j_rev_5 in range(n_joints_5):
                            i_j_5_ = n_joints_5 - 1 - i_j_rev_5
                            i_j_5 = i_j_5_ + links_info.joint_start[I_l]
                            I_j_5 = [i_j_5, i_b] if qd.static(static_rigid_sim_config.batch_joints_info) else i_j_5
                            curr_idx_5 = i_j_5_
                            next_idx_5 = i_j_5_ + 1
                            j_quat_bw_curr_5 = j_quat_bw_const
                            j_pos_bw_next_grad_5 = links_state.j_pos_bw.grad[i_l, next_idx_5, i_b]
                            j_quat_bw_next_grad_5 = links_state.j_quat_bw.grad[i_l, next_idx_5, i_b]
                            joint_pos_off_5 = joints_info.pos[I_j_5]
                            quat_grad_from_pos_5 = d_transform_by_quat__dq(
                                joint_pos_off_5, j_quat_bw_curr_5, j_pos_bw_next_grad_5
                            )
                            for k in qd.static(range(3)):
                                links_state.j_pos_bw.grad[i_l, curr_idx_5, i_b][k] = (
                                    links_state.j_pos_bw.grad[i_l, curr_idx_5, i_b][k] + j_pos_bw_next_grad_5[k]
                                )
                            for k in qd.static(range(4)):
                                links_state.j_quat_bw.grad[i_l, curr_idx_5, i_b][k] = (
                                    links_state.j_quat_bw.grad[i_l, curr_idx_5, i_b][k]
                                    + quat_grad_from_pos_5[k]
                                    + j_quat_bw_next_grad_5[k]
                                )
                            for k in qd.static(range(3)):
                                links_state.j_pos_bw.grad[i_l, next_idx_5, i_b][k] = 0.0
                            for k in qd.static(range(4)):
                                links_state.j_quat_bw.grad[i_l, next_idx_5, i_b][k] = 0.0

                        l_info_pos_5 = links_info.pos[I_l]
                        l_info_quat_5 = links_info.quat[I_l]
                        j_pos_bw_0_grad_5 = links_state.j_pos_bw.grad[i_l, 0, i_b]
                        j_quat_bw_0_grad_5 = links_state.j_quat_bw.grad[i_l, 0, i_b]

                        if i_p != -1:
                            p_quat_5 = links_state.quat[i_p, i_b]
                            p_quat_grad_from_pos_5 = d_transform_by_quat__dq(l_info_pos_5, p_quat_5, j_pos_bw_0_grad_5)
                            p_quat_grad_from_quat_5 = d_quat_mul__dlhs(p_quat_5, l_info_quat_5, j_quat_bw_0_grad_5)
                            for k in qd.static(range(3)):
                                links_state.pos.grad[i_p, i_b][k] = (
                                    links_state.pos.grad[i_p, i_b][k] + j_pos_bw_0_grad_5[k]
                                )
                            for k in qd.static(range(4)):
                                links_state.quat.grad[i_p, i_b][k] = (
                                    links_state.quat.grad[i_p, i_b][k]
                                    + p_quat_grad_from_pos_5[k]
                                    + p_quat_grad_from_quat_5[k]
                                )
                        for k in qd.static(range(3)):
                            links_state.j_pos_bw.grad[i_l, 0, i_b][k] = 0.0
                        for k in qd.static(range(4)):
                            links_state.j_quat_bw.grad[i_l, 0, i_b][k] = 0.0

            # ─────────────────────────────────────────────────────────────
            # PHASE 4 reverse — (cinr + i_pos chain)
            # Forward: i_pos[i_l] = i_pos_bw[i_l] - root_COM[i_l]
            #          mass = inertial_mass + mass_shift[i_l]
            #          (cinr_inertial, cinr_pos, cinr_quat, cinr_mass) =
            #              qd_transform_inertia_by_trans_quat(inertial_i, mass, i_pos, i_quat, EPS)
            # ─────────────────────────────────────────────────────────────
            for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
                I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l

                inertial_i_const = links_info.inertial_i[I_l]
                mass = links_info.inertial_mass[I_l] + links_state.mass_shift[i_l, i_b]
                i_pos_primal = links_state.i_pos[i_l, i_b]
                i_quat_primal = links_state.i_quat[i_l, i_b]

                cinr_i_g = links_state.cinr_inertial.grad[i_l, i_b]
                cinr_pos_g = links_state.cinr_pos.grad[i_l, i_b]
                cinr_quat_g = links_state.cinr_quat.grad[i_l, i_b]
                cinr_mass_g = links_state.cinr_mass.grad[i_l, i_b]

                (mass_grad_from_cinr, trans_grad_from_cinr, quat_grad_from_cinr) = d_qd_transform_inertia_by_trans_quat(
                    inertial_i_const,
                    mass,
                    i_pos_primal,
                    i_quat_primal,
                    EPS,
                    cinr_i_g,
                    cinr_pos_g,
                    cinr_quat_g,
                    cinr_mass_g,
                )

                # Accumulate trans_grad → i_pos.grad
                for k in qd.static(range(3)):
                    links_state.i_pos.grad[i_l, i_b][k] = links_state.i_pos.grad[i_l, i_b][k] + trans_grad_from_cinr[k]
                # i_quat.grad += quat_grad
                for k in qd.static(range(4)):
                    links_state.i_quat.grad[i_l, i_b][k] = links_state.i_quat.grad[i_l, i_b][k] + quat_grad_from_cinr[k]
                # mass.grad → mass_shift.grad (mass = inertial_mass + mass_shift)
                links_state.mass_shift.grad[i_l, i_b] = links_state.mass_shift.grad[i_l, i_b] + mass_grad_from_cinr
                # P8 consume cinr.grad
                for r in qd.static(range(3)):
                    for c in qd.static(range(3)):
                        links_state.cinr_inertial.grad[i_l, i_b][r, c] = 0.0
                for k in qd.static(range(3)):
                    links_state.cinr_pos.grad[i_l, i_b][k] = 0.0
                for k in qd.static(range(4)):
                    links_state.cinr_quat.grad[i_l, i_b][k] = 0.0
                links_state.cinr_mass.grad[i_l, i_b] = 0.0

                # i_pos = i_pos_bw - root_COM → i_pos_bw.grad += i_pos.grad; root_COM.grad += -i_pos.grad
                ip_g = links_state.i_pos.grad[i_l, i_b]
                for k in qd.static(range(3)):
                    links_state.i_pos_bw.grad[i_l, i_b][k] = links_state.i_pos_bw.grad[i_l, i_b][k] + ip_g[k]
                    links_state.root_COM.grad[i_l, i_b][k] = links_state.root_COM.grad[i_l, i_b][k] - ip_g[k]
                # P8 consume i_pos.grad
                for k in qd.static(range(3)):
                    links_state.i_pos.grad[i_l, i_b][k] = 0.0

            # ─────────────────────────────────────────────────────────────
            # PHASE 3 reverse — root_COM[i_l] = root_COM[i_r] propagation
            # ─────────────────────────────────────────────────────────────
            for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
                I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
                i_r = links_info.root_idx[I_l]
                if i_l != i_r:
                    for k in qd.static(range(3)):
                        links_state.root_COM.grad[i_r, i_b][k] = (
                            links_state.root_COM.grad[i_r, i_b][k] + links_state.root_COM.grad[i_l, i_b][k]
                        )
                        links_state.root_COM.grad[i_l, i_b][k] = 0.0

            # ─────────────────────────────────────────────────────────────
            # PHASE 2 reverse — root_COM[i_r] = root_COM_bw[i_r] / mass_sum[i_r]
            # (only i_r = root link)
            # ─────────────────────────────────────────────────────────────
            for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
                I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
                i_r = links_info.root_idx[I_l]
                if i_l == i_r:
                    mass_sum_val = links_state.mass_sum[i_l, i_b]
                    root_COM_g = links_state.root_COM.grad[i_l, i_b]
                    if mass_sum_val > EPS:
                        # root_COM = root_COM_bw / mass_sum
                        for k in qd.static(range(3)):
                            links_state.root_COM_bw.grad[i_l, i_b][k] = (
                                links_state.root_COM_bw.grad[i_l, i_b][k] + root_COM_g[k] / mass_sum_val
                            )
                        # mass_sum.grad += -(root_COM_g · root_COM_bw) / mass_sum²
                        rcb = links_state.root_COM_bw[i_l, i_b]
                        rg_dot_rcb = root_COM_g[0] * rcb[0] + root_COM_g[1] * rcb[1] + root_COM_g[2] * rcb[2]
                        links_state.mass_sum.grad[i_l, i_b] = links_state.mass_sum.grad[i_l, i_b] - rg_dot_rcb / (
                            mass_sum_val * mass_sum_val
                        )
                    else:
                        # Degenerate: root_COM[i_r] = i_pos_bw[i_r]
                        for k in qd.static(range(3)):
                            links_state.i_pos_bw.grad[i_l, i_b][k] = (
                                links_state.i_pos_bw.grad[i_l, i_b][k] + root_COM_g[k]
                            )
                    # P8 consume root_COM.grad (only root has nonzero by now after Phase 3)
                    for k in qd.static(range(3)):
                        links_state.root_COM.grad[i_l, i_b][k] = 0.0

            # ─────────────────────────────────────────────────────────────
            # PHASE 1 reverse — i_pos_bw / i_quat compute + mass_sum/root_COM_bw atomic adds
            # ─────────────────────────────────────────────────────────────
            for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
                I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
                i_r = links_info.root_idx[I_l]

                mass = links_info.inertial_mass[I_l] + links_state.mass_shift[i_l, i_b]
                i_pos_bw_primal = links_state.i_pos_bw[i_l, i_b]
                rcb_g = links_state.root_COM_bw.grad[i_r, i_b]
                ms_g = links_state.mass_sum.grad[i_r, i_b]

                # mass_sum[i_r] += mass: mass.grad += ms_g
                # root_COM_bw[i_r] += mass · i_pos_bw[i_l]:
                #   i_pos_bw.grad += mass · rcb_g
                #   mass.grad += rcb_g · i_pos_bw
                mass_grad = ms_g + (
                    rcb_g[0] * i_pos_bw_primal[0] + rcb_g[1] * i_pos_bw_primal[1] + rcb_g[2] * i_pos_bw_primal[2]
                )
                links_state.mass_shift.grad[i_l, i_b] = links_state.mass_shift.grad[i_l, i_b] + mass_grad
                for k in qd.static(range(3)):
                    links_state.i_pos_bw.grad[i_l, i_b][k] = links_state.i_pos_bw.grad[i_l, i_b][k] + mass * rcb_g[k]

                # (i_pos_bw[i_l], i_quat[i_l]) = qd_transform_pos_quat_by_trans_quat(
                #     inertial_pos + i_pos_shift, inertial_quat, pos[i_l], quat[i_l]
                # )
                v_pos = links_info.inertial_pos[I_l] + links_state.i_pos_shift[i_l, i_b]
                v_quat_const = links_info.inertial_quat[I_l]
                t_pos = links_state.pos[i_l, i_b]
                t_quat = links_state.quat[i_l, i_b]
                ipb_g = links_state.i_pos_bw.grad[i_l, i_b]
                iq_g = links_state.i_quat.grad[i_l, i_b]

                pos_g_contrib, _, t_trans_g_contrib, t_quat_g_contrib = d_qd_transform_pos_quat_by_trans_quat(
                    v_pos, v_quat_const, t_pos, t_quat, ipb_g, iq_g
                )
                # t_trans_g_contrib = ipb_g (identity)
                for k in qd.static(range(3)):
                    links_state.pos.grad[i_l, i_b][k] = links_state.pos.grad[i_l, i_b][k] + t_trans_g_contrib[k]
                    links_state.i_pos_shift.grad[i_l, i_b][k] = (
                        links_state.i_pos_shift.grad[i_l, i_b][k] + pos_g_contrib[k]
                    )
                for k in qd.static(range(4)):
                    links_state.quat.grad[i_l, i_b][k] = links_state.quat.grad[i_l, i_b][k] + t_quat_g_contrib[k]

                # P8 consume i_pos_bw / i_quat .grad
                for k in qd.static(range(3)):
                    links_state.i_pos_bw.grad[i_l, i_b][k] = 0.0
                for k in qd.static(range(4)):
                    links_state.i_quat.grad[i_l, i_b][k] = 0.0

            # Consume mass_sum.grad / root_COM_bw.grad now that Phase 1 is done.
            for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
                links_state.mass_sum.grad[i_l, i_b] = 0.0
                for k in qd.static(range(3)):
                    links_state.root_COM_bw.grad[i_l, i_b][k] = 0.0


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
