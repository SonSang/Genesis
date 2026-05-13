"""Manual backward kernels for `update_cartesian_space` (FK reverse).

Bypasses Quadrants AD's silent drop on
`func_forward_kinematics_entity_one_link` (Phase A
`pos_ = parent_pos + qd_transform_by_quat(arm_local, parent_quat)`) by
computing the FK Jacobian-transpose explicitly.

Scope (initial): J4 / J5 topologies — free joint chassis + revolute
chains. Single-batch. Other joint types (PRISMATIC / SPHERICAL / FIXED)
not implemented yet.

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

    Joint types supported: FREE, REVOLUTE.
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

            elif joint_type == gs.JOINT_TYPE.REVOLUTE:
                parent_idx = links_info.parent_idx[I_l]
                # Forward primal: parent_quat = links_state.quat[parent_idx, i_b]
                parent_quat = links_state.quat[parent_idx, i_b]
                arm_local = links_info.pos[I_l]

                I_d = [dof_start, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else dof_start
                axis = dofs_info.motion_ang[I_d]
                angle = rigid_global_info.qpos[q_start, i_b] - rigid_global_info.qpos0[q_start, i_b]
                rotvec = axis * angle
                qloc = gu.qd_rotvec_to_quat(rotvec, rigid_global_info.EPS[None])

                # Backward chain
                arm_pos_grad = links_state.pos.grad[i_l, i_b]
                arm_quat_grad = links_state.quat.grad[i_l, i_b]

                # arm_pos = parent_pos + R(parent_quat) · arm_local
                parent_quat_grad_from_pos = d_transform_by_quat__dq(arm_local, parent_quat, arm_pos_grad)

                # arm_quat = qloc ⊗ parent_quat  (Hamilton: out = qloc·parent_quat)
                qloc_grad = d_quat_mul__dlhs(qloc, parent_quat, arm_quat_grad)
                parent_quat_grad_from_quat = d_quat_mul__drhs(qloc, parent_quat, arm_quat_grad)

                # qloc = rotvec_to_quat(axis · angle)
                rotvec_grad = d_rotvec_to_quat__drotvec(rotvec, rigid_global_info.EPS[None], qloc_grad)
                angle_grad = axis[0] * rotvec_grad[0] + axis[1] * rotvec_grad[1] + axis[2] * rotvec_grad[2]

                # Accumulate into qpos
                rigid_global_info.qpos.grad[q_start, i_b] = rigid_global_info.qpos.grad[q_start, i_b] + angle_grad

                # Accumulate into parent's links_state grads (NOTE: same-buffer
                # cross-index write within a single launch — write only, not
                # read in same launch, so per-launch access rule is honored).
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
            # PRISMATIC / SPHERICAL / FIXED not yet implemented — skipped.
