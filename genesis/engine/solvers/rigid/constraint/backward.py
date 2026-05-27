import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu


@qd.func
def func_matvec_Ap(
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    i_b,
):
    """
    Compute Ap = (M + J^T * diag(D) * J) * p on the current active set, which is used for solving the adjoint u.

    Specifically, M = mass matrix, J = Jacobian, diag(D) = diagonal matrix of efc_D, and p = search direction.
    """
    n_dofs = constraint_state.bw_p.shape[0]
    for i_d in range(n_dofs):
        constraint_state.bw_Ap[i_d, i_b] = 0.0

    # Mp: Block multiplication
    n_entities = entities_info.n_links.shape[0]
    for i_e in range(n_entities):
        for i_d1 in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
            acc = gs.qd_float(0.0)
            for i_d2 in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                acc += rigid_global_info.mass_mat[i_d1, i_d2, i_b] * constraint_state.bw_p[i_d2, i_b]
            constraint_state.bw_Ap[i_d1, i_b] += acc

    # tmp = J v
    for i_c in range(constraint_state.n_constraints[i_b]):
        jv = gs.qd_float(0.0)
        if qd.static(static_rigid_sim_config.sparse_solve):
            for k in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                i_d = constraint_state.jac_relevant_dofs[i_c, k, i_b]
                jv += constraint_state.jac[i_c, i_d, i_b] * constraint_state.bw_p[i_d, i_b]
        else:
            for i_d in range(n_dofs):
                jv += constraint_state.jac[i_c, i_d, i_b] * constraint_state.bw_p[i_d, i_b]
        # only active constraints contribute
        jv *= constraint_state.efc_D[i_c, i_b] * constraint_state.active[i_c, i_b]
        # out += J^T (D * J v)
        if qd.static(static_rigid_sim_config.sparse_solve):
            for k in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                i_d = constraint_state.jac_relevant_dofs[i_c, k, i_b]
                constraint_state.bw_Ap[i_d, i_b] += constraint_state.jac[i_c, i_d, i_b] * jv
        else:
            for i_d in range(n_dofs):
                constraint_state.bw_Ap[i_d, i_b] += constraint_state.jac[i_c, i_d, i_b] * jv


@qd.kernel
def kernel_solve_adjoint_u(
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    r"""
    Solve for the adjoint vector [u] from Au = g, where A = dF/dqacc (primal Hessian on the active set) and g = dL/dqacc.
    Intuitively, [u] is a sensitivity vector that translates the upstream gradient dL/dqacc into the primal space.
    This adjoint vector [u] can be used an intermediate variable to compute the downstream gradients. Since A is a
    Semi-Positive Definite (SPD) matrix, we can solve A * u = g using either Cholesky decomposition or CG solver.
    When Newton solver was used, we reuse the Cholesky decomposition of A (= L * L^T) to solve A * u = g. Otherwise,
    we use CG solver.

    Specifically, A = M + J^T * diag(D) * J, where M = mass matrix, J = Jacobian, diag(D) = diagonal matrix of efc_D.
    """
    n_dofs = constraint_state.bw_u.shape[0]
    _B = constraint_state.bw_u.shape[1]

    # Initialize u
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        constraint_state.bw_u[i_d, i_b] = 0.0

    if qd.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
        # Since we already have the Cholesky decomposition of A (= L * L^T), we can use it to solve A * u = g.
        for i_b in range(_B):
            # z = L^{-1} g  (forward substitution)
            # Save solution to bw_r
            for i_d in range(n_dofs):
                z = constraint_state.dL_dqacc[i_d, i_b]
                for j_d in range(i_d):
                    z -= constraint_state.nt_H[i_b, i_d, j_d] * constraint_state.bw_r[j_d, i_b]
                z /= constraint_state.nt_H[i_b, i_d, i_d]
                constraint_state.bw_r[i_d, i_b] = z

            # u = L^{-T} z  (back substitution)
            for i_d_ in range(n_dofs):
                i_d = n_dofs - 1 - i_d_
                u = constraint_state.bw_r[i_d, i_b]
                for j_d in range(i_d + 1, n_dofs):
                    u -= constraint_state.nt_H[i_b, j_d, i_d] * constraint_state.bw_u[j_d, i_b]
                u /= constraint_state.nt_H[i_b, i_d, i_d]
                constraint_state.bw_u[i_d, i_b] = u
    else:
        # Use CG solver for solving A * u = g.
        # 2. Local buffers for solving A * u = g
        # Initialize r, p with dL_dqacc
        for i_d, i_b in qd.ndrange(n_dofs, _B):
            # Residual: g - A * 0 (u = 0)
            constraint_state.bw_r[i_d, i_b] = constraint_state.dL_dqacc[i_d, i_b]
            # Search direction: p = r
            constraint_state.bw_p[i_d, i_b] = constraint_state.bw_r[i_d, i_b]

        # 3. Solve A * u = g, parallelized over batch dimension
        for i_b in range(_B):
            # Compute Ap for the current search direction
            for it in range(static_rigid_sim_config.iterations):
                func_matvec_Ap(
                    entities_info=entities_info,
                    rigid_global_info=rigid_global_info,
                    constraint_state=constraint_state,
                    static_rigid_sim_config=static_rigid_sim_config,
                    i_b=i_b,
                )

                # alpha = (r,r)/(p,Hp)
                num = gs.qd_float(0.0)
                den = gs.qd_float(0.0)
                for i_d in range(n_dofs):
                    num += constraint_state.bw_r[i_d, i_b] * constraint_state.bw_r[i_d, i_b]
                    den += constraint_state.bw_p[i_d, i_b] * constraint_state.bw_Ap[i_d, i_b]
                alpha = num / qd.max(den, rigid_global_info.EPS[None])

                # u += alpha p ; r -= alpha Hp
                for i_d in range(n_dofs):
                    constraint_state.bw_u[i_d, i_b] += alpha * constraint_state.bw_p[i_d, i_b]
                    constraint_state.bw_r[i_d, i_b] -= alpha * constraint_state.bw_Ap[i_d, i_b]

                # check tol (optional: per-batch)
                # TODO: Might need lower tolerance?
                if num < rigid_global_info.EPS[None]:
                    break

                # beta = (r_new,r_new)/(r_old,r_old)
                num_new = gs.qd_float(0.0)
                for i_d in range(n_dofs):
                    num_new += constraint_state.bw_r[i_d, i_b] * constraint_state.bw_r[i_d, i_b]
                beta = num_new / qd.max(num, rigid_global_info.EPS[None])

                # p = r + beta p
                for i_d in range(n_dofs):
                    constraint_state.bw_p[i_d, i_b] = (
                        constraint_state.bw_r[i_d, i_b] + beta * constraint_state.bw_p[i_d, i_b]
                    )


@qd.kernel
def kernel_compute_gradients(
    entities_info: array_class.EntitiesInfo,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    r"""
    Compute gradients of the loss with respect to the input variables to this solver. Note that we use the intermediate
    adjoint vector [u] computed in [kernel_solve_adjoint_u] to compute these gradients.

    Specifically, the gradients are computed as follows:
    - dL_dM = -u * qacc^T
    - dL_djac = -[u * y^T + qacc * (D \odot (Ju))^T] (y = D \odot w, w = (Jqacc - aref))
    - dL_daref = Ju \odot D
    - dL_defc_D = -Ju \odot (Jqacc - aref)
    - dL_dforce = u
    """
    _B = constraint_state.bw_u.shape[1]
    n_dofs = constraint_state.bw_u.shape[0]
    n_constraints = constraint_state.bw_Ju.shape[0]

    # clear grads
    for i_d0, i_d1, i_b in qd.ndrange(n_dofs, n_dofs, _B):
        constraint_state.dL_dM[i_d0, i_d1, i_b] = gs.qd_float(0.0)
    for i_c, i_d, i_b in qd.ndrange(n_constraints, n_dofs, _B):
        constraint_state.dL_djac[i_c, i_d, i_b] = gs.qd_float(0.0)
    for i_c, i_b in qd.ndrange(n_constraints, _B):
        constraint_state.dL_daref[i_c, i_b] = gs.qd_float(0.0)
        constraint_state.dL_defc_D[i_c, i_b] = gs.qd_float(0.0)
        constraint_state.bw_Ju[i_c, i_b] = gs.qd_float(0.0)
        constraint_state.bw_y[i_c, i_b] = gs.qd_float(0.0)
        constraint_state.bw_w[i_c, i_b] = gs.qd_float(0.0)
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        constraint_state.dL_dforce[i_d, i_b] = gs.qd_float(0.0)

    # Ju, w, y
    for i_b in range(_B):
        # Ju
        for i_c in range(constraint_state.n_constraints[i_b]):
            s = gs.qd_float(0.0)
            if qd.static(static_rigid_sim_config.sparse_solve):
                for k in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                    i_d = constraint_state.jac_relevant_dofs[i_c, k, i_b]
                    s += constraint_state.jac[i_c, i_d, i_b] * constraint_state.bw_u[i_d, i_b]
            else:
                for i_d in range(n_dofs):
                    s += constraint_state.jac[i_c, i_d, i_b] * constraint_state.bw_u[i_d, i_b]
            constraint_state.bw_Ju[i_c, i_b] = s

        # w = J qacc - aref
        # y = D \odot w
        for i_c in range(constraint_state.n_constraints[i_b]):
            t = gs.qd_float(0.0)
            if qd.static(static_rigid_sim_config.sparse_solve):
                for k in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                    i_d = constraint_state.jac_relevant_dofs[i_c, k, i_b]
                    t += constraint_state.jac[i_c, i_d, i_b] * constraint_state.qacc[i_d, i_b]
            else:
                for i_d in range(n_dofs):
                    t += constraint_state.jac[i_c, i_d, i_b] * constraint_state.qacc[i_d, i_b]
            constraint_state.bw_w[i_c, i_b] = t - constraint_state.aref[i_c, i_b]
            constraint_state.bw_y[i_c, i_b] = constraint_state.efc_D[i_c, i_b] * constraint_state.bw_w[i_c, i_b]

        # grads
        # force: u
        for i_d in range(n_dofs):
            constraint_state.dL_dforce[i_d, i_b] += constraint_state.bw_u[i_d, i_b]

        # per-constraint (mask by active)
        # aref: Ju \odot D
        # D: -Ju \odot w
        # J: -[u * y^T + qacc * (D \odot (Ju)^T)]
        for i_c in range(constraint_state.n_constraints[i_b]):
            if constraint_state.active[i_c, i_b] != 0:
                # aref: Ju \odot D
                constraint_state.dL_daref[i_c, i_b] += (
                    constraint_state.efc_D[i_c, i_b] * constraint_state.bw_Ju[i_c, i_b]
                )
                # D: -Ju \odot w
                constraint_state.dL_defc_D[i_c, i_b] -= (
                    constraint_state.bw_Ju[i_c, i_b] * constraint_state.bw_w[i_c, i_b]
                )

                # J: -[u * y^T + qacc * (D \odot (Ju))^T]
                DJu_i = constraint_state.efc_D[i_c, i_b] * constraint_state.bw_Ju[i_c, i_b]
                y_i = constraint_state.bw_y[i_c, i_b]

                if qd.static(static_rigid_sim_config.sparse_solve):
                    for k in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                        i_d = constraint_state.jac_relevant_dofs[i_c, k, i_b]
                        constraint_state.dL_djac[i_c, i_d, i_b] += -(
                            constraint_state.bw_u[i_d, i_b] * y_i + constraint_state.qacc[i_d, i_b] * DJu_i
                        )
                else:
                    for i_d in range(n_dofs):
                        constraint_state.dL_djac[i_c, i_d, i_b] += -(
                            constraint_state.bw_u[i_d, i_b] * y_i + constraint_state.qacc[i_d, i_b] * DJu_i
                        )

        # M: -u * qacc^T
        n_entities = entities_info.n_links.shape[0]
        for i_e in range(n_entities):
            s = entities_info.dof_start[i_e]
            e = entities_info.dof_end[i_e]
            for i in range(s, e):
                for j in range(s, e):
                    val0 = -constraint_state.bw_u[i, i_b] * constraint_state.qacc[j, i_b]
                    val1 = -constraint_state.bw_u[j, i_b] * constraint_state.qacc[i, i_b]
                    constraint_state.dL_dM[i, j, i_b] += (val0 + val1) * 0.5  # symmetrize


@qd.kernel(fastcache=True)
def kernel_manual_add_joint_limit_constraints_bw(
    links_info: array_class.LinksInfo,
    joints_info: array_class.JointsInfo,
    dofs_info: array_class.DofsInfo,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    collider_state: array_class.ColliderState,
    static_rigid_sim_config: qd.template(),
    enable_collision: qd.template(),
):
    """Manual reverse of `add_joint_limit_constraints`.

    Consumes the upstream gradients populated by `kernel_compute_gradients`:
        constraint_state.dL_daref[n_con, i_b]      -- dL/d(aref)
        constraint_state.dL_defc_D[n_con, i_b]    -- dL/d(efc_D)
        (dL_djac is *not* used: jac entries are piecewise-constant ±1 w.r.t.
         the differentiable inputs, so the sub-gradient is 0.)

    Accumulates into:
        rigid_global_info.qpos.grad[i_q, i_b]
        dofs_state.vel.grad[i_d, i_b]

    Strategy: re-walk the same joint loop as the forward to deterministically
    recover `n_con` per active joint limit (no atomic_add, no `n_constraints`
    reset). Assumes the joint-limit constraints are the *first* group in
    `constraint_state.n_constraints` -- i.e. no equality / collision /
    frictionloss constraints precede them in the batch. This is sufficient
    for cartpole-style scenes; generalising requires stashing the starting
    offset during the forward pass.

    Chain rule (per active joint, `pos_delta < 0`):

        Forward:
            pos_delta_min = qpos[i_q] - limit_lo
            pos_delta_max = limit_hi - qpos[i_q]
            pos_delta     = min(pos_delta_min, pos_delta_max)
            sign          = +1 if pos_delta_min < pos_delta_max else -1
            jac_qvel      = sign * dofs_vel[i_d]
            imp, aref     = gu.imp_aref(sol_params, pos_delta, jac_qvel, pos_delta)
            diag_raw      = invweight * (1 - imp) / imp
            diag          = max(diag_raw, EPS)
            efc_D         = 1 / diag

        d(pos_delta) / d(qpos) = sign      (chosen branch of `min`)
        d(jac_qvel) / d(vel)   = sign

        dL/d(imp) = ga * d(aref)/d(imp) + gD * d(efc_D)/d(imp)
                  ga = dL_daref[n_con],  gD = dL_defc_D[n_con]

        dL/d(pos_delta) = ga * d(aref)/d(pos_delta)|_direct
                        + dL/d(imp) * d(imp)/d(imp_x) * d(imp_x)/d(pos_delta)

        dL/d(jac_qvel)  = ga * d(aref)/d(jac_qvel) = -ga * b_coef

        dL/d(qpos)      += sign * dL/d(pos_delta)
        dL/d(vel)       += sign * dL/d(jac_qvel)
    """
    EPS = rigid_global_info.EPS[None]
    _B = constraint_state.jac.shape[2]
    n_links = links_info.root_idx.shape[0]

    qd.loop_config(
        name="kernel_manual_add_joint_limit_constraints_bw",
        serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL),
    )
    for i_b in range(_B):
        # Collision constraints (4 rows per contact) are added before joint
        # limits in `add_inequality_constraints`, so offset the joint-limit row
        # counter past them when collision is on.
        n_con_counter = gs.qd_int(0)
        if qd.static(enable_collision):
            n_con_counter = gs.qd_int(collider_state.n_contacts[i_b] * 4)

        for i_l in range(n_links):
            I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l
            for i_j in range(links_info.joint_start[I_l], links_info.joint_end[I_l]):
                I_j = [i_j, i_b] if qd.static(static_rigid_sim_config.batch_joints_info) else i_j

                if joints_info.type[I_j] == gs.JOINT_TYPE.REVOLUTE or joints_info.type[I_j] == gs.JOINT_TYPE.PRISMATIC:
                    i_q = joints_info.q_start[I_j]
                    i_d = joints_info.dof_start[I_j]
                    I_d = [i_d, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else i_d

                    pos_delta_min = rigid_global_info.qpos[i_q, i_b] - dofs_info.limit[I_d][0]
                    pos_delta_max = dofs_info.limit[I_d][1] - rigid_global_info.qpos[i_q, i_b]
                    pos_delta = qd.min(pos_delta_min, pos_delta_max)

                    if pos_delta < 0:
                        n_con = n_con_counter
                        n_con_counter = n_con_counter + 1

                        # Replay forward intermediates (cheap, avoids stashing).
                        sign_pos = (pos_delta_min < pos_delta_max) * 2 - 1
                        sign_f = gs.qd_float(sign_pos)

                        sol_params = joints_info.sol_params[I_j]
                        timeconst = sol_params[0]
                        dampratio = sol_params[1]
                        dmin = sol_params[2]
                        dmax = sol_params[3]
                        width = sol_params[4]
                        mid = sol_params[5]
                        power = sol_params[6]

                        imp_x = qd.abs(pos_delta) / width
                        imp_a_coef = 1.0 / mid ** (power - 1.0)
                        imp_b_coef = 1.0 / (1.0 - mid) ** (power - 1.0)
                        imp_a = imp_a_coef * imp_x**power
                        imp_b = 1.0 - imp_b_coef * (1.0 - imp_x) ** power
                        imp_y = imp_a if imp_x < mid else imp_b
                        imp_raw = dmin + imp_y * (dmax - dmin)
                        imp_clamped = qd.math.clamp(imp_raw, dmin, dmax)
                        imp = dmax if imp_x > 1.0 else imp_clamped

                        b_coef = 2.0 / (dmax * timeconst)
                        k_coef = 1.0 / (dmax * dmax * timeconst * timeconst * dampratio * dampratio)

                        invweight = dofs_info.invweight[I_d]
                        diag_raw = invweight * (1.0 - imp) / imp
                        diag = qd.max(diag_raw, EPS)

                        # Upstream grads.
                        ga = constraint_state.dL_daref[n_con, i_b]
                        gD = constraint_state.dL_defc_D[n_con, i_b]

                        # --- Partials of forward outputs w.r.t. intermediates ---
                        # aref = -b_coef * jac_qvel - k_coef * imp * pos_delta
                        d_aref_d_imp = -k_coef * pos_delta
                        d_aref_d_jac_qvel = -b_coef
                        d_aref_d_pos_delta_direct = -k_coef * imp

                        # diag_raw = invweight*(1-imp)/imp  ⇒  d(diag_raw)/d(imp) = -invweight/imp^2
                        # diag = max(diag_raw, EPS); efc_D = 1/diag
                        # d(efc_D)/d(imp) = -1/diag^2 · d(diag)/d(imp), 0 if clamped to EPS
                        d_diag_d_imp = gs.qd_float(0.0)
                        if diag_raw > EPS:
                            d_diag_d_imp = -invweight / (imp * imp)
                        d_efc_D_d_imp = -d_diag_d_imp / (diag * diag)

                        # d(imp)/d(imp_x): active only inside the smooth clamp band.
                        within_clamp = (imp_raw > dmin) and (imp_raw < dmax) and (imp_x <= 1.0)
                        d_imp_y_d_imp_x = gs.qd_float(0.0)
                        if imp_x < mid:
                            d_imp_y_d_imp_x = power * imp_a_coef * imp_x ** (power - 1.0)
                        else:
                            d_imp_y_d_imp_x = power * imp_b_coef * (1.0 - imp_x) ** (power - 1.0)
                        d_imp_d_imp_x = gs.qd_float(0.0)
                        if within_clamp:
                            d_imp_d_imp_x = (dmax - dmin) * d_imp_y_d_imp_x

                        # d(imp_x)/d(pos_delta) = sign(pos_delta)/width; pos_delta < 0 ⇒ -1/width
                        d_imp_x_d_pos_delta = -1.0 / width
                        d_imp_d_pos_delta = d_imp_d_imp_x * d_imp_x_d_pos_delta

                        # --- Combine ---
                        dL_d_imp = ga * d_aref_d_imp + gD * d_efc_D_d_imp
                        dL_d_pos_delta = ga * d_aref_d_pos_delta_direct + dL_d_imp * d_imp_d_pos_delta
                        dL_d_jac_qvel = ga * d_aref_d_jac_qvel

                        # --- Propagate ---
                        rigid_global_info.qpos.grad[i_q, i_b] += sign_f * dL_d_pos_delta
                        dofs_state.vel.grad[i_d, i_b] += sign_f * dL_d_jac_qvel


@qd.kernel(fastcache=True)
def kernel_manual_add_collision_constraints_bw(
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    collider_state: array_class.ColliderState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Manual reverse of `add_collision_constraints`.

    Consumes the upstream gradients populated by `kernel_compute_gradients`
    (`constraint_state.dL_djac / dL_daref / dL_defc_D`) and produces the
    gradients of the loss w.r.t. the collision constraint's differentiable
    inputs:
        collider_state.contact_data.{pos, normal, penetration}.grad   (-> collider.backward)
        dofs_state.{cdof_ang, cdof_vel, vel}.grad
        links_state.root_COM.grad
    (cdof / root_COM / vel grads are consumed by the existing COM / forward-
    dynamics reverse chain; contact_data grads feed `collider.backward`.)

    Assumes collision is the FIRST constraint group (so `n_con = i_col*4 + i`),
    which holds because `add_inequality_constraints` calls
    `add_collision_constraints` before `add_joint_limit_constraints`.

    Forward recap (per contact `i_col`, per friction-pyramid row `i` in 0..3):
        d1, d2 = qd_orthogonals(normal);  d = s_i * (d1 if i<2 else d2),  s_i = 2*(i%2)-1
        n      = d * friction - normal
        jac[n_con, i_d] = sum_chain (sign * vel_motion(i_d)) . n
            vel_motion = cdof_vel - t_pos x cdof_ang,  t_pos = contact_pos - root_COM[link]
        jac_qvel = sum_chain jac[n_con, i_d] * dofs_vel[i_d]
        imp, aref = imp_aref(sol_params, -penetration, jac_qvel, -penetration)
        diag = (invweight + friction^2 invweight) * 2 friction^2 (1-imp)/imp ; efc_D = 1/diag
    """
    EPS = rigid_global_info.EPS[None]
    _B = dofs_state.ctrl_mode.shape[1]
    n_dofs = dofs_state.ctrl_mode.shape[0]
    max_contact_pairs = collider_state.contact_data.link_a.shape[0]

    qd.loop_config(
        name="kernel_manual_add_collision_constraints_bw",
        serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL),
    )
    for flat_idx in range(max_contact_pairs * _B):
        i_b = flat_idx % _B
        i_col = flat_idx // _B
        if i_col < collider_state.n_contacts[i_b]:
            link_a = collider_state.contact_data.link_a[i_col, i_b]
            link_b = collider_state.contact_data.link_b[i_col, i_b]
            contact_pos = collider_state.contact_data.pos[i_col, i_b]
            normal = collider_state.contact_data.normal[i_col, i_b]
            friction = collider_state.contact_data.friction[i_col, i_b]
            sol_params = collider_state.contact_data.sol_params[i_col, i_b]
            penetration = collider_state.contact_data.penetration[i_col, i_b]

            link_a_maybe_batch = [link_a, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else link_a
            invweight = links_info.invweight[link_a_maybe_batch][0]
            if link_b > -1:
                link_b_maybe_batch = [link_b, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else link_b
                invweight = invweight + links_info.invweight[link_b_maybe_batch][0]

            # --- forward intermediates of qd_orthogonals(normal) ---
            #   b_raw branches on |normal[1]| < 0.5; b = normalize(b_raw)
            #   d1 = b x normal, d2 = b
            n0, n1, n2 = normal[0], normal[1], normal[2]
            branch_a = qd.abs(n1) < 0.5
            b_raw = gs.qd_vec3(0.0, 0.0, 0.0)
            if branch_a:
                b_raw = gs.qd_vec3(-n0 * n1, 1.0 - n1 * n1, -n2 * n1)
            else:
                b_raw = gs.qd_vec3(-n0 * n2, -n1 * n2, 1.0 - n2 * n2)
            b_raw_norm = b_raw.norm()
            b = b_raw / b_raw_norm
            d1 = b.cross(normal)
            d2 = b

            sol_timeconst = sol_params[0]
            sol_dampratio = sol_params[1]
            sol_dmin = sol_params[2]
            sol_dmax = sol_params[3]
            sol_width = sol_params[4]
            sol_mid = sol_params[5]
            sol_power = sol_params[6]

            neg_pen = -penetration
            imp_x = qd.abs(neg_pen) / sol_width
            # d(imp_x)/d(penetration) = -sign(neg_pen)/width
            sign_neg = gs.qd_float(1.0) if neg_pen >= 0 else gs.qd_float(-1.0)
            d_imp_x_d_pen = -sign_neg / sol_width

            imp_a_coef = 1.0 / sol_mid ** (sol_power - 1.0)
            imp_b_coef = 1.0 / (1.0 - sol_mid) ** (sol_power - 1.0)
            imp_a = imp_a_coef * imp_x**sol_power
            imp_b = 1.0 - imp_b_coef * (1.0 - imp_x) ** sol_power
            imp_y = imp_a if imp_x < sol_mid else imp_b
            imp_raw = sol_dmin + imp_y * (sol_dmax - sol_dmin)
            imp_clamped = qd.math.clamp(imp_raw, sol_dmin, sol_dmax)
            imp = sol_dmax if imp_x > 1.0 else imp_clamped

            b_coef = 2.0 / (sol_dmax * sol_timeconst)
            # k_coef matches gu.imp_aref's k = 1/(dmax^2 timeconst^2 dampratio^2)
            k_coef = 1.0 / (sol_dmax * sol_dmax * sol_timeconst * sol_timeconst * sol_dampratio * sol_dampratio)

            # diag = C0 * (1-imp)/imp, C0 = 2 friction^2 invweight (1 + friction^2)
            C0 = (invweight + friction * friction * invweight) * 2.0 * friction * friction
            diag_raw = C0 * (1.0 - imp) / imp
            diag = qd.max(diag_raw, EPS)

            within_clamp = (imp_raw > sol_dmin) and (imp_raw < sol_dmax) and (imp_x <= 1.0)
            d_imp_y_d_imp_x = gs.qd_float(0.0)
            if imp_x < sol_mid:
                d_imp_y_d_imp_x = sol_power * imp_a_coef * imp_x ** (sol_power - 1.0)
            else:
                d_imp_y_d_imp_x = sol_power * imp_b_coef * (1.0 - imp_x) ** (sol_power - 1.0)
            d_imp_d_imp_x = gs.qd_float(0.0)
            if within_clamp:
                d_imp_d_imp_x = (sol_dmax - sol_dmin) * d_imp_y_d_imp_x

            d_diag_d_imp = gs.qd_float(0.0)
            if diag_raw > EPS:
                d_diag_d_imp = -C0 / (imp * imp)
            d_efc_D_d_imp = -d_diag_d_imp / (diag * diag)

            # Accumulators for this contact's differentiable inputs.
            g_pos = gs.qd_vec3(0.0, 0.0, 0.0)
            g_normal = gs.qd_vec3(0.0, 0.0, 0.0)
            g_pen = gs.qd_float(0.0)
            g_d1 = gs.qd_vec3(0.0, 0.0, 0.0)
            g_d2 = gs.qd_vec3(0.0, 0.0, 0.0)

            for i in range(4):
                s_i = gs.qd_float(2 * (i % 2) - 1)
                d = s_i * d1 if i < 2 else s_i * d2
                n = d * friction - normal
                n_con = i_col * 4 + i

                ga = constraint_state.dL_daref[n_con, i_b]
                gD = constraint_state.dL_defc_D[n_con, i_b]

                # aref = -b_coef*jac_qvel + k_coef*imp*penetration  (pos arg = -penetration)
                d_aref_d_imp = k_coef * penetration
                d_aref_d_pen_direct = k_coef * imp
                d_aref_d_jac_qvel = -b_coef

                dL_d_imp = ga * d_aref_d_imp + gD * d_efc_D_d_imp
                dL_d_pen = ga * d_aref_d_pen_direct + dL_d_imp * d_imp_d_imp_x * d_imp_x_d_pen
                g_pen += dL_d_pen
                dL_d_jac_qvel = ga * d_aref_d_jac_qvel

                # Reverse jac[n_con, i_d] over the kinematic chain.
                dL_dn = gs.qd_vec3(0.0, 0.0, 0.0)
                for i_ab in range(2):
                    sign = gs.qd_float(-1.0)
                    link = link_a
                    if i_ab == 1:
                        sign = gs.qd_float(1.0)
                        link = link_b
                    while link > -1:
                        link_mb = [link, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else link
                        for i_d_ in range(links_info.n_dofs[link_mb]):
                            i_d = links_info.dof_end[link_mb] - 1 - i_d_

                            cdof_ang = dofs_state.cdof_ang[i_d, i_b]
                            cdof_vel = dofs_state.cdof_vel[i_d, i_b]
                            t_pos = contact_pos - links_state.root_COM[link, i_b]
                            vel_motion = cdof_vel - t_pos.cross(cdof_ang)

                            jac_stored = constraint_state.jac[n_con, i_d, i_b]
                            g_jac = constraint_state.dL_djac[n_con, i_d, i_b] + dL_d_jac_qvel * dofs_state.vel[i_d, i_b]
                            dofs_state.vel.grad[i_d, i_b] += dL_d_jac_qvel * jac_stored

                            # jac_contrib = (sign * vel_motion) . n
                            dL_dn += g_jac * sign * vel_motion
                            g_vm = g_jac * sign * n  # dL/d(vel_motion)

                            # vel_motion = cdof_vel - t_pos x cdof_ang
                            dofs_state.cdof_vel.grad[i_d, i_b] += g_vm
                            dofs_state.cdof_ang.grad[i_d, i_b] += t_pos.cross(g_vm)
                            dt = -(cdof_ang.cross(g_vm))  # dL/d(t_pos)
                            g_pos += dt
                            links_state.root_COM.grad[link, i_b] += -dt

                        link = links_info.parent_idx[link_mb]

                # n = d*friction - normal
                g_normal += -dL_dn
                g_d = dL_dn * friction
                if i < 2:
                    g_d1 += s_i * g_d
                else:
                    g_d2 += s_i * g_d

            # Reverse qd_orthogonals: d1 = b x normal, d2 = b, b = normalize(b_raw(normal)).
            dL_db = g_d2 + normal.cross(g_d1)  # d2=b ; d1=b x normal -> dL/db += normal x g_d1
            g_normal += g_d1.cross(b)  # d1=b x normal -> dL/d(normal) += g_d1 x b
            # b = b_raw / |b_raw|
            dL_db_raw = (dL_db - dL_db.dot(b) * b) / b_raw_norm
            # b_raw(normal) branch Jacobian
            if branch_a:
                # b_raw = (-n0 n1, 1 - n1^2, -n2 n1)
                g_normal[0] += dL_db_raw[0] * (-n1)
                g_normal[1] += dL_db_raw[0] * (-n0) + dL_db_raw[1] * (-2.0 * n1) + dL_db_raw[2] * (-n2)
                g_normal[2] += dL_db_raw[2] * (-n1)
            else:
                # b_raw = (-n0 n2, -n1 n2, 1 - n2^2)
                g_normal[0] += dL_db_raw[0] * (-n2)
                g_normal[1] += dL_db_raw[1] * (-n2)
                g_normal[2] += dL_db_raw[0] * (-n0) + dL_db_raw[1] * (-n1) + dL_db_raw[2] * (-2.0 * n2)

            for j in qd.static(range(3)):
                collider_state.contact_data.pos.grad[i_col, i_b][j] = g_pos[j]
                collider_state.contact_data.normal.grad[i_col, i_b][j] = g_normal[j]
            collider_state.contact_data.penetration.grad[i_col, i_b] = g_pen
