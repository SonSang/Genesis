"""Forward perturbation: vs analytical prediction.

For J4 N=1 seed=1001, perturb ctrl_force[d=4] by ±eps and read out the
*intermediate forward state* (links_state.pos) after 1 step. Compare
δlinks_pos[1][2] (arm z) to our analytical prediction:

    δpos_chassis[2] ≈ dt² · M⁻¹[2,4] · ε = 6.897e-5 · ε
    δpos_arm[2]     ≈ -7.74e-5 · ε

If signs match analytical -> backward chain is wrong. If signs mismatch
analytical (e.g., +7.74e-5 instead of -7.74e-5) -> forward analysis is
wrong; backward chain may actually be consistent with the actual forward.

The mismatch sign tells us which side of the analysis to revisit.
"""

import sys
import numpy as np
import genesis as gs

sys.path.insert(0, "notes")


def main():
    gs.init(precision="64", backend=gs.cpu, performance_mode=False, logging_level="warning")
    from diag_multistep_worst_case import TOPOLOGIES, build, loss_fn

    name_map = {t[0]: t for t in TOPOLOGIES}
    _, mjcf, n_dofs = name_map["J4_free_rev"]

    seed = 1001
    rng = np.random.default_rng(seed)
    u_base = rng.normal(size=n_dofs) * 0.3
    target_d = 4

    sb, rb = build(mjcf, False)

    def fwd_and_read(u):
        sb.reset()
        rb.control_dofs_force(gs.tensor(u, dtype=gs.tc_float))
        sb.step()
        # read out links_state.pos and qpos after step
        from genesis.utils.misc import qd_to_torch

        lp = qd_to_torch(sb.rigid_solver.links_state.pos, copy=True).detach().cpu().numpy()
        qpos = qd_to_torch(sb.rigid_solver._rigid_global_info.qpos, copy=True).detach().cpu().numpy()
        return lp, qpos

    eps = 1e-5

    lp_0, qpos_0 = fwd_and_read(u_base)
    u_plus = u_base.copy()
    u_plus[target_d] += eps
    lp_p, qpos_p = fwd_and_read(u_plus)
    u_minus = u_base.copy()
    u_minus[target_d] -= eps
    lp_m, qpos_m = fwd_and_read(u_minus)

    print(f"J4 N=1 seed={seed}, forward perturb at ctrl_force[d={target_d}], eps={eps}\n")
    print("links_state.pos (shape [n_links, n_envs, 3], squeezed [n_links, 3]):")
    print(f"  baseline = \n{lp_0.squeeze(1)}")
    print(f"  +eps     = \n{lp_p.squeeze(1)}")
    print(f"  -eps     = \n{lp_m.squeeze(1)}\n")

    delta_p = (lp_p.squeeze(1) - lp_0.squeeze(1)) / eps  # finite diff in +eps direction
    delta_central = (lp_p.squeeze(1) - lp_m.squeeze(1)) / (2 * eps)
    print(f"d(links_pos)/d(u[{target_d}])  forward diff  (lp_p - lp_0)/eps:")
    print(delta_p)
    print(f"\nd(links_pos)/d(u[{target_d}])  central diff  (lp_p - lp_m)/(2eps):")
    print(delta_central)

    # Predicted from our analytical chain:
    M_inv_4 = np.array([0.0, 0.0, 0.6897, 0.0, 7.3276, 0.0, -1.2931])
    dt = 0.01
    # δpos_chassis = dt² · M_inv[0:3, 4] · 1
    pred_pos_chassis = dt**2 * M_inv_4[0:3]
    # δang_y = dt² · M_inv[4, 4]
    delta_ang_y = dt**2 * M_inv_4[4]
    # δqy = δang/2
    delta_qy = delta_ang_y / 2
    # δR·arm_local 의 z component = 0.2 · (-2·δqy) (from quat-to-R formula)
    delta_R_arm_z = 0.2 * (-2 * delta_qy)
    pred_pos_arm = pred_pos_chassis.copy()
    pred_pos_arm[2] += delta_R_arm_z

    print("\n--- Analytical prediction ---")
    print(f"δpos_chassis (3,) = {pred_pos_chassis}")
    print(f"δpos_arm (3,)     = {pred_pos_arm}")
    print(f"\nFor reference: dump's analytical chain implied δpos_chassis[2]= {pred_pos_chassis[2]:.3e}")
    print(f"                                              δpos_arm[2]    = {pred_pos_arm[2]:.3e}")
    print("\n=> if actual δpos_arm[2] (from forward) has same sign as analytical (-7.74e-5),")
    print("   then backward chain is faulty.")
    print("=> if actual has OPPOSITE sign (+7.74e-5), then our forward analysis is wrong.")


if __name__ == "__main__":
    main()
