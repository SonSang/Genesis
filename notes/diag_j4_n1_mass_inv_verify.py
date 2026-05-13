"""Verify the `force.grad = mass^{-1} · acc.grad` chain in compute_qacc.grad.

Steps:
  1. Run J4 N=1 seed=1001 forward + backward.
  2. After backward, read out:
     - mass_mat  (forward primal)
     - mass_mat_L, mass_mat_D_inv  (LDLT factors)
     - dofs_state.acc.grad  (input to compute_qacc.grad's reverse)
     - dofs_state.force.grad  (output)
     - dofs_state.ctrl_force.grad  (final endpoint, should equal force.grad)
  3. Independently compute expected_force_grad = inv(mass_mat) @ acc.grad
     via numpy, and compare to the dumped force.grad.

This isolates whether the compute_qacc.grad stage is numerically faithful
or introduces error in the `mass^{-1}` apply.
"""

import os
import sys
import tempfile

import numpy as np

import genesis as gs

sys.path.insert(0, "notes")


def build(MJCF, requires_grad):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, 0), requires_grad=requires_grad),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            enable_self_collision=False,
            enable_joint_limit=False,
            disable_constraint=True,
            use_hibernation=False,
            use_contact_island=False,
        ),
        show_viewer=False,
    )
    fd, p = tempfile.mkstemp(suffix=".xml")
    with os.fdopen(fd, "w") as fh:
        fh.write(MJCF)
    robot = scene.add_entity(gs.morphs.MJCF(file=p))
    scene.build(n_envs=0)
    return scene, robot


def loss_fn(scene):
    state = scene.get_state()
    ss = state.solvers_state[scene.solvers.index(scene.rigid_solver)]
    return (ss.links_pos.reshape(-1) ** 2).sum()


def main():
    gs.init(precision="64", backend=gs.cpu, performance_mode=False, logging_level="warning")
    from diag_multistep_worst_case import MJCF_J4
    from genesis.utils.misc import qd_to_torch

    seed = 1001
    rng = np.random.default_rng(seed)
    n_dofs = 7
    u = rng.normal(size=n_dofs) * 0.3
    u_tensor = gs.tensor(u, dtype=gs.tc_float, requires_grad=True)

    scene, robot = build(MJCF_J4, True)
    scene.reset()
    robot.control_dofs_force(u_tensor)
    scene.step()
    loss = loss_fn(scene)
    loss.backward()

    solver = scene.rigid_solver

    # Read forward primal mass_mat (post-backward, but it's set by
    # prepare_backward_substep to the forward value).
    mass_mat = qd_to_torch(solver._rigid_global_info.mass_mat, copy=True).detach().cpu().numpy()
    mass_mat_L = qd_to_torch(solver._rigid_global_info.mass_mat_L, copy=True).detach().cpu().numpy()
    mass_mat_D_inv = qd_to_torch(solver._rigid_global_info.mass_mat_D_inv, copy=True).detach().cpu().numpy()
    # Squeeze the trailing batch=1 axis. mass_mat shape [7, 7, 1] -> [7, 7]
    M = mass_mat[..., 0]
    L = mass_mat_L[..., 0]
    D_inv = mass_mat_D_inv[..., 0]

    np.set_printoptions(precision=4, suppress=True, linewidth=120)

    print("=" * 80)
    print("Forward primals (after prepare_backward_substep)")
    print("=" * 80)
    print(f"\nmass_mat  shape={M.shape}")
    print(M)
    print(f"\nmass_mat_L  shape={L.shape}  (Cholesky-Banachiewicz L factor)")
    print(L)
    print(f"\nmass_mat_D_inv  shape={D_inv.shape}")
    print(D_inv)

    # Reconstruct M from LDL^T (sanity check the factors)
    D = np.diag(1.0 / D_inv)
    M_reconstructed = L @ D @ L.T
    print("\nL · diag(1/D_inv) · L^T  (should equal mass_mat):")
    print(M_reconstructed)
    print(f"\n|| L·D·L^T - mass_mat ||_inf = {np.abs(M_reconstructed - M).max():.3e}")

    # Direct inverse
    M_inv_direct = np.linalg.inv(M)
    # LDLT inverse: M^{-1} = L^{-T} · diag(D_inv) · L^{-1}
    L_inv = np.linalg.inv(L)
    M_inv_ldlt = L_inv.T @ np.diag(D_inv) @ L_inv

    print("\nmass_mat^{-1} (direct np.linalg.inv):")
    print(M_inv_direct)
    print("\nmass_mat^{-1} (via LDLT: L^{-T} · diag(D_inv) · L^{-1}):")
    print(M_inv_ldlt)
    print(f"\n|| M_inv_direct - M_inv_ldlt ||_inf = {np.abs(M_inv_direct - M_inv_ldlt).max():.3e}")

    # Read the relevant .grad fields after backward
    acc_grad = qd_to_torch(solver.dofs_state.acc.grad, copy=True).detach().cpu().numpy()[..., 0]
    force_grad = qd_to_torch(solver.dofs_state.force.grad, copy=True).detach().cpu().numpy()[..., 0]
    ctrl_force_grad = qd_to_torch(solver.dofs_state.ctrl_force.grad, copy=True).detach().cpu().numpy()[..., 0]
    u_grad = u_tensor.grad.detach().cpu().numpy()

    print("\n" + "=" * 80)
    print("Backward grads (post-backward, before any per-substep zeroing)")
    print("=" * 80)
    # Note: acc.grad gets ZEROED inside the backward as part of begin_backward_substep,
    # so by the time we read it here it is 0. Same for force.grad. ctrl_force.grad
    # is the one that survives because process_input_grad reads it. Show whatever
    # we can.
    print(f"\nacc.grad        = {acc_grad}")
    print(f"force.grad      = {force_grad}")
    print(f"ctrl_force.grad = {ctrl_force_grad}")
    print(f"u_tensor.grad   = {u_grad}")

    # The values we actually want to verify are the ones from the dump:
    # acc.grad at "step_2.grad" stage and force.grad at "compute_qacc.grad" stage.
    # Take them from the captured dump text as ground truth.
    print("\n" + "=" * 80)
    print("Manual verification — using dumped values from notes/diag_j4_n1_grad_dump_full.txt")
    print("=" * 80)
    print("(from `f=0 after step_2.grad`)")
    acc_grad_dumped = np.array([4.001e-05, -1.081e-09, -7.755e-09, 0.0, -6.655e-10, -4.104e-10, 0.0])
    print(f"acc.grad      = {acc_grad_dumped}")

    expected_force_grad = M_inv_direct @ acc_grad_dumped
    print("\nexpected force.grad = mass_mat^{-1} @ acc.grad =")
    print(f"  {expected_force_grad}")

    print("\n(from `f=0 after compute_qacc.grad`)")
    force_grad_dumped = np.array([2.667e-05, np.nan, np.nan, 0.0, np.nan, np.nan, np.nan])
    print(f"force.grad (dumped, partial) = {force_grad_dumped}")
    print("  (we only logged d=0 magnitude; need full dump for the rest)")


if __name__ == "__main__":
    main()
