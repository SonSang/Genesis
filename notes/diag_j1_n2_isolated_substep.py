"""Isolate the second backward substep (t=0) from the cross-substep chain
to determine whether our single-substep backward itself is accurate
or whether cross-substep state pollution corrupts the result.

Procedure:
  Scene A — standard:
    1. reset
    2. apply u_0 → step()  (substep t=0 forward)
    3. apply u_1 → step()  (substep t=1 forward)
    4. loss = (links_pos**2).sum()
    5. loss.backward()  → records ana_A[t=0] = u_0.grad
       During backward, monkey-patched solver.substep_pre_coupling_grad
       captures qpos.grad / vel.grad immediately after the FIRST substep_pre_
       coupling_grad call (= after t=1 backward done). That snapshot is the
       'upstream' grad arriving at step_0's output (qpos_1, vel_1).

  Scene B — isolated 1-substep backward seeded by captured upstream:
    1. reset
    2. apply u_0 → step()  (substep t=0 forward only)
    3. loss_iso = (qpos * captured_qpos_grad).sum() + (vel * captured_vel_grad).sum()
       (qpos / vel here are GsTensors from scene_B.get_state())
    4. loss_iso.backward()  → records ana_B[t=0] = u_0.grad

  Compare ana_A[t=0] vs ana_B[t=0] vs fd[t=0].

If ana_A ≈ ana_B but both differ from fd:
  Our single-substep backward is consistent (matches the upstream seed)
  but the upstream seed itself (captured from Scene A) is wrong → some
  field's grad in the chain at the t=1 backward end is off.

If ana_A differs from ana_B:
  The cross-substep state pollution corrupts Scene A's t=0 substep
  backward — i.e. running the same backward with the same upstream
  seed gives a different answer when sandwiched between t=1's backward
  vs. run in isolation.

If ana_B ≈ fd[t=0]:
  Our single-substep backward is correct. cross-substep is the bug.
"""

import os
import sys
import numpy as np

sys.path.insert(0, "notes")
from diag_multistep_worst_case import TOPOLOGIES, build, loss_fn
from genesis.utils.misc import qd_to_torch


def make_capture_patch(solver, captures):
    """Hook into _debug_grad_dump to capture qpos/vel.grad at the exact
    "after post-update_cartesian_space.grad" moment (mid-substep, before
    step_2.grad runs). This is the moment the user pointed out where
    qpos.grad / vel.grad should be the ONLY non-zero grads (FK chain done,
    further chain not yet started)."""
    orig = solver._debug_grad_dump

    def patched(tag):
        orig(tag)
        if "after post-update_cartesian_space.grad" in tag:
            qg = qd_to_torch(solver._rigid_global_info.qpos.grad, copy=True).clone()
            vg = qd_to_torch(solver.dofs_state.vel.grad, copy=True).clone()
            captures.append({"tag": tag, "qpos_grad": qg, "vel_grad": vg})

    solver._debug_grad_dump = patched
    return orig


def main():
    import genesis as gs

    gs.init(precision="64", backend=gs.cpu, performance_mode=False, logging_level="warning")
    from genesis.utils.misc import qd_to_torch

    name_map = {t[0]: t for t in TOPOLOGIES}
    _, mjcf, n_dofs = name_map["J1_free"]

    seed = 1000
    rng = np.random.default_rng(seed)
    N = 2
    u_list = [rng.normal(size=n_dofs) * 0.3 for _ in range(N)]

    # =================== Scene A: standard N=2 backward ===================
    sa, ra = build(mjcf, True)
    captures = []
    orig = make_capture_patch(sa.rigid_solver, captures)

    u_as = [gs.tensor(u, dtype=gs.tc_float, requires_grad=True) for u in u_list]
    sa.reset()
    for t in range(N):
        ra.control_dofs_force(u_as[t])
        sa.step()
    loss_fn(sa).backward()
    ana_A = np.array([u.grad.detach().cpu().numpy() for u in u_as])
    sa.rigid_solver.substep_pre_coupling_grad = orig  # restore

    print(f"# captures ({len(captures)} post-UCS.grad dumps):")
    for i, c in enumerate(captures):
        qg = c["qpos_grad"]
        vg = c["vel_grad"]
        qg_max = float(qg.abs().max()) if qg is not None else float("nan")
        vg_max = float(vg.abs().max()) if vg is not None else float("nan")
        print(f"  [{i}] {c['tag']}: qpos.grad max={qg_max:.3e}  vel.grad max={vg_max:.3e}")

    print()
    print(f"Scene A ana[t=0] = {ana_A[0]}")
    print(f"Scene A ana[t=1] = {ana_A[1]}")

    # The "upstream" arriving at step_0's output = .grad state RIGHT AFTER
    # the first substep_pre_coupling_grad call (= the t=N-1=1 substep)
    # has finished but BEFORE t=0 substep backward starts.
    # captures[0] = first post-UCS.grad call = t=N-1's UCS.grad done.
    # This is the upstream gradient arriving at step_0's output (qpos_1, vel_1).
    upstream = captures[0]
    print()
    print(f"Upstream (captured at {upstream['tag']}):")
    print(f"  qpos.grad = {upstream['qpos_grad'].detach().cpu().numpy()}")
    print(f"  vel.grad  = {upstream['vel_grad'].detach().cpu().numpy()}")

    # =================== Scene B: isolated 1-substep backward ===================
    # N=1 forward (step_0 only). For the backward, monkey-patch the rigid
    # solver's `_debug_grad_dump` so that at the
    # "after post-update_cartesian_space.grad" moment in the substep_pre_
    # coupling_grad of the t=0 backward, qpos.grad and vel.grad are FORCIBLY
    # overwritten with the values captured from Scene A's t=1 post-UCS.grad
    # moment. Semantically: skip the FK chain (forward_velocity.grad /
    # COM_links.grad / UCS.grad) and substitute pre-captured upstream grads;
    # the rest of substep_pre_coupling_grad (step_2.grad / manual_compute_qacc_bw
    # / fwd_dynamics_without_qacc.grad) runs as usual.
    sb, rb = build(mjcf, True)
    u_bs = [gs.tensor(u_list[0], dtype=gs.tc_float, requires_grad=True)]
    sb.reset()
    rb.control_dofs_force(u_bs[0])
    sb.step()

    # Install override hook
    upstream_q_torch = upstream["qpos_grad"].clone()
    upstream_v_torch = upstream["vel_grad"].clone()

    orig_dump_b = sb.rigid_solver._debug_grad_dump
    override_count = {"n": 0}

    def override_dump(tag):
        orig_dump_b(tag)
        if "after post-update_cartesian_space.grad" in tag:
            # Overwrite qpos.grad and vel.grad with captured upstream
            qpos_grad_view = qd_to_torch(sb.rigid_solver._rigid_global_info.qpos.grad, copy=False)
            vel_grad_view = qd_to_torch(sb.rigid_solver.dofs_state.vel.grad, copy=False)
            qpos_grad_view.copy_(upstream_q_torch.to(qpos_grad_view))
            vel_grad_view.copy_(upstream_v_torch.to(vel_grad_view))
            override_count["n"] += 1

    sb.rigid_solver._debug_grad_dump = override_dump
    # Force GENESIS_DEBUG_GRAD path so the hook fires (no-op without env var
    # since _debug_grad_dump early-returns)
    os.environ["GENESIS_DEBUG_GRAD"] = "1"

    # Build a dummy loss that triggers backward — content doesn't matter since
    # qpos.grad/vel.grad will be overwritten by the hook anyway.
    state = sb.get_state()
    ss = state.solvers_state[sb.solvers.index(sb.rigid_solver)]
    loss_iso = ss.qpos.sum() + ss.dofs_vel.sum()
    loss_iso.backward()
    sb.rigid_solver._debug_grad_dump = orig_dump_b
    print(f"  override fired {override_count['n']} time(s)")
    ana_B = u_bs[0].grad.detach().cpu().numpy()

    print()
    print(f"Scene B isolated ana[t=0] = {ana_B}")

    # =================== Comparison ===================
    diff_AB = ana_A[0] - ana_B
    print()
    print("Comparison ana_A[t=0] vs ana_B (isolated):")
    print(f"  ana_A[t=0]   = {ana_A[0]}")
    print(f"  ana_B        = {ana_B}")
    print(f"  diff (A - B) = {diff_AB}")
    print(f"  max|diff|    = {float(np.abs(diff_AB).max()):.3e}")

    # =================== FD baseline ===================
    # also do FD on scene A's loss vs u_0 to know the ground truth
    print()
    print("Computing FD on u_0...")
    sc, rc = build(mjcf, False)
    eps = 1e-5
    fd_t0 = np.zeros(n_dofs)
    for d in range(n_dofs):
        sc.reset()
        for t2 in range(N):
            inp = u_list[t2].copy()
            if t2 == 0:
                inp[d] += eps
            rc.control_dofs_force(gs.tensor(inp, dtype=gs.tc_float))
            sc.step()
        lp = float(loss_fn(sc).detach().cpu())
        sc.reset()
        for t2 in range(N):
            inp = u_list[t2].copy()
            if t2 == 0:
                inp[d] -= eps
            rc.control_dofs_force(gs.tensor(inp, dtype=gs.tc_float))
            sc.step()
        lm = float(loss_fn(sc).detach().cpu())
        fd_t0[d] = (lp - lm) / (2 * eps)
    print(f"  FD[t=0]      = {fd_t0}")
    print(f"  ana_A - FD   = {ana_A[0] - fd_t0}, max|diff| = {float(np.abs(ana_A[0] - fd_t0).max()):.3e}")
    print(f"  ana_B - FD   = {ana_B - fd_t0},   max|diff| = {float(np.abs(ana_B - fd_t0).max()):.3e}")


if __name__ == "__main__":
    main()
