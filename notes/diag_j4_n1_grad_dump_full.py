"""Like diag_j4_grad_dump.py but dumps the FULL per-element .grad of every
field at every stage (not just max/norm or the verbose-set subset).
Writes the dump to `notes/diag_j4_n1_grad_dump_full.txt`."""

import os
import sys
import tempfile

import numpy as np

import genesis as gs

# Monkey-patch _debug_grad_dump to print FULL per-element grad for every field.
sys.path.insert(0, "notes")


def _full_dump(self, tag):
    from genesis.utils.misc import qd_to_torch

    level = os.environ.get("GENESIS_DEBUG_GRAD", "0")
    if level not in ("1", "2"):
        return
    fields = [
        ("rigid_global_info.qpos", self._rigid_global_info.qpos),
        ("dofs_state.ctrl_force", self.dofs_state.ctrl_force),
        ("dofs_state.vel", self.dofs_state.vel),
        ("dofs_state.pos", self.dofs_state.pos),
        ("dofs_state.acc", self.dofs_state.acc),
        ("dofs_state.acc_smooth", self.dofs_state.acc_smooth),
        ("dofs_state.acc_smooth_bw", self.dofs_state.acc_smooth_bw),
        ("dofs_state.force", self.dofs_state.force),
        ("dofs_state.qf_bias", self.dofs_state.qf_bias),
        ("dofs_state.qf_smooth", self.dofs_state.qf_smooth),
        ("dofs_state.qf_passive", self.dofs_state.qf_passive),
        ("dofs_state.qf_applied", self.dofs_state.qf_applied),
        ("links_state.pos", self.links_state.pos),
        ("links_state.quat", self.links_state.quat),
        ("links_state.cd_vel", self.links_state.cd_vel),
        ("links_state.cd_ang", self.links_state.cd_ang),
        ("links_state.cdd_vel", self.links_state.cdd_vel),
        ("links_state.cdd_ang", self.links_state.cdd_ang),
        ("links_state.cfrc_vel", self.links_state.cfrc_vel),
        ("links_state.cfrc_ang", self.links_state.cfrc_ang),
        ("dofs_state.cdofd_vel", self.dofs_state.cdofd_vel),
        ("dofs_state.cdofd_ang", self.dofs_state.cdofd_ang),
        ("dofs_state.cdof_vel", self.dofs_state.cdof_vel),
        ("dofs_state.cdof_ang", self.dofs_state.cdof_ang),
        ("links_state.cinr_inertial", self.links_state.cinr_inertial),
        ("links_state.cinr_pos", self.links_state.cinr_pos),
        ("links_state.cinr_mass", self.links_state.cinr_mass),
        ("links_state.crb_inertial", self.links_state.crb_inertial),
        ("links_state.crb_pos", self.links_state.crb_pos),
        ("links_state.crb_mass", self.links_state.crb_mass),
        ("links_state.cfrc_applied_ang", self.links_state.cfrc_applied_ang),
        ("links_state.cfrc_applied_vel", self.links_state.cfrc_applied_vel),
        ("links_state.cfrc_coupling_ang", self.links_state.cfrc_coupling_ang),
        ("links_state.cfrc_coupling_vel", self.links_state.cfrc_coupling_vel),
        ("links_state.i_pos", self.links_state.i_pos),
        ("links_state.i_quat", self.links_state.i_quat),
        ("joints_state.xanchor", self.joints_state.xanchor),
        ("joints_state.xaxis", self.joints_state.xaxis),
        ("rigid_global_info.mass_mat", self._rigid_global_info.mass_mat),
        ("rigid_global_info.mass_mat_L", self._rigid_global_info.mass_mat_L),
        ("rigid_global_info.mass_mat_D_inv", self._rigid_global_info.mass_mat_D_inv),
        ("rigid_global_info.mass_mat_L_bw", self._rigid_global_info.mass_mat_L_bw),
    ]
    print(f"\n===== [{tag}] =====", flush=True)
    for name, field in fields:
        grad = getattr(field, "grad", None)
        if grad is None:
            continue
        try:
            t = qd_to_torch(grad, copy=True)
        except Exception:
            continue
        if t.numel() == 0:
            continue
        arr = t.detach().cpu().numpy()
        amx = float(np.abs(arr).max())
        # Pretty print shape + full array
        print(f"  {name}  shape={list(arr.shape)}  max|.|={amx:.3e}", flush=True)
        if amx > 0:
            print(f"    {np.array2string(arr, precision=3, separator=', ', suppress_small=False)}", flush=True)

    # Also dump forward primal values for fields that participate in the
    # `force.grad = mass^{-1} · acc.grad` chain. These are stage-invariant
    # (set during prepare_backward_substep) but helpful to inspect once
    # per stage so the manual verification is self-contained.
    primal_fields = [
        ("FWD rigid_global_info.mass_mat", self._rigid_global_info.mass_mat),
        ("FWD rigid_global_info.mass_mat_L", self._rigid_global_info.mass_mat_L),
        ("FWD rigid_global_info.mass_mat_D_inv", self._rigid_global_info.mass_mat_D_inv),
    ]
    if tag.endswith("entry"):  # dump primals only once per substep
        for name, field in primal_fields:
            try:
                t = qd_to_torch(field, copy=True)
            except Exception:
                continue
            if t.numel() == 0:
                continue
            arr = t.detach().cpu().numpy()
            print(f"  {name}  shape={list(arr.shape)}", flush=True)
            print(f"    {np.array2string(arr, precision=4, separator=', ', suppress_small=False)}", flush=True)


def _install_patch():
    import genesis.engine.solvers.rigid.rigid_solver as RS

    RS.RigidSolver._debug_grad_dump = _full_dump


def build(requires_grad):
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
        fh.write(_MJCF_J4)  # type: ignore[name-defined]
    robot = scene.add_entity(gs.morphs.MJCF(file=p))
    scene.build(n_envs=0)
    return scene, robot


def loss_fn(scene):
    state = scene.get_state()
    ss = state.solvers_state[scene.solvers.index(scene.rigid_solver)]
    return (ss.links_pos.reshape(-1) ** 2).sum()


def main():
    gs.init(precision="64", backend=gs.cpu, performance_mode=False, logging_level="warning")
    _install_patch()
    from diag_multistep_worst_case import MJCF_J4  # noqa

    global _MJCF_J4
    _MJCF_J4 = MJCF_J4
    N = 1
    n_dofs = 7
    seed = 1001  # worst N=1 seed

    rng = np.random.default_rng(seed)
    u_list = [rng.normal(size=n_dofs) * 0.3 for _ in range(N)]
    u_tensors = [gs.tensor(u, dtype=gs.tc_float, requires_grad=True) for u in u_list]

    scene, robot = build(True)
    scene.reset()
    for t in range(N):
        robot.control_dofs_force(u_tensors[t])
        scene.step()
    loss = loss_fn(scene)
    print(f"\n[loss] = {float(loss.detach().cpu()):.6e}", flush=True)
    loss.backward()
    for t, u in enumerate(u_tensors):
        g = u.grad.detach().cpu().numpy()
        print(f"u[{t}].grad = {g}")


if __name__ == "__main__":
    main()
