"""Integration-level optimization convergence for the differentiable rigid solver.

Within the differentiable-rigid test suite, this is the only *end-to-end* check.
The others verify the gradient is **locally** correct (FD-vs-analytical at a
point); this one asks whether that gradient is **useful** — does plain Adam,
driven by the diff-mode backward over a multi-step horizon, actually converge to
a known answer? Siblings:
  - test_diff_forward_kinematics : local FD of the FK + velocity gradient.
  - test_diff_joint_limit        : the joint-limit constraint gradient.
  - test_diff_contact            : the collision / diff-GJK contact gradient.
  - test_diff_scene_backward     : the scene.backward() API + horizon truncation.
  - test_diff_optim              : *this file* — multi-step Adam convergence.

Two optimization targets on the cartpole (contact-free), each recovering a
final-state target produced by an identical rollout from known inputs:
  1. `test_diff_optim_init_vel_cartpole`      — Adam on the initial `dofs_velocity`.
  2. `test_diff_optim_control_force_cartpole` — Adam on per-step `control_dofs_force`
     (the SHAC swing-up pattern: a fresh control input applied every step).
Each asserts the per-env loss (a) drops by ≥2 orders of magnitude and (b) ends
below an absolute threshold — i.e. the backward yields an informative descent
direction over the horizon, not merely a locally correct gradient.

Parametrized over precision ∈ {fp64, fp32} and n_envs ∈ {0 (single), 4
(batched, per-env distinct target / init)}. fp32 uses looser thresholds for its
lower precision floor.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import genesis as gs


MJCF_CARTPOLE = (Path(__file__).resolve().parent.parent / "examples" / "diffrl" / "envs" / "cartpole.xml").read_text()


_PRECISION_PARAMS = [
    pytest.param("64", marks=pytest.mark.precision("64"), id="fp64"),
    pytest.param("32", marks=pytest.mark.precision("32"), id="fp32"),
]

_N_ENVS_PARAMS = [
    pytest.param(0, id="single"),
    pytest.param(4, id="batched"),
]


def _mjcf_to_tmpfile(mjcf_str: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".xml")
    with os.fdopen(fd, "w") as f:
        f.write(mjcf_str)
    return path


def _build_scene(mjcf_path: str, *, requires_grad: bool, n_envs: int = 0, substeps: int = 1):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=substeps,
            gravity=(0.0, 0.0, -9.81),
            requires_grad=requires_grad,
        ),
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
    robot = scene.add_entity(gs.morphs.MJCF(file=mjcf_path))
    scene.build(n_envs=n_envs)
    return scene, robot


def _rigid_state(scene):
    state = scene.get_state()
    return state.solvers_state[scene.solvers.index(scene.rigid_solver)]


def _rollout(scene, robot, init_vel, n_steps):
    """Apply `init_vel` via `set_dofs_velocity` (the @tracked setter) on a
    fresh `scene.reset()`, step `n_steps` times, and return the post-rollout
    (qpos, dofs_vel) tensors from the solver state."""
    scene.reset()
    robot.set_dofs_velocity(init_vel)
    for _ in range(n_steps):
        scene.step()
    s = _rigid_state(scene)
    return s.qpos, s.dofs_vel


def _input_shape(n_dofs: int, n_envs: int):
    return (n_dofs,) if n_envs == 0 else (n_envs, n_dofs)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("precision", _PRECISION_PARAMS)
@pytest.mark.parametrize("n_envs", _N_ENVS_PARAMS)
def test_diff_optim_init_vel_cartpole(show_viewer, n_envs, precision, backend):
    """Adam on `init_dofs_velocity` (2 params per env: slider + hinge)
    recovers the per-env final-state target produced by an identical
    rollout from a known `target_init_vel`. Verifies the diff-mode
    backward yields an informative descent direction over an N=32 step
    horizon, with batch independence when n_envs > 0."""
    N_STEPS = 32
    N_ITER = 200
    LR = 1e-2
    N_DOFS = 2  # cartpole: slider + hinge
    B = n_envs if n_envs > 0 else 1

    # Precision-specific tolerances. fp32 has ~7 significant digits, so a
    # rollout-trained scalar loss can't get below ~1e-4 even at the
    # optimum, and the rate of improvement plateaus earlier.
    if precision == "64":
        REL_REDUCTION = 1e-2
        ABS_THRESHOLD = 1e-4
    else:
        REL_REDUCTION = 1e-1
        ABS_THRESHOLD = 1e-2

    rng = np.random.default_rng(seed=11)

    # --- target trajectory (non-differentiable scene). Per-env distinct
    # target_init_vel when n_envs > 0, so each env converges to its own
    # answer.
    target_init_vel_np = rng.normal(size=_input_shape(N_DOFS, n_envs)) * 0.5

    mjcf_path = _mjcf_to_tmpfile(MJCF_CARTPOLE)
    scene_ref, robot_ref = _build_scene(mjcf_path, requires_grad=False, n_envs=n_envs)
    target_init_vel_t = gs.tensor(target_init_vel_np, dtype=gs.tc_float)
    with torch.no_grad():
        target_qpos, target_vel = _rollout(scene_ref, robot_ref, target_init_vel_t, N_STEPS)
        target_qpos = target_qpos.detach().clone()
        target_vel = target_vel.detach().clone()

    # --- differentiable scene to optimize on. Per-env distinct init noise.
    scene_opt, robot_opt = _build_scene(mjcf_path, requires_grad=True, n_envs=n_envs)
    init_offset = rng.normal(size=_input_shape(N_DOFS, n_envs)) * 0.3
    init_vel_np = target_init_vel_np + init_offset
    init_vel = gs.tensor(init_vel_np, dtype=gs.tc_float, requires_grad=True)

    opt = torch.optim.Adam([init_vel], lr=LR)
    loss_history = []  # list of (B,) per-env loss arrays

    for it in range(N_ITER):
        opt.zero_grad(set_to_none=False)
        pred_qpos, pred_vel = _rollout(scene_opt, robot_opt, init_vel, N_STEPS)
        diff_pos = (pred_qpos - target_qpos).reshape(B, -1)
        diff_vel = (pred_vel - target_vel).reshape(B, -1)
        loss_per_env = (diff_pos**2).sum(dim=-1) + (diff_vel**2).sum(dim=-1)
        loss = loss_per_env.sum()
        loss_history.append(loss_per_env.detach().cpu().numpy().copy())
        loss.backward()
        assert init_vel.grad is not None, f"iter {it}: init_vel.grad is None"
        opt.step()

    history = np.asarray(loss_history)  # (N_ITER, B)
    initial = history[0]
    final = history[-1]

    # Save per-env loss curves for visual inspection.
    if not os.environ.get("GENESIS_DIFF_OPTIM_NO_PLOT"):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            backend_name = getattr(backend, "name", str(backend))
            tag = f"{backend_name}_{precision}_{'batched' if n_envs > 0 else 'single'}"
            default_out = (
                Path(__file__).resolve().parent.parent / "runs" / "tmp" / f"diff_optim_init_vel_cartpole_{tag}.png"
            )
            out_path = Path(os.environ.get("GENESIS_DIFF_OPTIM_PLOT_PATH", str(default_out)))
            out_path.parent.mkdir(parents=True, exist_ok=True)

            cmap = plt.get_cmap("tab10")
            fig, ax = plt.subplots(figsize=(7, 4))
            for b in range(B):
                ax.plot(history[:, b], lw=1.2, color=cmap(b % 10), label=f"env{b}")
            ax.set_yscale("log")
            ax.set_xlabel("iteration")
            ax.set_ylabel("loss (log scale)")
            ax.set_title(
                f"cartpole init_vel optim [{tag}]: "
                f"init={initial.max():.2e} → final={final.max():.2e} "
                f"(worst-env ratio {(final / initial).max():.2e})"
            )
            ax.grid(True, which="both", alpha=0.3)
            if B > 1:
                ax.legend(loc="upper right", fontsize=8)
            fig.tight_layout()
            fig.savefig(str(out_path), dpi=120)
            plt.close(fig)
            print(f"\n[diff_optim] loss curve saved to {out_path}")
        except ImportError:
            pass

    # Per-env assertions: every env must satisfy both criteria.
    rel_ratios = final / initial
    worst_rel_env = int(np.argmax(rel_ratios))
    assert (rel_ratios < REL_REDUCTION).all(), (
        f"loss reduction insufficient (worst env={worst_rel_env}): "
        f"initial={initial[worst_rel_env]:.3e}, final={final[worst_rel_env]:.3e}, "
        f"ratio={rel_ratios[worst_rel_env]:.3e} (>= {REL_REDUCTION:.0e})"
    )
    worst_abs_env = int(np.argmax(final))
    assert (final < ABS_THRESHOLD).all(), (
        f"final loss above absolute threshold (worst env={worst_abs_env}): "
        f"{final[worst_abs_env]:.3e} >= {ABS_THRESHOLD:.0e}"
    )


def _save_loss_plot(history: np.ndarray, *, title_tag: str, plot_name: str):
    """history: (N_ITER, B). Saves a per-env log-scale loss curve."""
    if os.environ.get("GENESIS_DIFF_OPTIM_NO_PLOT"):
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    default_out = Path(__file__).resolve().parent.parent / "runs" / "tmp" / f"{plot_name}.png"
    out_path = Path(os.environ.get("GENESIS_DIFF_OPTIM_PLOT_PATH", str(default_out)))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    initial = history[0]
    final = history[-1]
    B = history.shape[1]
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7, 4))
    for b in range(B):
        ax.plot(history[:, b], lw=1.2, color=cmap(b % 10), label=f"env{b}")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss (log scale)")
    ax.set_title(
        f"{title_tag}: init={initial.max():.2e} → final={final.max():.2e} "
        f"(worst-env ratio {(final / initial).max():.2e})"
    )
    ax.grid(True, which="both", alpha=0.3)
    if B > 1:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    print(f"\n[diff_optim] loss curve saved to {out_path}")


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("precision", _PRECISION_PARAMS)
@pytest.mark.parametrize("n_envs", _N_ENVS_PARAMS)
def test_diff_optim_control_force_cartpole(show_viewer, n_envs, precision, backend):
    """Adam on per-step `control_dofs_force` (N_STEPS × n_dofs params per
    env) recovers the per-env final-state target. Mirrors the SHAC
    swing-up pattern: a fresh control input is applied at every
    simulation step and the loss only depends on the terminal qpos/vel.
    Exercises the per-step `_tgt_buffer` backward walk."""
    N_STEPS = 32
    N_ITER = 200
    LR = 1e-2
    N_DOFS = 2
    B = n_envs if n_envs > 0 else 1

    if precision == "64":
        REL_REDUCTION = 1e-2
        ABS_THRESHOLD = 1e-4
    else:
        REL_REDUCTION = 1e-1
        ABS_THRESHOLD = 1e-2

    rng = np.random.default_rng(seed=23)

    shape_per_step = _input_shape(N_DOFS, n_envs)
    target_force_np = rng.normal(size=(N_STEPS,) + shape_per_step) * 0.2

    # --- target trajectory (non-differentiable scene) ---
    mjcf_path = _mjcf_to_tmpfile(MJCF_CARTPOLE)
    scene_ref, robot_ref = _build_scene(mjcf_path, requires_grad=False, n_envs=n_envs)
    with torch.no_grad():
        scene_ref.reset()
        for t in range(N_STEPS):
            robot_ref.control_dofs_force(gs.tensor(target_force_np[t], dtype=gs.tc_float))
            scene_ref.step()
        s = _rigid_state(scene_ref)
        target_qpos = s.qpos.detach().clone()
        target_vel = s.dofs_vel.detach().clone()

    # --- differentiable scene + learnable per-step force tensors ---
    scene_opt, robot_opt = _build_scene(mjcf_path, requires_grad=True, n_envs=n_envs)
    init_offset = rng.normal(size=(N_STEPS,) + shape_per_step) * 0.1
    init_force_np = target_force_np + init_offset
    forces = [gs.tensor(init_force_np[t], dtype=gs.tc_float, requires_grad=True) for t in range(N_STEPS)]
    optimizer = torch.optim.Adam(forces, lr=LR)

    loss_history = []
    for it in range(N_ITER):
        optimizer.zero_grad(set_to_none=False)
        scene_opt.reset()
        for t in range(N_STEPS):
            robot_opt.control_dofs_force(forces[t])
            scene_opt.step()
        s = _rigid_state(scene_opt)
        diff_pos = (s.qpos - target_qpos).reshape(B, -1)
        diff_vel = (s.dofs_vel - target_vel).reshape(B, -1)
        loss_per_env = (diff_pos**2).sum(dim=-1) + (diff_vel**2).sum(dim=-1)
        loss = loss_per_env.sum()
        loss_history.append(loss_per_env.detach().cpu().numpy().copy())
        loss.backward()
        for t, f in enumerate(forces):
            assert f.grad is not None, f"iter {it} step {t}: force.grad is None"
        optimizer.step()

    history = np.asarray(loss_history)  # (N_ITER, B)
    initial = history[0]
    final = history[-1]

    backend_name = getattr(backend, "name", str(backend))
    tag = f"{backend_name}_{precision}_{'batched' if n_envs > 0 else 'single'}"
    _save_loss_plot(
        history,
        title_tag=f"cartpole control_force optim [{tag}]",
        plot_name=f"diff_optim_control_force_cartpole_{tag}",
    )

    rel_ratios = final / initial
    worst_rel_env = int(np.argmax(rel_ratios))
    assert (rel_ratios < REL_REDUCTION).all(), (
        f"loss reduction insufficient (worst env={worst_rel_env}): "
        f"initial={initial[worst_rel_env]:.3e}, final={final[worst_rel_env]:.3e}, "
        f"ratio={rel_ratios[worst_rel_env]:.3e} (>= {REL_REDUCTION:.0e})"
    )
    worst_abs_env = int(np.argmax(final))
    assert (final < ABS_THRESHOLD).all(), (
        f"final loss above absolute threshold (worst env={worst_abs_env}): "
        f"{final[worst_abs_env]:.3e} >= {ABS_THRESHOLD:.0e}"
    )
