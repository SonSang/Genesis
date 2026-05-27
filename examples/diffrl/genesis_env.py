"""Differentiable RL environment base on top of Genesis.

The key methods are:

  * ``step(actions) -> (obs, rew, reset, extras)`` — torch-only API; the
    forward pass through the differentiable simulator returns
    grad-aware tensors when ``no_grad=False``.
  * ``reset(env_ids=None, force_reset=True) -> obs`` — reset the indicated
    environments to their (optionally stochastic) initial state.
  * ``clear_grad()`` — snapshot the current (qpos, qvel) and rebind it as a
    detached tensor, cutting the autograd tape so the next horizon's
    backward pass doesn't accumulate adjoints from prior horizons.
  * ``initialize_trajectory() -> obs`` — wraps ``clear_grad`` + recomputes
    observations; called by SHAC at the start of each horizon.

Subclasses must implement:
  * ``_build_scene()`` — construct ``self.scene`` and ``self.robot`` (the
    primary articulated entity), and any per-env initialization state.
  * ``_apply_actions(actions)`` — translate the normalized action tensor
    (clipped to [-1, 1]) into joint forces / torques and apply them to
    ``self.robot`` for the upcoming step.
  * ``compute_observations()`` — fill ``self.obs_buf``.
  * ``compute_reward()`` — fill ``self.rew_buf`` and update
    ``self.reset_buf`` / ``self.termination_buf`` based on the post-step
    state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch

import genesis as gs


class _BoxSpace:
    """Minimal Gym Box-style space — just low/high/shape, no validation."""

    def __init__(self, low: np.ndarray, high: np.ndarray):
        self.low = low
        self.high = high
        self.shape = low.shape


class GenesisDiffRLEnv(ABC):
    """Base class for Genesis-backed differentiable RL environments."""

    def __init__(
        self,
        num_envs: int,
        num_obs: int,
        num_actions: int,
        episode_length: int,
        seed: int = 0,
        no_grad: bool = True,
        render: bool = False,
        device: str = "cpu",
        stochastic_init: bool = False,
        early_termination: bool = False,
    ):
        # Bookkeeping that the trainer reads off the env object.
        self.num_envs = num_envs
        self.num_environments = num_envs
        self.num_agents = 1
        self.num_obs = num_obs
        self.num_observations = num_obs
        self.num_actions = num_actions
        self.episode_length = episode_length
        self.seed = seed
        self.no_grad = no_grad
        self.visualize = render
        self.stochastic_init = stochastic_init
        self.early_termination = early_termination
        self.device = device

        self.sim_time = 0.0
        self.num_frames = 0

        # Gym Box-style spaces.
        self.obs_space = _BoxSpace(
            low=-np.inf * np.ones(num_obs, dtype=np.float32),
            high=np.inf * np.ones(num_obs, dtype=np.float32),
        )
        self.act_space = _BoxSpace(
            low=-np.ones(num_actions, dtype=np.float32),
            high=np.ones(num_actions, dtype=np.float32),
        )

        # Trajectory buffers (torch). `obs_buf` / `rew_buf` may be re-bound
        # to grad-aware tensors during step() when `no_grad=False`.
        torch_device = torch.device(device)
        self._torch_device = torch_device
        self.obs_buf = torch.zeros((num_envs, num_obs), device=torch_device, dtype=torch.float32)
        self.rew_buf = torch.zeros(num_envs, device=torch_device, dtype=torch.float32)
        self.reset_buf = torch.ones(num_envs, device=torch_device, dtype=torch.long)
        self.termination_buf = torch.zeros(num_envs, device=torch_device, dtype=torch.long)
        self.progress_buf = torch.zeros(num_envs, device=torch_device, dtype=torch.long)
        self.actions = torch.zeros((num_envs, num_actions), device=torch_device, dtype=torch.float32)
        self.extras: dict = {}

        # When True (training default), `step()` auto-resets envs whose
        # reset_buf is set (time limit / early termination) in the same call.
        # Eval sets this False so the terminal frame can be rendered before the
        # reset clobbers it — otherwise the last recorded video frame is the
        # post-reset init pose. Callers that disable it must handle resets
        # themselves (run_eval just breaks on `reset.any()`).
        self.auto_reset = True

        # Subclass populates `self.scene`, `self.robot`, and the initial
        # state buffers (`self._init_qpos`, `self._init_qvel`).
        self.scene: gs.Scene | None = None
        self.robot = None
        self._init_qpos: torch.Tensor | None = None  # (num_envs, n_qpos)
        self._init_qvel: torch.Tensor | None = None  # (num_envs, n_qvel)
        self._build_scene()
        assert self.scene is not None and self.robot is not None, "_build_scene must populate self.scene and self.robot"
        assert self._init_qpos is not None and self._init_qvel is not None, (
            "_build_scene must populate self._init_qpos and self._init_qvel"
        )

        # Reset all envs to populate initial obs.
        self.reset(force_reset=True)

    # ------------------------------------------------------------------ #
    # Subclass hooks
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _build_scene(self):
        """Build `self.scene`, add `self.robot`, set `self._init_qpos` / `self._init_qvel`."""

    @abstractmethod
    def _apply_actions(self, actions: torch.Tensor):
        """Translate normalized actions (clipped to [-1, 1]) into joint forces and apply."""

    @abstractmethod
    def compute_observations(self):
        """Fill `self.obs_buf` based on current scene state."""

    @abstractmethod
    def compute_reward(self):
        """Fill `self.rew_buf`, update `self.reset_buf` / `self.termination_buf`."""

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #
    def step(self, actions: torch.Tensor):
        actions = actions.view(self.num_envs, self.num_actions)
        actions = torch.clip(actions, -1.0, 1.0)
        self.actions = actions
        self._apply_actions(actions)

        self.scene.step()
        self.sim_time += self.scene._sim.dt
        self.num_frames += 1
        self.progress_buf = self.progress_buf + 1

        # Default: no early termination, just check time limit.
        self.reset_buf = torch.zeros_like(self.reset_buf)
        self.termination_buf = torch.zeros_like(self.termination_buf)

        self.compute_observations()
        self.compute_reward()

        if not self.no_grad:
            self.extras = {
                "obs_before_reset": self.obs_buf.clone(),
                "episode_end": self.termination_buf.clone(),
            }

        # Reset any envs whose reset_buf is set. For now we reset all
        # specified envs together; per-env partial resets via envs_idx
        # are supported by the underlying Genesis API. Skipped when
        # `auto_reset` is False (eval), so the terminal state survives for
        # rendering instead of being overwritten by the init pose.
        if self.auto_reset:
            env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if env_ids.numel() > 0:
                self.reset(env_ids=env_ids)

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self, env_ids=None, force_reset: bool = True):
        if env_ids is None:
            if force_reset:
                env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self._torch_device)
            else:
                return self.obs_buf

        if env_ids.numel() == 0:
            return self.obs_buf

        # Build per-env qpos / qvel: start from the canonical init, optionally add stochastic noise.
        new_qpos = self._init_qpos.clone()
        new_qvel = self._init_qvel.clone()
        if self.stochastic_init:
            new_qpos, new_qvel = self._stochastic_init_qpos_qvel(env_ids, new_qpos, new_qvel)

        # Apply via the rigid solver. We use set_dofs_position / set_dofs_velocity which is
        # the closest analog to dflex's joint_q / joint_qd direct write.
        if env_ids.numel() == self.num_envs:
            # All envs — simple path, no envs_idx required.
            self.robot.set_dofs_position(new_qpos)
            self.robot.set_dofs_velocity(new_qvel)
        else:
            # Partial reset. set_dofs_position supports envs_idx.
            self.robot.set_dofs_position(new_qpos[env_ids], envs_idx=env_ids)
            self.robot.set_dofs_velocity(new_qvel[env_ids], envs_idx=env_ids)

        self.progress_buf[env_ids] = 0
        self.compute_observations()
        return self.obs_buf

    def _stochastic_init_qpos_qvel(self, env_ids, qpos, qvel):
        """Hook for subclasses to add per-env noise to the initial state.

        Default: leave qpos/qvel unchanged. Subclasses override if their env
        wants randomized starts (e.g. cartpole's ``+np.pi * uniform(-0.5, 0.5)``).
        """
        return qpos, qvel

    def clear_grad(self):
        """Detach the current physics state from any prior autograd tape.

        Implementation: snapshot ``qpos`` and ``qvel`` (these come back as
        detached torch tensors), clear Genesis's internal gradient buffers
        and queried-state cache via ``scene.reset_grad()``, then push the
        snapshot back into the solver. After this call the simulator is
        ready to start a fresh forward pass that *does* track gradients,
        but whose tape has no link to the previous horizon.
        """
        with torch.no_grad():
            current_qpos = self.robot.get_dofs_position().detach().clone()
            current_qvel = self.robot.get_dofs_velocity().detach().clone()
        self.scene.reset_grad()
        # `_t` is unchanged by `scene.reset_grad()` (it's not a state reset). If a
        # subclass needs the simulator's substep counter zeroed too, override
        # `clear_grad` to additionally call `scene.reset(state)` with a snapshot.
        self.robot.set_dofs_position(current_qpos)
        self.robot.set_dofs_velocity(current_qvel)

    def initialize_trajectory(self):
        self.clear_grad()
        self.compute_observations()
        return self.obs_buf

    def render(self, mode: str = "human"):
        if self.visualize and self.scene is not None and self.scene._visualizer is not None:
            self.scene._visualizer.render()

    def close(self):
        gs.destroy()
