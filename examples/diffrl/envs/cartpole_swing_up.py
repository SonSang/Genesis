"""Cartpole swing-up environment, ported from DiffRL (NVlabs/DiffRL, ICLR 2022 SHAC).

Reward / observation / action / episode-length match the dflex implementation
1:1 so the SHAC config from `DiffRL/examples/cfg/shac/cartpole_swing_up.yaml`
applies directly.

  num_obs = 5     : [x, xdot, sin(theta), cos(theta), theta_dot]
  num_act = 1     : cart force, scaled by action_strength=1000.0
  episode_length  : 240 steps
  dt              : 1/60
  initial state   : cart at x=0, pole hanging down (theta = -pi)
  reward          : -theta^2 * 1.0 - theta_dot^2 * 0.1 - x^2 * 0.05 - xdot^2 * 0.1
                    (pole_angle_penalty, pole_velocity_penalty, cart_position_penalty,
                     cart_velocity_penalty exactly as in DiffRL)
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch

import genesis as gs

from ..genesis_env import GenesisDiffRLEnv


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CARTPOLE_XML = os.path.join(_THIS_DIR, "cartpole.xml")


def _normalize_angle(theta: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-pi, pi] (matches `utils.torch_utils.normalize_angle`)."""
    return ((theta + math.pi) % (2.0 * math.pi)) - math.pi


class CartPoleSwingUpEnv(GenesisDiffRLEnv):
    def __init__(
        self,
        num_envs: int = 64,
        episode_length: int = 240,
        seed: int = 0,
        no_grad: bool = True,
        render: bool = False,
        device: str = "cpu",
        stochastic_init: bool = True,
        early_termination: bool = False,
        MM_caching_frequency: int = 1,  # accepted for DiffRL config compatibility (ignored)
    ):
        # DiffRL parameters (verbatim from cartpole_swing_up.py)
        self.action_strength = 1000.0
        self.pole_angle_penalty = 1.0
        self.pole_velocity_penalty = 0.1
        self.cart_position_penalty = 0.05
        self.cart_velocity_penalty = 0.1
        self.cart_action_penalty = 0.0

        super().__init__(
            num_envs=num_envs,
            num_obs=5,
            num_actions=1,
            episode_length=episode_length,
            seed=seed,
            no_grad=no_grad,
            render=render,
            device=device,
            stochastic_init=stochastic_init,
            early_termination=early_termination,
        )

    def _build_scene(self):
        # Genesis Scene. dt = 1/60 to match DiffRL. The diff rigid solver
        # works at the dt granularity (no substep loop needed for cartpole).
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=1.0 / 60.0,
                gravity=(0.0, 0.0, -9.81),
                requires_grad=not self.no_grad,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_collision=False,
                enable_self_collision=False,
                enable_joint_limit=False,
                disable_constraint=True,
                use_hibernation=False,
                use_contact_island=False,
            ),
            show_viewer=self.visualize,
        )
        self.robot = self.scene.add_entity(gs.morphs.MJCF(file=_CARTPOLE_XML))
        self.scene.build(n_envs=self.num_envs)

        # Initial state: cart at 0, pole hanging down (theta = -pi).
        # qpos layout for cartpole: [cart_x, pole_theta]
        init_qpos = torch.zeros((self.num_envs, 2), device=self._torch_device, dtype=torch.float32)
        init_qpos[:, 1] = -math.pi
        init_qvel = torch.zeros((self.num_envs, 2), device=self._torch_device, dtype=torch.float32)
        self._init_qpos = init_qpos
        self._init_qvel = init_qvel

    def _stochastic_init_qpos_qvel(self, env_ids, qpos, qvel):
        # DiffRL noise: qpos += pi * U(-0.5, 0.5) , qvel += 0.5 * U(-0.5, 0.5)
        # Applied independently to both joints (cart and pole).
        n = env_ids.numel()
        qpos_noise = math.pi * (torch.rand((n, 2), device=self._torch_device) - 0.5)
        qvel_noise = 0.5 * (torch.rand((n, 2), device=self._torch_device) - 0.5)
        qpos[env_ids] = qpos[env_ids] + qpos_noise
        qvel[env_ids] = qvel[env_ids] + qvel_noise
        return qpos, qvel

    def _rigid_state(self):
        """Return the current rigid-solver state (gs.Tensors with .scene set).

        Used by both `_apply_actions` (to anchor `.scene` on the force tensor
        so the actor's gradient flows through `control_dofs_force`) and by
        `compute_observations` / `compute_reward` (so the obs/rew tensors
        also carry `.scene` and `loss.backward()` triggers
        `scene._backward()`).

        Genesis registers the returned `state` on
        `scene._sim._queried_states`, which is the list the backward unroll
        replays — so calling this every step is the right pattern.
        """
        return self.scene.get_state().solvers_state[self.scene.solvers.index(self.scene.rigid_solver)]

    def _apply_actions(self, actions: torch.Tensor):
        # Only the cart joint is actuated. Force = action * action_strength.
        # Pole joint takes zero force. `control_dofs_force` is `@tracked`, so
        # the gradient flows back to `actions` via `process_input_grad` ->
        # `_backward_from_qd` -> `set_dofs_force_grad` (which reads
        # `ctrl_force.grad` from the simulator-side backward chain).
        cart_force = actions[:, 0:1] * self.action_strength
        pole_force = torch.zeros_like(cart_force)
        force = torch.cat([cart_force, pole_force], dim=-1)
        self.robot.control_dofs_force(force)

    def compute_observations(self):
        rigid_state = self._rigid_state()
        qpos = rigid_state.qpos  # (num_envs, 2), gs.Tensor with .scene
        qvel = rigid_state.dofs_vel
        x = qpos[:, 0:1]
        theta = qpos[:, 1:2]
        xdot = qvel[:, 0:1]
        theta_dot = qvel[:, 1:2]
        self.obs_buf = torch.cat([x, xdot, torch.sin(theta), torch.cos(theta), theta_dot], dim=-1)

    def compute_reward(self):
        rigid_state = self._rigid_state()
        qpos = rigid_state.qpos
        qvel = rigid_state.dofs_vel
        x = qpos[:, 0]
        theta = _normalize_angle(qpos[:, 1])
        xdot = qvel[:, 0]
        theta_dot = qvel[:, 1]
        actions = self.actions

        self.rew_buf = (
            -torch.pow(theta, 2.0) * self.pole_angle_penalty
            - torch.pow(theta_dot, 2.0) * self.pole_velocity_penalty
            - torch.pow(x, 2.0) * self.cart_position_penalty
            - torch.pow(xdot, 2.0) * self.cart_velocity_penalty
            - torch.sum(actions**2, dim=-1) * self.cart_action_penalty
        )

        # Time-limit reset (no early termination for cartpole).
        self.reset_buf = torch.where(
            self.progress_buf > self.episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf,
        )
