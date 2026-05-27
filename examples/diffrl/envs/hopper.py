"""Hopper environment, ported from DiffRL (NVlabs/DiffRL, ICLR 2022 SHAC).

Reward / observation / action / episode-length match the dflex implementation
(`envs/hopper.py`) so the SHAC config from
`DiffRL/examples/cfg/shac/hopper.yaml` applies directly.

  num_obs = 11    : [height, angle, thigh, leg, foot,
                     vx, vheight, vangle, vthigh, vleg, vfoot]
                    (= dflex `cat([joint_q[1:], joint_qd])`)
  num_act = 3     : thigh/leg/foot joint forces, scaled by action_strength=200
  episode_length  : 1000 steps
  dt              : 1/60, sim_substeps = 16
  reward          : progress(vx) + height_reward + angle_reward
                    + action_penalty * sum(action^2)
  termination     : torso height drops below termination_height (-0.45) when
                    early_termination is on, or the time limit is reached.

NOTE: the hopper needs foot-ground contact, so this env enables collision
(unlike the collision-free cartpole). Differentiable contact backward is not
wired up yet — forward rollouts work; gradient-based SHAC training through
contact is future work.
"""

from __future__ import annotations

import math
import os

import torch

import genesis as gs

from ..genesis_env import GenesisDiffRLEnv


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_HOPPER_XML = os.path.join(_THIS_DIR, "hopper.xml")


class HopperEnv(GenesisDiffRLEnv):
    def __init__(
        self,
        num_envs: int = 256,
        episode_length: int = 1000,
        seed: int = 0,
        no_grad: bool = True,
        render: bool = False,
        device: str = "cpu",
        stochastic_init: bool = True,
        early_termination: bool = True,
        enable_camera: bool = False,
        camera_kwargs: dict | None = None,
        MM_caching_frequency: int = 1,  # accepted for DiffRL config compatibility (ignored)
    ):
        # DiffRL hopper parameters (verbatim from envs/hopper.py).
        self.termination_height = -0.45
        self.termination_angle = math.pi / 6.0
        self.termination_height_tolerance = 0.15
        self.termination_angle_tolerance = 0.05
        self.height_rew_scale = 1.0
        self.action_strength = 200.0
        self.action_penalty = -1e-1

        # Off-screen camera (eval rollout mp4s). Must be set BEFORE super().__init__
        # because the base constructor calls `_build_scene`.
        self.enable_camera = enable_camera
        self._camera_kwargs = camera_kwargs or {}
        self.cam = None

        super().__init__(
            num_envs=num_envs,
            num_obs=11,
            num_actions=3,
            episode_length=episode_length,
            seed=seed,
            no_grad=no_grad,
            render=render,
            device=device,
            stochastic_init=stochastic_init,
            early_termination=early_termination,
        )

    def _build_scene(self):
        # dt = 1/60, substeps = 16 to match DiffRL (`sim_substeps = 16`).
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=1.0 / 60.0,
                substeps=16,
                gravity=(0.0, 0.0, -9.81),
                requires_grad=not self.no_grad,
            ),
            rigid_options=gs.options.RigidOptions(
                # Hopper needs foot-ground contact, so collision is ON (unlike
                # cartpole). Differentiable contact backward is not wired yet.
                enable_collision=True,
                enable_self_collision=False,
                enable_joint_limit=True,
                disable_constraint=False,
                use_hibernation=False,
                use_contact_island=False,
            ),
            show_viewer=self.visualize,
        )
        # Ground plane (the upstream MJCF floor is stripped; added here instead).
        self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(gs.morphs.MJCF(file=_HOPPER_XML))

        if self.enable_camera:
            cam_kwargs = dict(
                res=(640, 480),
                pos=(0.0, -5.0, 1.2),
                lookat=(0.0, 0.0, 0.8),
                fov=40,
                GUI=False,
                env_idx=0,
            )
            cam_kwargs.update(self._camera_kwargs)
            self.cam = self.scene.add_camera(**cam_kwargs)
            # Base camera pose; `update_camera()` slides it in x to track the
            # torso. NOTE: `cam.follow_entity` is unusable here — the planar
            # floating base makes the entity's base link a fixed anchor at the
            # origin (robot.get_pos() == 0), so the torso (link 1) moving in x is
            # not reflected. We track qpos[0] (rootx) directly instead.
            self._cam_base_pos = list(cam_kwargs["pos"])
            self._cam_base_lookat = list(cam_kwargs["lookat"])

        self.scene.build(n_envs=self.num_envs)

        # Home pose: qpos = 0 (torso at its declared z=1.25), zero velocity.
        # qpos layout: [rootx, rootz, rooty, thigh, leg, foot].
        init_qpos = torch.zeros((self.num_envs, 6), device=self._torch_device, dtype=torch.float32)
        init_qvel = torch.zeros((self.num_envs, 6), device=self._torch_device, dtype=torch.float32)
        self._init_qpos = init_qpos
        self._init_qvel = init_qvel

    def _stochastic_init_qpos_qvel(self, env_ids, qpos, qvel):
        # DiffRL hopper randomization (envs/hopper.py reset):
        #   q[0:2] += 0.05 * U(-1,1)          (x, height)
        #   q[2]    = U(-0.05, 0.05)           (angle, SET not added)
        #   q[3:]  += 0.05 * U(-1,1)           (thigh, leg, foot)
        #   qd[:]   = 0.05 * U(-1,1)
        n = env_ids.numel()
        dev, dt = self._torch_device, qpos.dtype

        def u(*shape):
            return (torch.rand(shape, device=dev, dtype=dt) - 0.5) * 2.0

        qpos[env_ids, 0:2] = qpos[env_ids, 0:2] + 0.05 * u(n, 2)
        qpos[env_ids, 2] = u(n) * 0.05
        qpos[env_ids, 3:] = qpos[env_ids, 3:] + 0.05 * u(n, 3)
        qvel[env_ids, :] = 0.05 * u(n, 6)
        return qpos, qvel

    def _rigid_state(self):
        """Current rigid-solver state (gs.Tensors carrying `.scene`)."""
        return self.scene.get_state().solvers_state[self.scene.solvers.index(self.scene.rigid_solver)]

    def update_camera(self):
        """Slide the off-screen camera in x to keep the hopper framed as it
        moves forward. Tracks qpos[0] (rootx = torso world x) of env 0; the side
        distance (y), height (z), and look-at height stay at the camera's
        configured base pose. Renderers (eval / rollout) call this per frame."""
        if self.cam is None:
            return
        x = float(self._rigid_state().qpos[0, 0].detach())
        bp, bl = self._cam_base_pos, self._cam_base_lookat
        self.cam.set_pose(pos=(bp[0] + x, bp[1], bp[2]), lookat=(bl[0] + x, bl[1], bl[2]))

    def _apply_actions(self, actions: torch.Tensor):
        # Only the 3 leg joints (thigh, leg, foot) are actuated; the 3 root DOFs
        # (x / height / pitch) take zero force. Force = action * action_strength.
        rigid_state = self._rigid_state()
        scene_anchor = rigid_state.qpos[:, 0:1] * 0.0  # gs.Tensor, scene=self.scene
        act_force = actions * self.action_strength + scene_anchor
        zero_root = torch.zeros_like(act_force)
        force = torch.cat([zero_root, act_force], dim=-1)  # (num_envs, 6)
        self.robot.control_dofs_force(force)

    def compute_observations(self):
        rigid_state = self._rigid_state()
        qpos = rigid_state.qpos  # (num_envs, 6)
        qvel = rigid_state.dofs_vel  # (num_envs, 6)
        # obs = [qpos[1:], qvel] : drop the (translation-invariant) x position.
        self.obs_buf = torch.cat([qpos[:, 1:], qvel], dim=-1)

    def compute_reward(self):
        height = self.obs_buf[:, 0]
        angle = self.obs_buf[:, 1]
        forward_vel = self.obs_buf[:, 5]  # qvel[0] = x velocity = progress

        height_diff = height - (self.termination_height + self.termination_height_tolerance)
        height_reward = torch.clip(height_diff, -1.0, 0.3)
        height_reward = torch.where(height_reward < 0.0, -200.0 * height_reward * height_reward, height_reward)
        height_reward = torch.where(height_reward > 0.0, self.height_rew_scale * height_reward, height_reward)

        angle_reward = 1.0 * (-(angle**2) / (self.termination_angle**2) + 1.0)
        progress_reward = forward_vel

        self.rew_buf = (
            progress_reward + height_reward + angle_reward + torch.sum(self.actions**2, dim=-1) * self.action_penalty
        )

        # Time-limit reset.
        self.reset_buf = torch.where(
            self.progress_buf > self.episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf,
        )
        # Early termination on a fallen torso.
        if self.early_termination:
            fell = height < self.termination_height
            self.reset_buf = torch.where(fell, torch.ones_like(self.reset_buf), self.reset_buf)
            self.termination_buf = torch.where(fell, torch.ones_like(self.termination_buf), self.termination_buf)
