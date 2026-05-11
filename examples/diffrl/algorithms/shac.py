"""Short-Horizon Actor-Critic (SHAC) for Genesis-backed differentiable envs.

A pragmatic port of `algorithms/shac.py` from NVlabs/DiffRL (ICLR 2022).
Mirrors the original's structure / hyperparameter naming so the published
configs apply with minimal translation, but drops the heavier baselines
(USD rendering hooks, EMA target critic etc.) until they're needed.

Algorithm in brief:
  1. **Actor rollout**: H=`steps_num` differentiable env steps, accumulating
     a *gamma-discounted* reward. Last step bootstraps the critic value (or
     critic of `obs_before_reset` if the env reset mid-horizon).
  2. **Actor update**: `actor_loss = -mean(discounted_reward + γ^H · V_H)`,
     backprop through sim, clip global grad norm, Adam step.
  3. **Critic update**: TD-lambda targets computed over the rollout, then a
     few epochs of MLP regression against those targets.
  4. **Detach + repeat**: `env.clear_grad()` snapshots state, calls
     `scene.reset_grad()`, and we start the next horizon from the same
     physics state with a fresh autograd tape.
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass

import numpy as np
import torch

from ..models import ActorStochasticMLP, ActorDeterministicMLP, CriticMLP
from ..running_mean_std import RunningMeanStd


@dataclass
class SHACConfig:
    # Network shapes
    actor_units: list[int]
    critic_units: list[int]
    activation: str = "elu"
    actor_stochastic: bool = True
    logstd_init: float = -1.0

    # Training
    max_epochs: int = 500
    steps_num: int = 32  # H, horizon length
    num_actors: int = 64  # parallel envs
    actor_lr: float = 2e-3
    critic_lr: float = 2e-3
    lr_schedule: str = "linear"  # "linear" or "constant"
    betas: tuple[float, float] = (0.7, 0.95)
    gamma: float = 0.99
    lam: float = 0.95
    grad_norm: float = 1.0
    truncate_grads: bool = True

    # Critic update
    critic_iterations: int = 16
    critic_method: str = "td-lambda"
    num_batch: int = 4
    target_critic_alpha: float = 0.2

    # Normalization
    obs_rms: bool = True

    # Misc
    save_interval: int = 0
    log_interval: int = 1
    device: str = "cpu"
    seed: int = 0


class SHAC:
    def __init__(self, env, cfg: SHACConfig, log_dir: str | None = None):
        self.env = env
        self.cfg = cfg
        self.device = cfg.device
        self.log_dir = log_dir
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        obs_dim = env.num_obs
        act_dim = env.num_actions

        if cfg.actor_stochastic:
            self.actor = ActorStochasticMLP(obs_dim, act_dim, cfg.actor_units, cfg.activation, cfg.logstd_init).to(
                self.device
            )
        else:
            self.actor = ActorDeterministicMLP(obs_dim, act_dim, cfg.actor_units, cfg.activation).to(self.device)
        self.critic = CriticMLP(obs_dim, cfg.critic_units, cfg.activation).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad = False

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr, betas=cfg.betas)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr, betas=cfg.betas)

        self.obs_rms = RunningMeanStd(obs_dim, device=self.device) if cfg.obs_rms else None

        # Stats
        self.history = {
            "epoch": [],
            "mean_horizon_reward": [],
            "actor_loss": [],
            "critic_loss": [],
            "grad_norm": [],
            "episode_reward": [],
            "episode_length": [],
        }
        # Per-env running episode trackers
        self._ep_rew = torch.zeros(env.num_envs, device=self.device)
        self._ep_len = torch.zeros(env.num_envs, device=self.device, dtype=torch.long)
        self._ep_rew_history: list[float] = []
        self._ep_len_history: list[int] = []

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    def _normalize(self, obs: torch.Tensor) -> torch.Tensor:
        if self.obs_rms is None:
            return obs
        return self.obs_rms.normalize(obs)

    def _current_lr(self, epoch: int, base_lr: float) -> float:
        if self.cfg.lr_schedule == "constant":
            return base_lr
        frac = 1.0 - epoch / max(1, self.cfg.max_epochs)
        return base_lr * frac

    def _set_optimizer_lr(self, opt: torch.optim.Optimizer, lr: float):
        for group in opt.param_groups:
            group["lr"] = lr

    # ------------------------------------------------------------------ #
    # SHAC actor pass
    # ------------------------------------------------------------------ #
    def _compute_actor_loss_and_traj(self):
        """Run one horizon, accumulating discounted reward and stashing per-step
        obs / rew / done / value tensors that the critic update will read."""
        cfg = self.cfg
        H = cfg.steps_num
        num_envs = self.env.num_envs

        obs_buf = torch.zeros((H + 1, num_envs, self.env.num_obs), device=self.device)
        rew_buf = torch.zeros((H, num_envs), device=self.device)
        done_buf = torch.zeros((H, num_envs), device=self.device)
        next_value_buf = torch.zeros((H, num_envs), device=self.device)

        obs = self.env.initialize_trajectory()
        if self.obs_rms is not None:
            self.obs_rms.update(obs.detach())
        actor_loss = torch.tensor(0.0, device=self.device)
        gamma = cfg.gamma

        for t in range(H):
            obs_norm = self._normalize(obs)
            obs_buf[t] = obs_norm.detach()
            actions = self.actor(obs_norm)
            obs, rew, reset, extras = self.env.step(actions)
            if self.obs_rms is not None:
                self.obs_rms.update(obs.detach())

            with torch.no_grad():
                done = reset.float()
                done_buf[t] = done
                # Critic bootstrap: V(obs_after_reset) for surviving envs, V(obs_before_reset) for terminating.
                if "obs_before_reset" in extras and extras["obs_before_reset"] is not None:
                    bs_obs = torch.where(done.unsqueeze(-1) > 0.5, extras["obs_before_reset"], obs)
                else:
                    bs_obs = obs
                bs_obs_norm = self._normalize(bs_obs)
                next_value_buf[t] = self.target_critic(bs_obs_norm)

            actor_loss = actor_loss + (gamma**t) * rew
            rew_buf[t] = rew.detach()

            # Per-env episode tracking (uses detached reward).
            with torch.no_grad():
                self._ep_rew += rew.detach()
                self._ep_len += 1
                terminated = reset > 0
                if terminated.any():
                    idx = terminated.nonzero(as_tuple=False).squeeze(-1)
                    for i in idx.tolist():
                        self._ep_rew_history.append(float(self._ep_rew[i]))
                        self._ep_len_history.append(int(self._ep_len[i]))
                    self._ep_rew[idx] = 0.0
                    self._ep_len[idx] = 0

        # Terminal bootstrap V(obs_H) added once.
        obs_buf[H] = self._normalize(obs).detach()
        with torch.no_grad():
            terminal_value = self.target_critic(obs_buf[H])
        actor_loss = actor_loss + (gamma**H) * terminal_value

        # We want to MAXIMIZE total return, so loss is the negative mean.
        actor_loss = -actor_loss.mean()

        return actor_loss, obs_buf, rew_buf, done_buf, next_value_buf

    # ------------------------------------------------------------------ #
    # Critic update — td-lambda
    # ------------------------------------------------------------------ #
    def _compute_td_lambda_targets(self, rew_buf, done_buf, next_value_buf, last_value):
        cfg = self.cfg
        H, num_envs = rew_buf.shape
        targets = torch.zeros((H, num_envs), device=self.device)
        gae = torch.zeros(num_envs, device=self.device)
        next_v = last_value
        for t in reversed(range(H)):
            not_done = 1.0 - done_buf[t]
            delta = rew_buf[t] + cfg.gamma * (next_value_buf[t] * not_done + 0.0) - next_v + next_v
            # td-lambda style accumulation (mirrors DiffRL's td_lambda critic loss)
            gae = rew_buf[t] + cfg.gamma * (cfg.lam * gae * not_done + (1.0 - cfg.lam) * next_value_buf[t] * not_done)
            targets[t] = gae
            next_v = next_value_buf[t]
            del delta
        return targets

    def _critic_update(self, obs_buf, rew_buf, done_buf, next_value_buf):
        cfg = self.cfg
        with torch.no_grad():
            last_value = self.target_critic(obs_buf[-1])
            targets = self._compute_td_lambda_targets(rew_buf, done_buf, next_value_buf, last_value)

        flat_obs = obs_buf[:-1].reshape(-1, self.env.num_obs)
        flat_tgt = targets.reshape(-1)
        n = flat_obs.shape[0]

        critic_losses = []
        for _ in range(cfg.critic_iterations):
            perm = torch.randperm(n, device=self.device)
            batch_size = max(1, n // cfg.num_batch)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                pred = self.critic(flat_obs[idx])
                loss = (pred - flat_tgt[idx]).pow(2).mean()
                self.critic_opt.zero_grad()
                loss.backward()
                if cfg.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_norm)
                self.critic_opt.step()
                critic_losses.append(float(loss.detach()))

        # Polyak update on target critic.
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
                tp.copy_(cfg.target_critic_alpha * p + (1.0 - cfg.target_critic_alpha) * tp)

        return float(np.mean(critic_losses)) if critic_losses else float("nan")

    # ------------------------------------------------------------------ #
    # Train loop
    # ------------------------------------------------------------------ #
    def train(self):
        cfg = self.cfg
        for epoch in range(cfg.max_epochs):
            # LR schedule.
            self._set_optimizer_lr(self.actor_opt, self._current_lr(epoch, cfg.actor_lr))
            self._set_optimizer_lr(self.critic_opt, self._current_lr(epoch, cfg.critic_lr))

            # 1) Actor rollout + loss.
            self.env.no_grad = False
            actor_loss, obs_buf, rew_buf, done_buf, next_value_buf = self._compute_actor_loss_and_traj()
            self.actor_opt.zero_grad()
            actor_loss.backward(retain_graph=True)
            # Grad norm before clipping.
            total_norm = math.sqrt(
                sum((p.grad.norm().item() ** 2) for p in self.actor.parameters() if p.grad is not None)
            )
            if cfg.truncate_grads:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_norm)
            if not math.isfinite(total_norm) or total_norm > 1e6:
                # Diverged step — skip update.
                self.actor_opt.zero_grad()
            else:
                self.actor_opt.step()

            # Detach physics state from the tape — `env.clear_grad` snapshots
            # qpos/qvel and calls `scene.reset_grad()`, so the next horizon
            # starts from the same state with a fresh autograd tape.
            self.env.clear_grad()

            # 2) Critic update.
            critic_loss = self._critic_update(obs_buf, rew_buf, done_buf, next_value_buf)

            # 3) Logging.
            mean_horizon_rew = float(rew_buf.sum(dim=0).mean())
            self.history["epoch"].append(epoch)
            self.history["mean_horizon_reward"].append(mean_horizon_rew)
            self.history["actor_loss"].append(float(actor_loss.detach()))
            self.history["critic_loss"].append(critic_loss)
            self.history["grad_norm"].append(total_norm)
            if self._ep_rew_history:
                self.history["episode_reward"].append(float(np.mean(self._ep_rew_history[-32:])))
                self.history["episode_length"].append(float(np.mean(self._ep_len_history[-32:])))
            else:
                self.history["episode_reward"].append(float("nan"))
                self.history["episode_length"].append(float("nan"))

            if epoch % cfg.log_interval == 0:
                last_ep = self.history["episode_reward"][-1]
                ep_str = f"ep_ret={last_ep:.2f}" if not math.isnan(last_ep) else "ep_ret=nan(no_done)"
                print(
                    f"[epoch {epoch:4d}] actor_loss={actor_loss.item():+8.3f}  "
                    f"horizon_rew={mean_horizon_rew:+8.3f}  {ep_str}  "
                    f"critic_loss={critic_loss:8.3f}  grad_norm={total_norm:7.2f}  "
                    f"actor_lr={self._current_lr(epoch, cfg.actor_lr):.2e}"
                )

            # 4) Save checkpoint / plot periodically.
            if self.log_dir is not None and cfg.save_interval > 0 and (epoch + 1) % cfg.save_interval == 0:
                self._save_checkpoint(epoch)
                self._save_plot()

        if self.log_dir is not None:
            self._save_checkpoint(cfg.max_epochs)
            self._save_plot()

    def _save_checkpoint(self, epoch: int):
        ckpt = {
            "epoch": epoch,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "obs_rms": self.obs_rms.state_dict() if self.obs_rms is not None else None,
            "history": self.history,
        }
        torch.save(ckpt, os.path.join(self.log_dir, f"shac_{epoch:06d}.pt"))

    def _save_plot(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        hist = self.history
        epochs = hist["epoch"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        axes[0].plot(epochs, hist["mean_horizon_reward"], label="horizon reward")
        axes[0].set_title("Mean horizon reward (per H={} window)".format(self.cfg.steps_num))
        axes[0].set_xlabel("epoch")
        axes[0].grid(alpha=0.3)
        axes[1].plot(epochs, hist["episode_reward"], label="episode reward")
        axes[1].set_title("Mean episode reward (most recent 32 episodes)")
        axes[1].set_xlabel("epoch")
        axes[1].grid(alpha=0.3)
        axes[2].plot(epochs, hist["actor_loss"], label="actor loss", color="tab:red")
        axes[2].plot(epochs, hist["critic_loss"], label="critic loss", color="tab:green")
        axes[2].set_title("Losses")
        axes[2].set_xlabel("epoch")
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        axes[3].plot(epochs, hist["grad_norm"], color="tab:orange")
        axes[3].set_title("Actor grad norm")
        axes[3].set_xlabel("epoch")
        axes[3].set_yscale("symlog")
        axes[3].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "training.png"), dpi=110)
        plt.close(fig)
