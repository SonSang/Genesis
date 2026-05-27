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
    eval_interval: int = 0  # 0 disables periodic eval rollouts
    eval_episodes: int = 5
    eval_video_fps: int = 60
    device: str = "cpu"
    seed: int = 0


class SHAC:
    def __init__(self, env, cfg: SHACConfig, log_dir: str | None = None, eval_env=None):
        self.env = env
        self.eval_env = eval_env  # optional 1-env scene with camera, used by _run_eval
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
            # Periodic eval rollouts: sparse — recorded only at eval_interval.
            "eval_epoch": [],
            "eval_mean": [],
            "eval_std": [],
        }
        # Per-env running episode trackers
        self._ep_rew = torch.zeros(env.num_envs, device=self.device)
        self._ep_len = torch.zeros(env.num_envs, device=self.device, dtype=torch.long)
        self._ep_rew_history: list[float] = []
        self._ep_len_history: list[int] = []

        # Frozen eval init states: capture `cfg.eval_episodes` starting states
        # ONCE (before training) so every periodic eval starts each episode
        # from identical states. This makes the eval-return curve comparable
        # across epochs (eval@epoch_a vs eval@epoch_b reflects policy change,
        # not per-eval init-noise change).
        self._eval_init_states = None
        if self.eval_env is not None:
            self._eval_init_states = self._make_eval_init_states(cfg.eval_episodes)

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
        # Linear decay from base_lr down to 1e-5 over max_epochs (DiffRL floors
        # at 1e-5 rather than 0).
        return (1e-5 - base_lr) * (epoch / max(1, self.cfg.max_epochs)) + base_lr

    def _set_optimizer_lr(self, opt: torch.optim.Optimizer, lr: float):
        for group in opt.param_groups:
            group["lr"] = lr

    # ------------------------------------------------------------------ #
    # SHAC actor pass
    # ------------------------------------------------------------------ #
    def _compute_actor_loss_and_traj(self):
        """One SHAC horizon, faithful to DiffRL's `compute_actor_loss`.

        Accumulates the per-env objective `-(sum_k gamma^k rew_k + gamma^{i+1} V(s_{i+1}))`
        with proper handling of episodes that terminate mid-horizon: the
        reward accumulator and per-env discount reset on `done`, and a
        terminal value bootstrap is added for each env at its segment end
        (every done env at its done step, plus all envs at the last step).

        The bootstrap `target_critic(obs)` is DIFFERENTIABLE — the actor
        gradient flows through the simulated state into the value function,
        which is the short-horizon value term that lets a length-H rollout
        "see" beyond the horizon (essential when episode_length > H). The
        target-critic weights are frozen (requires_grad=False), so only the
        obs->state->action path contributes gradient.

        Returns `actor_loss` plus the detached (obs, rew, done_mask, next_value)
        buffers the td-lambda critic update reads.
        """
        cfg = self.cfg
        H = cfg.steps_num
        num_envs = self.env.num_envs
        gamma_scalar = cfg.gamma
        max_ep_len = self.env.episode_length

        # Detached buffers for the critic update.
        obs_buf = torch.zeros((H, num_envs, self.env.num_obs), device=self.device)
        rew_buf = torch.zeros((H, num_envs), device=self.device)
        done_mask = torch.zeros((H, num_envs), device=self.device)
        next_value_buf = torch.zeros((H, num_envs), device=self.device)

        # Differentiable rollout accumulators.
        rew_acc = torch.zeros((H + 1, num_envs), device=self.device)
        next_values = torch.zeros((H + 1, num_envs), device=self.device)
        gamma = torch.ones(num_envs, device=self.device)
        actor_loss = torch.tensor(0.0, device=self.device)

        # Freeze the obs-normalizer stats for the whole rollout (DiffRL snapshots
        # obs_rms): normalize with the frozen mean/var so what the actor/critic
        # see doesn't drift mid-horizon, while still updating the running stats
        # for the next iteration. We snapshot the mean/var tensors rather than
        # deepcopy the RunningMeanStd — its internal tensors get promoted to
        # gs.Tensors via `update(obs)` and gs.Tensor doesn't support deepcopy.
        if self.obs_rms is not None:
            snap_mean = self.obs_rms.mean.detach().clone()
            snap_var = self.obs_rms.var.detach().clone()

            def _norm(x):
                return (x - snap_mean) / torch.sqrt(snap_var + 1e-5)
        else:

            def _norm(x):
                return x

        obs = self.env.initialize_trajectory()
        if self.obs_rms is not None:
            with torch.no_grad():
                self.obs_rms.update(obs.detach())
        obs = _norm(obs)

        for t in range(H):
            with torch.no_grad():
                obs_buf[t] = obs.detach()
            actions = self.actor(obs)
            obs, rew, reset, extras = self.env.step(actions)
            if self.obs_rms is not None:
                with torch.no_grad():
                    self.obs_rms.update(obs.detach())
            obs = _norm(obs)

            self._ep_len += 1
            with torch.no_grad():
                done = reset.float()
                done_env_ids = (reset > 0).nonzero(as_tuple=False).squeeze(-1)

            # Differentiable bootstrap from the post-step (post-reset for done
            # envs) state, then override done envs with the value of their true
            # terminal obs (obs_before_reset), or 0 for early termination.
            next_values[t + 1] = self.target_critic(obs)
            obr = extras.get("obs_before_reset") if isinstance(extras, dict) else None
            for env_id in done_env_ids.tolist():
                if obr is None:
                    next_values[t + 1, env_id] = 0.0
                elif not torch.isfinite(obr[env_id]).all() or (obr[env_id].detach().abs() > 1e6).any():
                    next_values[t + 1, env_id] = 0.0
                elif int(self._ep_len[env_id]) < max_ep_len:
                    # Early termination -> no future value to bootstrap.
                    next_values[t + 1, env_id] = 0.0
                else:
                    next_values[t + 1, env_id] = self.target_critic(_norm(obr[env_id]))

            rew_acc[t + 1] = rew_acc[t] + gamma * rew

            if t < H - 1:
                if done_env_ids.numel() > 0:
                    actor_loss = (
                        actor_loss
                        + (
                            -rew_acc[t + 1, done_env_ids]
                            - gamma_scalar * gamma[done_env_ids] * next_values[t + 1, done_env_ids]
                        ).sum()
                    )
            else:
                # Terminate all envs at the end of the horizon.
                actor_loss = actor_loss + (-rew_acc[t + 1, :] - gamma_scalar * gamma * next_values[t + 1, :]).sum()

            gamma = gamma * gamma_scalar
            if done_env_ids.numel() > 0:
                gamma[done_env_ids] = 1.0
                rew_acc[t + 1, done_env_ids] = 0.0

            with torch.no_grad():
                rew_buf[t] = rew.detach()
                done_mask[t] = done if t < H - 1 else torch.ones_like(done)
                next_value_buf[t] = next_values[t + 1].detach()

            # Per-env episode-return logging (raw reward; resets on done).
            with torch.no_grad():
                self._ep_rew += rew.detach()
                if done_env_ids.numel() > 0:
                    for i in done_env_ids.tolist():
                        self._ep_rew_history.append(float(self._ep_rew[i]))
                        self._ep_len_history.append(int(self._ep_len[i]))
                    self._ep_rew[done_env_ids] = 0.0
                    self._ep_len[done_env_ids] = 0

        actor_loss = actor_loss / (H * num_envs)

        return actor_loss, obs_buf, rew_buf, done_mask, next_value_buf

    # ------------------------------------------------------------------ #
    # Critic update — td-lambda
    # ------------------------------------------------------------------ #
    def _compute_td_lambda_targets(self, rew_buf, done_mask, next_value_buf):
        """TD(lambda) critic targets, verbatim from DiffRL's
        `compute_target_values` (the Ai/Bi/lam recursion). `next_value_buf[i]`
        is the bootstrap value V(s_{i+1}); `done_mask[i]` is 1 at episode end
        (and forced to 1 at the last horizon step)."""
        cfg = self.cfg
        H, num_envs = rew_buf.shape
        targets = torch.zeros((H, num_envs), device=self.device)
        Ai = torch.zeros(num_envs, device=self.device)
        Bi = torch.zeros(num_envs, device=self.device)
        lam = torch.ones(num_envs, device=self.device)
        for i in reversed(range(H)):
            not_done = 1.0 - done_mask[i]
            lam = lam * cfg.lam * not_done + done_mask[i]
            Ai = not_done * (
                cfg.lam * cfg.gamma * Ai + cfg.gamma * next_value_buf[i] + (1.0 - lam) / (1.0 - cfg.lam) * rew_buf[i]
            )
            Bi = cfg.gamma * (next_value_buf[i] * done_mask[i] + Bi * not_done) + rew_buf[i]
            targets[i] = (1.0 - cfg.lam) * Ai + lam * Bi
        return targets

    def _critic_update(self, obs_buf, rew_buf, done_mask, next_value_buf):
        cfg = self.cfg
        with torch.no_grad():
            targets = self._compute_td_lambda_targets(rew_buf, done_mask, next_value_buf)

        flat_obs = obs_buf.reshape(-1, self.env.num_obs)
        flat_tgt = targets.reshape(-1)
        n = flat_obs.shape[0]

        # Report the LAST critic-iteration's mean loss (matches DiffRL, which
        # overwrites value_loss every iteration) — averaging over all
        # iterations would inflate it with the high early-iteration values.
        value_loss = float("nan")
        for _ in range(cfg.critic_iterations):
            perm = torch.randperm(n, device=self.device)
            batch_size = max(1, n // cfg.num_batch)
            total, cnt = 0.0, 0
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                pred = self.critic(flat_obs[idx])
                loss = (pred - flat_tgt[idx]).pow(2).mean()
                self.critic_opt.zero_grad()
                loss.backward()
                # DiffRL "ugly fix" for occasional sim-induced NaN grads.
                for p in self.critic.parameters():
                    if p.grad is not None:
                        p.grad.nan_to_num_(0.0, 0.0, 0.0)
                if cfg.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_norm)
                self.critic_opt.step()
                total += float(loss.detach())
                cnt += 1
            value_loss = total / max(1, cnt)

        # Polyak update on target critic.
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
                tp.copy_(cfg.target_critic_alpha * p + (1.0 - cfg.target_critic_alpha) * tp)

        return value_loss

    # ------------------------------------------------------------------ #
    # Train loop
    # ------------------------------------------------------------------ #
    def train(self):
        cfg = self.cfg

        # Baseline eval on the untrained (random) policy BEFORE any update, so
        # the eval curve shows how far training moves performance from random.
        # Uses the same frozen init states as every later eval (labelled epoch
        # 0 / dir eval_epoch000000).
        if self.eval_env is not None and self.log_dir is not None:
            self._run_eval(0)

        for epoch in range(cfg.max_epochs):
            # LR schedule.
            self._set_optimizer_lr(self.actor_opt, self._current_lr(epoch, cfg.actor_lr))
            self._set_optimizer_lr(self.critic_opt, self._current_lr(epoch, cfg.critic_lr))

            # 1) Actor rollout + loss.
            self.env.no_grad = False
            actor_loss, obs_buf, rew_buf, done_buf, next_value_buf = self._compute_actor_loss_and_traj()

            self.actor_opt.zero_grad()
            # `scene.backward` snapshots the terminal state, runs the backward
            # unroll (which rewinds physics to step 0), then RESTORES the
            # terminal state. Without it, the plain `loss.backward()` left the
            # sim at the step-0 init pose, so the subsequent `clear_grad()`
            # snapshot captured the init state and every horizon restarted from
            # init instead of continuing the episode. (defaults retain_graph=True)
            self.env.scene.backward(actor_loss)
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

            # 4) Live plot refresh every log_interval — lets the user watch
            # `training.png` update during a long training run.
            if self.log_dir is not None and epoch % cfg.log_interval == 0:
                self._save_plot()

            # 5) Checkpoint at save_interval (heavier; pickled networks +
            # optimizer state).
            if self.log_dir is not None and cfg.save_interval > 0 and (epoch + 1) % cfg.save_interval == 0:
                self._save_checkpoint(epoch)

            # 6) Periodic eval rollout (records mp4s for the current policy).
            if (
                self.eval_env is not None
                and self.log_dir is not None
                and cfg.eval_interval > 0
                and (epoch + 1) % cfg.eval_interval == 0
            ):
                self._run_eval(epoch)

        if self.log_dir is not None:
            self._save_checkpoint(cfg.max_epochs)
            # Final eval on the fully-trained policy (labelled epoch
            # cfg.max_epochs), then refresh the plot so it includes this point.
            if self.eval_env is not None:
                self._run_eval(cfg.max_epochs)
            self._save_plot()

    def _make_eval_init_states(self, n: int):
        """Capture `n` frozen (qpos, qvel) init states from the eval env's
        (stochastic) reset. Called once at construction; the returned list is
        reused for every `_run_eval` so eval episodes start from fixed states
        and stay comparable across epochs."""
        states = []
        for _ in range(n):
            self.eval_env.reset()
            qpos = self.eval_env.robot.get_dofs_position().detach().clone()
            qvel = self.eval_env.robot.get_dofs_velocity().detach().clone()
            states.append((qpos, qvel))
        return states

    def _run_eval(self, epoch: int):
        """Run `cfg.eval_episodes` deterministic rollouts on `self.eval_env`
        and dump one mp4 per episode to `<log_dir>/eval/eval_epoch{epoch:06d}/`.
        Records (mean, std) of the per-episode returns to `self.history`."""
        from ..eval import run_eval

        cfg = self.cfg
        out_dir = os.path.join(self.log_dir, "eval", f"eval_epoch{epoch:06d}")
        print(f"[eval] epoch={epoch}  recording {cfg.eval_episodes} episodes → {out_dir}")
        returns = run_eval(
            actor=self.actor,
            obs_rms=self.obs_rms,
            env=self.eval_env,
            n_episodes=cfg.eval_episodes,
            output_dir=out_dir,
            fps=cfg.eval_video_fps,
            deterministic=True,
            init_states=self._eval_init_states,
        )
        if returns:
            arr = np.asarray(returns, dtype=np.float64)
            mean, std, mn, mx = float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())
            self.history["eval_epoch"].append(epoch)
            self.history["eval_mean"].append(mean)
            self.history["eval_std"].append(std)
            print(f"[eval] epoch={epoch}  mean={mean:+.2f}  std={std:.2f}  min={mn:+.2f}  max={mx:+.2f}")

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
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        axes = axes.flatten()
        # [0] horizon reward
        axes[0].plot(epochs, hist["mean_horizon_reward"], color="tab:blue")
        axes[0].set_title("Mean horizon reward (per H={} window)".format(self.cfg.steps_num))
        axes[0].set_xlabel("epoch")
        axes[0].grid(alpha=0.3)
        # [1] train episode reward
        axes[1].plot(epochs, hist["episode_reward"], color="tab:blue")
        axes[1].set_title("Train episode reward (most recent 32 episodes)")
        axes[1].set_xlabel("epoch")
        axes[1].grid(alpha=0.3)
        # [2] eval (deterministic) reward — mean ± std over eval_episodes rollouts
        if hist["eval_epoch"]:
            eval_ep = np.asarray(hist["eval_epoch"])
            eval_mean = np.asarray(hist["eval_mean"])
            eval_std = np.asarray(hist["eval_std"])
            axes[2].plot(eval_ep, eval_mean, color="tab:red", marker="o", markersize=3, label="mean")
            axes[2].fill_between(
                eval_ep, eval_mean - eval_std, eval_mean + eval_std, color="tab:red", alpha=0.2, label="±1σ"
            )
            axes[2].legend(loc="lower right", fontsize=8)
        axes[2].set_title(f"Eval episode reward (deterministic, {self.cfg.eval_episodes} rollouts)")
        axes[2].set_xlabel("epoch")
        axes[2].grid(alpha=0.3)
        # [3] actor loss (linear — can be negative, so no log scale)
        axes[3].plot(epochs, hist["actor_loss"], color="tab:red")
        axes[3].set_title("Actor loss")
        axes[3].set_xlabel("epoch")
        axes[3].grid(alpha=0.3)
        # [4] grad norm
        axes[4].plot(epochs, hist["grad_norm"], color="tab:orange")
        axes[4].set_title("Actor grad norm")
        axes[4].set_xlabel("epoch")
        axes[4].set_yscale("symlog")
        axes[4].grid(alpha=0.3)
        # [5] critic loss (log scale)
        axes[5].plot(epochs, hist["critic_loss"], color="tab:green")
        axes[5].set_title("Critic loss (log scale)")
        axes[5].set_xlabel("epoch")
        axes[5].set_yscale("log")
        axes[5].grid(alpha=0.3, which="both")
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "training.png"), dpi=110)
        plt.close(fig)
