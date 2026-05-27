"""Eval a trained policy checkpoint on a Genesis differentiable-RL env.

Loads `actor` + `obs_rms` from a `.pt` checkpoint, runs N deterministic
episodes through a single-env scene with an off-screen camera, and writes one
mp4 per episode to `<output_dir>/eval_ep{i}.mp4`.

Importable as `run_eval(...)` so the trainer can call it every `eval_interval`
epochs during training.

Usage:
    python -m examples.diffrl.eval \\
        --ckpt runs/cartpole/shac_000500.pt \\
        --cfg  examples/diffrl/cfg/shac/cartpole_swing_up.yaml \\
        --output_dir runs/cartpole/eval \\
        --backend gpu --precision 32 \\
        --episodes 5
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
import yaml

import genesis as gs


def run_eval(
    actor,
    obs_rms,
    env,
    n_episodes: int,
    output_dir: str | Path,
    fps: int = 60,
    deterministic: bool = True,
    init_states: list | None = None,
):
    """Run `n_episodes` deterministic rollouts on `env` (must have a
    single env and `env.cam` set) and write one mp4 per episode to
    `output_dir/eval_ep{i}.mp4`. Returns a list of per-episode return.

    The caller is responsible for putting `actor` in `.eval()` mode and
    passing a usable `obs_rms` (or `None`).

    `init_states`: optional list of `(qpos, qvel)` tensor pairs. When given,
    episode `ep` starts from `init_states[ep]` instead of a fresh (possibly
    stochastic) `env.reset()`. Passing the SAME frozen set on every call
    makes eval reproducible across epochs — eval@epoch_a and eval@epoch_b
    then start each episode from identical states, so the eval-return curve
    reflects policy change rather than init-noise change.
    """
    assert env.num_envs == 1, f"eval expects num_envs=1, got {env.num_envs}"
    assert env.cam is not None, "env must be built with enable_camera=True"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = next(actor.parameters()).device
    actor_was_training = actor.training
    actor.eval()

    # Disable the env's in-step auto-reset for the duration of eval: otherwise
    # the terminal step resets the env to the init pose BEFORE we render, so
    # the last video frame shows the reset pose instead of the true terminal
    # state. We break on `reset.any()` ourselves, so no reset is needed here.
    prev_auto_reset = getattr(env, "auto_reset", True)
    env.auto_reset = False

    def _reset_to(qpos, qvel):
        # Mirror GenesisDiffRLEnv.reset minus the stochastic-init path.
        env.robot.set_dofs_position(qpos.to(env._torch_device))
        env.robot.set_dofs_velocity(qvel.to(env._torch_device))
        env.progress_buf[:] = 0
        env.compute_observations()
        return env.obs_buf

    episode_returns: list[float] = []
    try:
        for ep in range(n_episodes):
            if init_states is not None:
                qpos, qvel = init_states[ep % len(init_states)]
                obs = _reset_to(qpos, qvel)
            else:
                obs = env.reset()
            ep_ret = 0.0
            steps = 0
            # If the env defines `update_camera` (e.g. hopper tracks its torso
            # in x), keep it framed throughout the rollout. Envs without it
            # (cartpole) leave the camera static.
            cam_follows = hasattr(env, "update_camera")
            if cam_follows:
                env.update_camera()
            env.cam.start_recording()
            with torch.no_grad():
                for _ in range(env.episode_length):
                    obs_in = obs.to(device)
                    if obs_rms is not None:
                        obs_in = obs_rms.normalize(obs_in)
                    action = actor(obs_in, deterministic=deterministic)
                    obs, rew, reset, _ = env.step(action)
                    if cam_follows:
                        env.update_camera()
                    env.cam.render()
                    ep_ret += float(rew.detach().sum())
                    steps += 1
                    if bool(reset.any()):
                        break
            video_path = output_dir / f"eval_ep{ep}.mp4"
            env.cam.stop_recording(save_to_filename=str(video_path), fps=fps)
            print(f"  ep {ep}: return={ep_ret:+.2f}  length={steps}  saved={video_path}")
            episode_returns.append(ep_ret)
    finally:
        env.auto_reset = prev_auto_reset
        if actor_was_training:
            actor.train()

    return episode_returns


def _load_ckpt_and_build(ckpt_path: Path, cfg_path: Path, backend: str, precision: str):
    """gs.init + ckpt load + build a 1-env CartPoleSwingUp scene with camera."""
    backend_enum = gs.cpu if backend == "cpu" else gs.gpu
    gs.init(backend=backend_enum, precision=precision, logging_level="warning")

    # Imports must happen AFTER gs.init so the module-level dtype constants resolve correctly.
    from examples.diffrl.envs import CartPoleSwingUpEnv
    from examples.diffrl.models import ActorStochasticMLP, ActorDeterministicMLP
    from examples.diffrl.running_mean_std import RunningMeanStd

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg["env"]
    actor_cfg = cfg["actor"]

    device = "cpu" if backend == "cpu" else "cuda"

    env = CartPoleSwingUpEnv(
        num_envs=1,
        episode_length=env_cfg.get("episode_length", 240),
        seed=0,
        no_grad=True,  # eval is grad-free
        render=False,
        device=device,
        stochastic_init=env_cfg.get("stochastic_init", True),
        enable_camera=True,
    )

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    if actor_cfg.get("stochastic", True):
        actor = ActorStochasticMLP(
            env.num_obs,
            env.num_actions,
            actor_cfg["units"],
            actor_cfg.get("activation", "elu"),
            actor_cfg.get("logstd_init", -1.0),
        ).to(device)
    else:
        actor = ActorDeterministicMLP(
            env.num_obs,
            env.num_actions,
            actor_cfg["units"],
            actor_cfg.get("activation", "elu"),
        ).to(device)
    actor.load_state_dict(ckpt["actor"])

    obs_rms = None
    if ckpt.get("obs_rms") is not None:
        obs_rms = RunningMeanStd(env.num_obs, device=device)
        obs_rms.load_state_dict(ckpt["obs_rms"])

    epoch = ckpt.get("epoch", "?")
    print(f"Loaded checkpoint epoch={epoch} from {ckpt_path}")
    return env, actor, obs_rms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to save mp4s. Default: <ckpt_dir>/eval_<ckpt_stem>/"
    )
    parser.add_argument("--backend", type=str, default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--precision", type=str, default="32", choices=["32", "64"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt).resolve()
    cfg_path = Path(args.cfg).resolve()
    if args.output_dir is None:
        output_dir = ckpt_path.parent / f"eval_{ckpt_path.stem}"
    else:
        output_dir = Path(args.output_dir).resolve()

    env, actor, obs_rms = _load_ckpt_and_build(ckpt_path, cfg_path, args.backend, args.precision)
    returns = run_eval(actor, obs_rms, env, args.episodes, output_dir, fps=args.fps)
    mean = sum(returns) / len(returns) if returns else float("nan")
    print("\n=== Summary ===")
    print(f"  episodes: {len(returns)}")
    print(f"  mean_return: {mean:+.3f}")
    if returns:
        print(f"  min: {min(returns):+.3f}   max: {max(returns):+.3f}")


if __name__ == "__main__":
    main()
