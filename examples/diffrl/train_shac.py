"""Entry point: train SHAC on a Genesis-backed differentiable RL environment.

Usage:
    python -m examples.diffrl.train_shac --cfg examples/diffrl/cfg/shac/cartpole_swing_up.yaml \
        --logdir runs/shac_cartpole
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

import genesis as gs

# Initialize Genesis BEFORE importing env modules that touch gs.qd_float / gs.tc_float.
parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, required=True, help="YAML config path.")
parser.add_argument("--logdir", type=str, default=None, help="Output directory for ckpts/plots.")
parser.add_argument(
    "--backend",
    type=str,
    default="cpu",
    choices=["cpu", "gpu"],
    help="Genesis backend (cpu/gpu).",
)
parser.add_argument(
    "--precision",
    type=str,
    default="32",
    choices=["32", "64"],
    help="Floating-point precision.",
)
args = parser.parse_args()

with open(args.cfg, "r") as f:
    cfg = yaml.safe_load(f)

backend = gs.cpu if args.backend == "cpu" else gs.gpu
gs.init(backend=backend, precision=args.precision, logging_level="warning")

from examples.diffrl.algorithms import SHAC, SHACConfig  # noqa: E402
from examples.diffrl.envs import CartPoleSwingUpEnv  # noqa: E402


ENV_REGISTRY = {
    "CartPoleSwingUpEnv": CartPoleSwingUpEnv,
}


def main():
    env_name = cfg["env"]["name"]
    if env_name not in ENV_REGISTRY:
        raise ValueError(f"Unknown env: {env_name}")
    env_cls = ENV_REGISTRY[env_name]

    trainer_cfg = cfg["trainer"]
    actor_cfg = cfg["actor"]
    critic_cfg = cfg["critic"]
    env_cfg = cfg["env"]

    env = env_cls(
        num_envs=trainer_cfg["num_actors"],
        episode_length=env_cfg.get("episode_length", 240),
        seed=trainer_cfg.get("seed", 0),
        no_grad=False,  # set per-call by SHAC
        render=False,
        device="cpu" if args.backend == "cpu" else "cuda",
        stochastic_init=env_cfg.get("stochastic_init", True),
    )

    log_dir = args.logdir or (Path("runs") / f"shac_{env_name}").as_posix()
    os.makedirs(log_dir, exist_ok=True)

    shac_cfg = SHACConfig(
        actor_units=actor_cfg["units"],
        critic_units=critic_cfg["units"],
        activation=actor_cfg.get("activation", "elu"),
        actor_stochastic=actor_cfg.get("stochastic", True),
        logstd_init=actor_cfg.get("logstd_init", -1.0),
        max_epochs=trainer_cfg.get("max_epochs", 500),
        steps_num=trainer_cfg.get("steps_num", 32),
        num_actors=trainer_cfg.get("num_actors", 64),
        actor_lr=trainer_cfg.get("actor_lr", 2e-3),
        critic_lr=trainer_cfg.get("critic_lr", 2e-3),
        lr_schedule=trainer_cfg.get("lr_schedule", "linear"),
        betas=tuple(trainer_cfg.get("betas", (0.7, 0.95))),
        gamma=trainer_cfg.get("gamma", 0.99),
        lam=trainer_cfg.get("lam", 0.95),
        grad_norm=trainer_cfg.get("grad_norm", 1.0),
        truncate_grads=trainer_cfg.get("truncate_grads", True),
        critic_iterations=trainer_cfg.get("critic_iterations", 16),
        critic_method=trainer_cfg.get("critic_method", "td-lambda"),
        num_batch=trainer_cfg.get("num_batch", 4),
        target_critic_alpha=trainer_cfg.get("target_critic_alpha", 0.2),
        obs_rms=trainer_cfg.get("obs_rms", True),
        save_interval=trainer_cfg.get("save_interval", 100),
        log_interval=trainer_cfg.get("log_interval", 1),
        device="cpu" if args.backend == "cpu" else "cuda",
        seed=trainer_cfg.get("seed", 0),
    )

    trainer = SHAC(env, shac_cfg, log_dir=log_dir)
    trainer.train()


if __name__ == "__main__":
    main()
