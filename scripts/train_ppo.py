"""
train_ppo.py
============
Entry point for training the baseline PPO agent.

Usage example::

    python scripts/train_ppo.py \\
        --env_id Hopper-v4 \\
        --seed 0 \\
        --total_timesteps 1000000 \\
        --save_dir runs/ppo_hopper_s0 \\
        --eval_every 10000

The script:
    1. Parses CLI arguments; merges with configs/ppo.yaml (CLI takes precedence).
    2. Resolves n_steps = min(cfg["n_steps"], total_timesteps) to avoid
       allocating buffers larger than the training budget.
    3. Sets global random seeds and creates the Gymnasium environment.
    4. Constructs PPOAgent and RolloutBuffer.
    5. Opens MetricLogger (creates metrics.csv immediately on disk).
    6. Saves a copy of the resolved config to <save_dir>/config.yaml.
    7. Runs num_updates = total_timesteps // n_steps rollout-update iterations.
    8. After each rollout, checks whether an eval step threshold was crossed;
       if so, runs deterministic evaluation and saves a checkpoint.
    9. Closes the logger and prints a completion summary.

CSV columns written:
    step, episode_return, episode_length,
    eval_return_mean, eval_return_std,
    policy_loss, value_loss, entropy, approx_kl, clip_fraction
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# ---------------------------------------------------------------------------
# Make the src/ layout importable regardless of CWD or install state
# ---------------------------------------------------------------------------
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import gymnasium as gym
import yaml

from robot_safe_ppo.ppo import PPOAgent
from robot_safe_ppo.buffers import RolloutBuffer
from robot_safe_ppo.utils import set_seeds, load_config, MetricLogger, save_checkpoint
from robot_safe_ppo.eval import evaluate_policy


# ---------------------------------------------------------------------------
# CSV schema — fixed upfront so MetricLogger writes the header immediately
# ---------------------------------------------------------------------------
CSV_FIELDNAMES = [
    "step",
    "episode_return",
    "episode_length",
    "eval_return_mean",
    "eval_return_std",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the PPO training run."""
    parser = argparse.ArgumentParser(
        description="Train a baseline PPO agent on a MuJoCo environment."
    )
    parser.add_argument("--env_id", type=str, default="Hopper-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument(
        "--save_dir",
        type=pathlib.Path,
        default=pathlib.Path("runs/ppo_default"),
    )
    parser.add_argument("--eval_every", type=int, default=10_000)
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent.parent / "configs" / "ppo.yaml",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load and merge config
    # ------------------------------------------------------------------
    cfg: dict = load_config(args.config)
    # CLI args override YAML values
    cfg["env_id"] = args.env_id
    cfg["seed"] = args.seed
    cfg["total_timesteps"] = args.total_timesteps
    cfg["save_dir"] = str(args.save_dir)
    cfg["eval_every"] = args.eval_every

    # ------------------------------------------------------------------
    # 2. Directories and saved config
    # ------------------------------------------------------------------
    save_dir = pathlib.Path(cfg["save_dir"])
    ckpt_dir = save_dir / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # ------------------------------------------------------------------
    # 3. Seeds
    # ------------------------------------------------------------------
    set_seeds(args.seed)

    # ------------------------------------------------------------------
    # 4. Environment
    # ------------------------------------------------------------------
    env = gym.make(args.env_id)
    obs, _ = env.reset(seed=args.seed)
    obs_dim: int = env.observation_space.shape[0]
    act_dim: int = env.action_space.shape[0]

    # ------------------------------------------------------------------
    # 5. Agent and buffer
    # ------------------------------------------------------------------
    agent = PPOAgent(obs_dim, act_dim, cfg)

    n_steps: int = min(int(cfg.get("n_steps", 2048)), args.total_timesteps)
    buffer = RolloutBuffer(
        buffer_size=n_steps,
        obs_dim=obs_dim,
        act_dim=act_dim,
        gamma=float(cfg.get("gamma", 0.99)),
        gae_lambda=float(cfg.get("gae_lambda", 0.95)),
    )

    # ------------------------------------------------------------------
    # 6. Metric logger  (creates metrics.csv on disk immediately)
    # ------------------------------------------------------------------
    logger = MetricLogger(save_dir / "metrics.csv", fieldnames=CSV_FIELDNAMES)

    # ------------------------------------------------------------------
    # 7. Training schedule
    # ------------------------------------------------------------------
    num_updates: int = max(1, args.total_timesteps // n_steps)
    lr_init: float = float(cfg.get("lr", 3e-4))
    lr_anneal: bool = bool(cfg.get("lr_anneal", True))
    eval_episodes: int = int(cfg.get("eval_episodes", 10))
    eval_seed: int = int(cfg.get("eval_seed", 1000))

    global_step: int = 0
    next_eval_step: int = args.eval_every

    # Running episode accumulators
    episode_return: float = 0.0
    episode_length: int = 0

    print(
        f"[train_ppo] env={args.env_id}  seed={args.seed}  "
        f"total_timesteps={args.total_timesteps}  n_steps={n_steps}  "
        f"num_updates={num_updates}  save_dir={save_dir}"
    )

    # ------------------------------------------------------------------
    # 8. Main loop
    # ------------------------------------------------------------------
    for _update_idx in range(num_updates):
        buffer.reset()

        # ---- Rollout collection --------------------------------------
        for _ in range(n_steps):
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.add(obs, action, float(reward), done, log_prob, value)

            obs = next_obs
            global_step += 1
            episode_return += float(reward)
            episode_length += 1

            if done:
                logger.log(
                    {
                        "step": global_step,
                        "episode_return": round(episode_return, 4),
                        "episode_length": episode_length,
                        "eval_return_mean": float("nan"),
                        "eval_return_std": float("nan"),
                        "policy_loss": float("nan"),
                        "value_loss": float("nan"),
                        "entropy": float("nan"),
                        "approx_kl": float("nan"),
                        "clip_fraction": float("nan"),
                    }
                )
                episode_return = 0.0
                episode_length = 0
                obs, _ = env.reset()

        # ---- Bootstrap last value ------------------------------------
        _, _, last_value = agent.select_action(obs)
        buffer.compute_advantages(last_value)

        # ---- LR annealing --------------------------------------------
        if lr_anneal:
            frac = max(0.0, 1.0 - global_step / args.total_timesteps)
            for pg in agent.optimizer.param_groups:
                pg["lr"] = lr_init * frac

        # ---- PPO update ----------------------------------------------
        update_metrics = agent.update(buffer)

        # ---- Periodic evaluation + checkpoint ------------------------
        if global_step >= next_eval_step:
            eval_results = evaluate_policy(
                agent,
                args.env_id,
                n_episodes=eval_episodes,
                eval_seed=eval_seed,
            )

            logger.log(
                {
                    "step": global_step,
                    "episode_return": float("nan"),
                    "episode_length": float("nan"),
                    "eval_return_mean": round(eval_results["eval_return_mean"], 4),
                    "eval_return_std": round(eval_results["eval_return_std"], 4),
                    "policy_loss": round(update_metrics["policy_loss"], 6),
                    "value_loss": round(update_metrics["value_loss"], 6),
                    "entropy": round(update_metrics["entropy"], 6),
                    "approx_kl": round(update_metrics["approx_kl"], 6),
                    "clip_fraction": round(update_metrics["clip_fraction"], 6),
                }
            )

            ckpt_path = ckpt_dir / f"step_{global_step:08d}.pt"
            save_checkpoint(
                agent,
                ckpt_path,
                metadata={
                    "step": global_step,
                    "seed": args.seed,
                    "env_id": args.env_id,
                    "eval_return_mean": eval_results["eval_return_mean"],
                    "eval_return_std": eval_results["eval_return_std"],
                },
            )

            print(
                f"[step {global_step:>8d}/{args.total_timesteps}]  "
                f"eval_return={eval_results['eval_return_mean']:.1f}"
                f" \u00b1 {eval_results['eval_return_std']:.1f}  "
                f"ckpt={ckpt_path.name}"
            )

            next_eval_step += args.eval_every

    # ------------------------------------------------------------------
    # 9. Finalise
    # ------------------------------------------------------------------
    logger.close()
    env.close()
    print(f"[train_ppo] Done. Artefacts saved to: {save_dir}")


if __name__ == "__main__":
    main()
