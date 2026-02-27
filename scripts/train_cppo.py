"""
train_cppo.py
=============
Entry point for training the safety-constrained PPO agent (C-PPO Lagrangian).

Usage example::

    python scripts/train_cppo.py \\
        --env_id Hopper-v4 \\
        --seed 0 \\
        --total_timesteps 1000000 \\
        --save_dir runs/cppo_hopper_s0 \\
        --eval_every 10000 \\
        --cost_limit 0.1 \\
        --cost_fn action_magnitude

The script:
    1. Parses CLI arguments (CLI overrides configs/cppo.yaml).
    2. Loads hyperparameters from configs/cppo.yaml.
    3. Constructs the environment, CPPOLagrangianAgent, and buffer (store_costs=True).
    4. Runs the C-PPO training loop:
         - Collects rollout with per-step costs from the chosen cost function.
         - Computes reward GAE and cost GAE.
         - Calls agent.update(buffer, avg_episode_cost).
         - Lambda is updated inside agent.update after all gradient epochs.
    5. Evaluates periodically (both return and cost).
    6. Logs all metrics to <save_dir>/metrics.csv (includes lambda column).
    7. Saves checkpoints to <save_dir>/checkpoints/.
    8. Saves resolved config to <save_dir>/config.yaml.

CSV columns written:
    step, episode_return, episode_cost, episode_length,
    eval_return_mean, eval_return_std, eval_cost_mean, eval_cost_std,
    lambda, policy_loss, cost_policy_loss, value_loss, cost_value_loss,
    entropy, approx_kl, clip_fraction
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

from robot_safe_ppo.cppo_lagrangian import (
    CPPOLagrangianAgent,
    _save_cppo_checkpoint,
    get_cost_fn,
)
from robot_safe_ppo.buffers import RolloutBuffer
from robot_safe_ppo.utils import set_seeds, load_config, MetricLogger
from robot_safe_ppo.eval import evaluate_policy


# ---------------------------------------------------------------------------
# CSV schema — fixed upfront so MetricLogger writes the header immediately
# ---------------------------------------------------------------------------
CSV_FIELDNAMES = [
    "step",
    "episode_return",
    "episode_cost",
    "episode_length",
    "eval_return_mean",
    "eval_return_std",
    "eval_cost_mean",
    "eval_cost_std",
    "lambda",
    "policy_loss",
    "cost_policy_loss",
    "value_loss",
    "cost_value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the C-PPO training run."""
    parser = argparse.ArgumentParser(
        description="Train a Lagrangian-constrained PPO agent on a MuJoCo environment."
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default="Hopper-v4",
        help="Gymnasium environment ID (default: Hopper-v4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0).",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=1_000_000,
        help="Total environment steps to train for (default: 1e6).",
    )
    parser.add_argument(
        "--save_dir",
        type=pathlib.Path,
        default=pathlib.Path("runs/cppo_default"),
        help="Directory for checkpoints, metrics, and config (default: runs/cppo_default).",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=10_000,
        help="Evaluate every N environment steps (default: 10 000).",
    )
    parser.add_argument(
        "--cost_limit",
        type=float,
        default=0.1,
        help="Safety threshold d: constraint is E[cost] <= cost_limit (default: 0.1).",
    )
    parser.add_argument(
        "--cost_fn",
        type=str,
        default="action_magnitude",
        choices=["action_magnitude", "torso_angle"],
        help="Which cost function to use (default: action_magnitude).",
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent.parent / "configs" / "cppo.yaml",
        help="Path to base YAML config (default: configs/cppo.yaml).",
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
    cfg["cost_limit"] = args.cost_limit
    cfg["cost_fn"] = args.cost_fn

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
    # 5. Cost function
    # ------------------------------------------------------------------
    cost_fn = get_cost_fn(args.cost_fn, cfg)

    # ------------------------------------------------------------------
    # 6. Agent and buffer
    # ------------------------------------------------------------------
    agent = CPPOLagrangianAgent(obs_dim, act_dim, cfg)

    n_steps: int = min(int(cfg.get("n_steps", 2048)), args.total_timesteps)
    buffer = RolloutBuffer(
        buffer_size=n_steps,
        obs_dim=obs_dim,
        act_dim=act_dim,
        gamma=float(cfg.get("gamma", 0.99)),
        gae_lambda=float(cfg.get("gae_lambda", 0.95)),
        store_costs=True,
    )

    # ------------------------------------------------------------------
    # 7. Metric logger  (creates metrics.csv on disk immediately)
    # ------------------------------------------------------------------
    logger = MetricLogger(save_dir / "metrics.csv", fieldnames=CSV_FIELDNAMES)

    # ------------------------------------------------------------------
    # 8. Training schedule
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
    episode_cost: float = 0.0
    episode_length: int = 0

    # Track completed-episode costs per rollout for the lambda update
    rollout_episode_costs: list[float] = []

    print(
        f"[train_cppo] env={args.env_id}  seed={args.seed}  "
        f"total_timesteps={args.total_timesteps}  n_steps={n_steps}  "
        f"num_updates={num_updates}  cost_limit={args.cost_limit}  "
        f"cost_fn={args.cost_fn}  save_dir={save_dir}"
    )

    # ------------------------------------------------------------------
    # 9. Main loop
    # ------------------------------------------------------------------
    for _update_idx in range(num_updates):
        buffer.reset()
        rollout_episode_costs = []

        # ---- Rollout collection --------------------------------------
        for _ in range(n_steps):
            action, log_prob, value = agent.select_action(obs)
            cost_value = agent.get_cost_value(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Compute per-step cost from (obs, action, next_obs)
            step_cost = cost_fn(obs, action, next_obs)

            buffer.add(
                obs, action, float(reward), done,
                log_prob, value,
                cost=step_cost, cost_value=cost_value,
            )

            obs = next_obs
            global_step += 1
            episode_return += float(reward)
            episode_cost += float(step_cost)
            episode_length += 1

            if done:
                logger.log(
                    {
                        "step": global_step,
                        "episode_return": round(episode_return, 4),
                        "episode_cost": round(episode_cost, 4),
                        "episode_length": episode_length,
                        "eval_return_mean": float("nan"),
                        "eval_return_std": float("nan"),
                        "eval_cost_mean": float("nan"),
                        "eval_cost_std": float("nan"),
                        "lambda": float("nan"),
                        "policy_loss": float("nan"),
                        "cost_policy_loss": float("nan"),
                        "value_loss": float("nan"),
                        "cost_value_loss": float("nan"),
                        "entropy": float("nan"),
                        "approx_kl": float("nan"),
                        "clip_fraction": float("nan"),
                    }
                )
                rollout_episode_costs.append(episode_cost)
                episode_return = 0.0
                episode_cost = 0.0
                episode_length = 0
                obs, _ = env.reset()

        # ---- Bootstrap last values -----------------------------------
        _, _, last_value = agent.select_action(obs)
        last_cost_value = agent.get_cost_value(obs)
        buffer.compute_advantages(last_value)
        buffer.compute_cost_advantages(last_cost_value)

        # ---- Compute avg episode cost for lambda update --------------
        # Use mean completed episode cost from this rollout.
        # Fallback: mean per-step cost if no episode completed.
        if rollout_episode_costs:
            avg_episode_cost = float(sum(rollout_episode_costs) / len(rollout_episode_costs))
        else:
            avg_episode_cost = float(buffer.costs.mean())

        # ---- LR annealing --------------------------------------------
        if lr_anneal:
            frac = max(0.0, 1.0 - global_step / args.total_timesteps)
            for pg in agent.optimizer.param_groups:
                pg["lr"] = lr_init * frac

        # ---- C-PPO update (lambda updated inside) --------------------
        update_metrics = agent.update(buffer, avg_episode_cost)

        # ---- Periodic evaluation + checkpoint ------------------------
        if global_step >= next_eval_step:
            eval_results = evaluate_policy(
                agent,
                args.env_id,
                n_episodes=eval_episodes,
                eval_seed=eval_seed,
                compute_cost=True,
                cost_fn=cost_fn,
            )

            logger.log(
                {
                    "step": global_step,
                    "episode_return": float("nan"),
                    "episode_cost": float("nan"),
                    "episode_length": float("nan"),
                    "eval_return_mean": round(eval_results["eval_return_mean"], 4),
                    "eval_return_std": round(eval_results["eval_return_std"], 4),
                    "eval_cost_mean": round(eval_results.get("eval_cost_mean", float("nan")), 4),
                    "eval_cost_std": round(eval_results.get("eval_cost_std", float("nan")), 4),
                    "lambda": round(update_metrics["lambda"], 6),
                    "policy_loss": round(update_metrics["policy_loss"], 6),
                    "cost_policy_loss": round(update_metrics["cost_policy_loss"], 6),
                    "value_loss": round(update_metrics["value_loss"], 6),
                    "cost_value_loss": round(update_metrics["cost_value_loss"], 6),
                    "entropy": round(update_metrics["entropy"], 6),
                    "approx_kl": round(update_metrics["approx_kl"], 6),
                    "clip_fraction": round(update_metrics["clip_fraction"], 6),
                }
            )

            ckpt_path = ckpt_dir / f"step_{global_step:08d}.pt"
            _save_cppo_checkpoint(
                agent,
                ckpt_path,
                metadata={
                    "step": global_step,
                    "seed": args.seed,
                    "env_id": args.env_id,
                    "eval_return_mean": eval_results["eval_return_mean"],
                    "eval_return_std": eval_results["eval_return_std"],
                    "eval_cost_mean": eval_results.get("eval_cost_mean", float("nan")),
                    "lambda": update_metrics["lambda"],
                },
            )

            print(
                f"[step {global_step:>8d}/{args.total_timesteps}]  "
                f"eval_return={eval_results['eval_return_mean']:.1f}"
                f" \u00b1 {eval_results['eval_return_std']:.1f}  "
                f"eval_cost={eval_results.get('eval_cost_mean', float('nan')):.3f}  "
                f"lambda={update_metrics['lambda']:.4f}  "
                f"ckpt={ckpt_path.name}"
            )

            next_eval_step += args.eval_every

    # ------------------------------------------------------------------
    # 10. Finalise
    # ------------------------------------------------------------------
    logger.close()
    env.close()
    print(f"[train_cppo] Done. Artefacts saved to: {save_dir}")


if __name__ == "__main__":
    main()
