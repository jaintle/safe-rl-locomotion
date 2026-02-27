"""
train_ppo.py
============
Entry point for training the baseline PPO agent.

Usage example::

    python scripts/train_ppo.py \
        --env_id Hopper-v4 \
        --seed 0 \
        --total_timesteps 1000000 \
        --save_dir runs/ppo_hopper_s0 \
        --eval_every 10000

The script:
    1. Parses CLI arguments (override precedence over the YAML config).
    2. Loads hyperparameters from configs/ppo.yaml.
    3. Constructs the environment, agent, and buffer.
    4. Runs the main PPO training loop.
    5. Calls periodic evaluation via robot_safe_ppo.eval.
    6. Logs all metrics to <save_dir>/metrics.csv.
    7. Saves checkpoints to <save_dir>/checkpoints/.
    8. Saves a copy of the resolved config to <save_dir>/config.yaml.

Status: STUB — training loop not yet implemented.
"""

from __future__ import annotations

import argparse
import pathlib
import sys


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the PPO training run."""
    parser = argparse.ArgumentParser(
        description="Train a baseline PPO agent on a MuJoCo environment."
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
        help="Total number of environment steps to train for (default: 1e6).",
    )
    parser.add_argument(
        "--save_dir",
        type=pathlib.Path,
        default=pathlib.Path("runs/ppo_default"),
        help="Directory for checkpoints, metrics, and config (default: runs/ppo_default).",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=10_000,
        help="Run evaluation every N environment steps (default: 10 000).",
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=pathlib.Path("configs/ppo.yaml"),
        help="Path to the base YAML hyperparameter config (default: configs/ppo.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    """Main training entry point — scaffold only."""
    args = parse_args()

    # Placeholder: print resolved args and exit cleanly so smoke tests pass.
    print(f"[train_ppo] env_id={args.env_id}  seed={args.seed}  "
          f"total_timesteps={args.total_timesteps}  save_dir={args.save_dir}  "
          f"eval_every={args.eval_every}")
    print("[train_ppo] Training loop not yet implemented (scaffold phase).")
    sys.exit(0)


if __name__ == "__main__":
    main()
