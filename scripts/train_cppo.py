"""
train_cppo.py
=============
Entry point for training the safety-constrained PPO agent (C-PPO Lagrangian).

Usage example::

    python scripts/train_cppo.py \
        --env_id Hopper-v4 \
        --seed 0 \
        --total_timesteps 1000000 \
        --save_dir runs/cppo_hopper_s0 \
        --eval_every 10000 \
        --cost_limit 0.1

The script:
    1. Parses CLI arguments (override precedence over configs/cppo.yaml).
    2. Loads hyperparameters from configs/cppo.yaml.
    3. Constructs the environment, CPPOLagrangianAgent, and buffer (with cost).
    4. Runs the C-PPO training loop, applying the Lagrangian penalty each update.
    5. Evaluates periodically (both return and cost).
    6. Logs all metrics to <save_dir>/metrics.csv (includes lambda column).
    7. Saves checkpoints to <save_dir>/checkpoints/.
    8. Saves resolved config to <save_dir>/config.yaml.

Status: STUB — training loop not yet implemented.
"""

from __future__ import annotations

import argparse
import pathlib
import sys


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
        default=pathlib.Path("configs/cppo.yaml"),
        help="Path to base YAML config (default: configs/cppo.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    """Main training entry point — scaffold only."""
    args = parse_args()

    # Placeholder: print resolved args and exit cleanly.
    print(f"[train_cppo] env_id={args.env_id}  seed={args.seed}  "
          f"total_timesteps={args.total_timesteps}  save_dir={args.save_dir}  "
          f"eval_every={args.eval_every}  cost_limit={args.cost_limit}  "
          f"cost_fn={args.cost_fn}")
    print("[train_cppo] Training loop not yet implemented (scaffold phase).")
    sys.exit(0)


if __name__ == "__main__":
    main()
