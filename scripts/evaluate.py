"""
evaluate.py
===========
Standalone post-hoc evaluation script.

Loads a saved checkpoint and runs deterministic evaluation episodes to
produce final performance numbers suitable for reporting.

Usage example::

    python scripts/evaluate.py \
        --checkpoint runs/ppo_hopper_s0/checkpoints/step_1000000.pt \
        --env_id Hopper-v4 \
        --n_episodes 20 \
        --seed 9999

Output:
    Prints a summary table to stdout.
    Optionally writes a one-row eval_summary.csv to the checkpoint directory.

Status: STUB — not yet implemented.
"""

from __future__ import annotations

import argparse
import pathlib
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic evaluation of a saved PPO or C-PPO checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        required=True,
        help="Path to a .pt checkpoint file produced by train_ppo.py or train_cppo.py.",
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default="Hopper-v4",
        help="Gymnasium environment ID (must match the training env).",
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=20,
        help="Number of deterministic evaluation episodes (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=9999,
        help="Base seed for evaluation episodes (default: 9999).",
    )
    parser.add_argument(
        "--cost_fn",
        type=str,
        default=None,
        choices=[None, "action_magnitude", "torso_angle"],
        help="Cost function to evaluate (only for C-PPO checkpoints).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[evaluate] checkpoint={args.checkpoint}  env_id={args.env_id}  "
          f"n_episodes={args.n_episodes}  seed={args.seed}")
    print("[evaluate] Evaluation not yet implemented (scaffold phase).")
    sys.exit(0)


if __name__ == "__main__":
    main()
