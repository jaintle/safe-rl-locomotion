"""
make_plots.py
=============
Generate training-curve plots from saved metrics.csv files.

Reads one or more metrics.csv files produced during training and calls the
functions in robot_safe_ppo.plotting to produce PNG figures.

Usage examples::

    # Single PPO run
    python scripts/make_plots.py \
        --ppo_csv runs/ppo_hopper_s0/metrics.csv \
        --env_id Hopper-v4 --seed 0 --total_timesteps 1000000 \
        --out_dir reports/figures

    # Comparison: PPO vs C-PPO
    python scripts/make_plots.py \
        --ppo_csv runs/ppo_hopper_s0/metrics.csv \
        --cppo_csv runs/cppo_hopper_s0/metrics.csv \
        --env_id Hopper-v4 --seed 0 --total_timesteps 1000000 \
        --cost_limit 0.1 \
        --out_dir reports/figures

Status: STUB — not yet implemented.
"""

from __future__ import annotations

import argparse
import pathlib
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate training-curve plots from metrics CSV files."
    )
    parser.add_argument(
        "--ppo_csv",
        type=pathlib.Path,
        default=None,
        help="Path to metrics.csv from a baseline PPO run.",
    )
    parser.add_argument(
        "--cppo_csv",
        type=pathlib.Path,
        default=None,
        help="Path to metrics.csv from a C-PPO run.",
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default="Hopper-v4",
        help="Environment name (used in plot titles).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for the training run (used in plot annotations).",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=1_000_000,
        help="Total training budget (used in plot annotations).",
    )
    parser.add_argument(
        "--cost_limit",
        type=float,
        default=0.1,
        help="Safety threshold (drawn as reference line on cost plots).",
    )
    parser.add_argument(
        "--out_dir",
        type=pathlib.Path,
        default=pathlib.Path("reports/figures"),
        help="Directory to write PNG plots (default: reports/figures).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[make_plots] ppo_csv={args.ppo_csv}  cppo_csv={args.cppo_csv}  "
          f"env_id={args.env_id}  out_dir={args.out_dir}")
    print("[make_plots] Plotting not yet implemented (scaffold phase).")
    sys.exit(0)


if __name__ == "__main__":
    main()
