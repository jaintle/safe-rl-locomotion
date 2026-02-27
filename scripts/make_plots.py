"""
make_plots.py
=============
Generate training-curve plots from saved metrics.csv files.

Reads one or more metrics.csv files produced during training and calls the
functions in robot_safe_ppo.plotting to produce PNG figures.

Usage examples::

    # Single PPO run — supply --run_dir or --ppo_csv
    python scripts/make_plots.py \\
        --run_dir runs/ppo_hopper_s0 \\
        --env_id Hopper-v4 --seed 0 --total_timesteps 1000000 \\
        --out_dir reports/figures

    # Explicit CSV paths
    python scripts/make_plots.py \\
        --ppo_csv runs/ppo_hopper_s0/metrics.csv \\
        --env_id Hopper-v4 --seed 0 --total_timesteps 1000000 \\
        --out_dir reports/figures

    # Comparison: PPO vs C-PPO
    python scripts/make_plots.py \\
        --ppo_csv runs/ppo_hopper_s0/metrics.csv \\
        --cppo_csv runs/cppo_hopper_s0/metrics.csv \\
        --env_id Hopper-v4 --seed 0 --total_timesteps 1000000 \\
        --cost_limit 0.1 \\
        --out_dir reports/figures

Plots generated:
    returns_vs_steps.png   — episode_return + eval_return_mean
    costs_vs_steps.png     — episode_cost + eval_cost_mean (if cost data present)
    lambda_vs_steps.png    — lambda column (C-PPO only, if present)
    losses_vs_steps.png    — policy_loss, value_loss, approx_kl, clip_fraction
    comparison.png         — PPO vs C-PPO overlay (only when both CSVs provided)

Exits non-zero with a helpful message if no metrics.csv can be resolved.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# ---------------------------------------------------------------------------
# Make src/ importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from robot_safe_ppo.plotting import (
    plot_returns,
    plot_costs,
    plot_lambda,
    plot_losses,
    plot_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate training-curve plots from metrics CSV files."
    )
    # Convenience shorthand: point at the run directory
    parser.add_argument(
        "--run_dir",
        type=pathlib.Path,
        default=None,
        help="Run directory containing metrics.csv (shorthand for --ppo_csv).",
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
        "--smooth_window",
        type=int,
        default=20,
        help="Rolling-average window for smoothing train curves (default: 20).",
    )
    parser.add_argument(
        "--out_dir",
        type=pathlib.Path,
        default=pathlib.Path("reports/figures"),
        help="Directory to write PNG plots (default: reports/figures).",
    )
    return parser.parse_args()


def _load(csv_path: pathlib.Path) -> pd.DataFrame:
    """Load a metrics CSV, converting all columns to numeric where possible."""
    df = pd.read_csv(csv_path)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Resolve CSV paths
    # ------------------------------------------------------------------
    # --run_dir is a shorthand: look for metrics.csv inside it
    if args.run_dir is not None and args.ppo_csv is None:
        candidate = args.run_dir / "metrics.csv"
        if candidate.exists():
            args.ppo_csv = candidate
        else:
            print(
                f"[make_plots] ERROR: --run_dir={args.run_dir} does not contain "
                f"metrics.csv. Run the training script first.",
                file=sys.stderr,
            )
            sys.exit(1)

    # At least one CSV must be provided
    if args.ppo_csv is None and args.cppo_csv is None:
        print(
            "[make_plots] ERROR: Provide at least one of --ppo_csv, --cppo_csv, "
            "or --run_dir.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate provided paths exist
    for csv_path in (args.ppo_csv, args.cppo_csv):
        if csv_path is not None and not csv_path.exists():
            print(
                f"[make_plots] ERROR: metrics.csv not found: {csv_path}",
                file=sys.stderr,
            )
            sys.exit(1)

    # ------------------------------------------------------------------
    # Load DataFrames
    # ------------------------------------------------------------------
    ppo_df  = _load(args.ppo_csv)  if args.ppo_csv  else None
    cppo_df = _load(args.cppo_csv) if args.cppo_csv else None

    # Use whichever is available as the primary df for single-run plots
    primary_df    = ppo_df if ppo_df is not None else cppo_df
    primary_label = "PPO" if ppo_df is not None else "C-PPO"

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kw = dict(
        env_id=args.env_id,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
    )

    print(
        f"[make_plots] env={args.env_id}  seed={args.seed}  "
        f"total_timesteps={args.total_timesteps}  out_dir={out_dir}"
    )

    # ------------------------------------------------------------------
    # 1. Returns plot
    # ------------------------------------------------------------------
    plot_returns(
        primary_df,
        **kw,
        out_path=out_dir / "returns_vs_steps.png",
        smooth_window=args.smooth_window,
        label=primary_label,
    )

    # ------------------------------------------------------------------
    # 2. Costs plot (only if cost data present in the primary df)
    # ------------------------------------------------------------------
    cost_df    = cppo_df if cppo_df is not None else ppo_df
    cost_label = "C-PPO" if cppo_df is not None else "PPO"

    has_cost = cost_df is not None and (
        "episode_cost" in cost_df.columns or "eval_cost_mean" in cost_df.columns
    )
    if has_cost:
        plot_costs(
            cost_df,
            **kw,
            cost_limit=args.cost_limit,
            out_path=out_dir / "costs_vs_steps.png",
            smooth_window=args.smooth_window,
            label=cost_label,
        )
    else:
        print("[make_plots] No cost columns found — skipping costs_vs_steps.png.")

    # ------------------------------------------------------------------
    # 3. Lambda plot (C-PPO only)
    # ------------------------------------------------------------------
    if cppo_df is not None:
        plot_lambda(
            cppo_df,
            **kw,
            out_path=out_dir / "lambda_vs_steps.png",
        )

    # ------------------------------------------------------------------
    # 4. Losses plot
    # ------------------------------------------------------------------
    plot_losses(
        primary_df,
        **kw,
        out_path=out_dir / "losses_vs_steps.png",
        label=primary_label,
    )

    # ------------------------------------------------------------------
    # 5. Comparison (only when both CSVs provided)
    # ------------------------------------------------------------------
    if ppo_df is not None and cppo_df is not None:
        plot_comparison(
            ppo_df,
            cppo_df,
            **kw,
            cost_limit=args.cost_limit,
            out_dir=out_dir,
        )

    print(f"[make_plots] Done. Plots written to: {out_dir}")


if __name__ == "__main__":
    main()
