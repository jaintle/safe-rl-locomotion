"""
plotting.py
===========
Plotting utilities for training curve visualisation.

Produces publication-style plots from the metrics.csv files written by
MetricLogger during training.

Required plot metadata (per CLAUDE.md):
    Every figure must include:
        - Title
        - Environment name
        - Seed information
        - Training budget (total timesteps)
        - Timestamp of plot generation

Plot types:
    1. returns_vs_steps.png   — train episode_return + eval_return_mean
    2. costs_vs_steps.png     — train episode_cost + eval_cost_mean (if present)
    3. lambda_vs_steps.png    — lambda vs step (C-PPO only, if column present)
    4. losses_vs_steps.png    — policy_loss, value_loss, approx_kl, clip_fraction

Design notes:
    - Uses matplotlib only; no seaborn dependency.
    - All plots saved to PNG at 150 dpi.
    - Functions accept a pandas DataFrame (loaded from metrics.csv) as input
      so they can be unit-tested independently of training.
    - Columns that are entirely NaN are silently skipped.
    - Rolling average uses a min_periods=1 window to handle short DataFrames.
"""

from __future__ import annotations

import datetime
import pathlib
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless environments
import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _subtitle(env_id: str, seed: int, total_timesteps: int) -> str:
    """Build a consistent subtitle string for all plots."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        f"env={env_id}  seed={seed}  budget={total_timesteps:,} steps  "
        f"generated={ts}"
    )


def _smooth(series: pd.Series, window: int) -> pd.Series:
    """Rolling mean with min_periods=1 so short DataFrames don't produce all-NaN."""
    return series.rolling(window=window, min_periods=1).mean()


def _finite(series: pd.Series) -> pd.Series:
    """Drop NaN / inf values from a series."""
    return series.replace([float("inf"), float("-inf")], pd.NA).dropna()


def _save(fig: plt.Figure, out_path: pathlib.Path) -> None:
    """Save figure, create parent dirs if necessary, then log the path."""
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plotting] saved → {out_path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_returns(
    df: pd.DataFrame,
    env_id: str,
    seed: int,
    total_timesteps: int,
    out_path: pathlib.Path,
    smooth_window: int = 20,
    label: str = "PPO",
) -> None:
    """
    Plot episode return vs environment steps.

    Draws two series when both are present:
        - episode_return  (per-episode train return, smoothed)
        - eval_return_mean  (periodic deterministic evaluation, with ±1 std band)

    Args:
        df              : pandas DataFrame loaded from metrics.csv.
        env_id          : Environment name for plot title.
        seed            : Training seed for annotation.
        total_timesteps : Total training budget for annotation.
        out_path        : File path to save the PNG.
        smooth_window   : Rolling-average window (applied to train returns).
        label           : Legend prefix (e.g. "PPO" or "C-PPO").
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Train episode returns (smoothed)
    if "episode_return" in df.columns:
        train_rows = df[["step", "episode_return"]].dropna(subset=["episode_return"])
        train_rows = train_rows[train_rows["episode_return"].apply(
            lambda x: str(x) not in ("nan", ""))]
        train_rows = train_rows.copy()
        train_rows["episode_return"] = pd.to_numeric(
            train_rows["episode_return"], errors="coerce")
        train_rows = train_rows.dropna(subset=["episode_return"])
        if not train_rows.empty:
            smoothed = _smooth(train_rows["episode_return"], smooth_window)
            ax.plot(
                train_rows["step"], smoothed,
                color="steelblue", alpha=0.85, linewidth=1.2,
                label=f"{label} train return (smooth={smooth_window})",
            )
            ax.plot(
                train_rows["step"], train_rows["episode_return"],
                color="steelblue", alpha=0.20, linewidth=0.6,
            )

    # Eval returns (mean ± std)
    if "eval_return_mean" in df.columns:
        eval_rows = df[["step", "eval_return_mean"]].copy()
        eval_rows["eval_return_mean"] = pd.to_numeric(
            eval_rows["eval_return_mean"], errors="coerce")
        eval_rows = eval_rows.dropna(subset=["eval_return_mean"])
        if not eval_rows.empty:
            ax.plot(
                eval_rows["step"], eval_rows["eval_return_mean"],
                color="darkorange", linewidth=2.0, marker="o", markersize=4,
                label=f"{label} eval return (mean)",
            )
            # Add std band if column present
            if "eval_return_std" in df.columns:
                std_col = pd.to_numeric(df["eval_return_std"], errors="coerce")
                eval_std = std_col.loc[eval_rows.index]
                if eval_std.notna().any():
                    ax.fill_between(
                        eval_rows["step"],
                        eval_rows["eval_return_mean"] - eval_std,
                        eval_rows["eval_return_mean"] + eval_std,
                        color="darkorange", alpha=0.15,
                    )

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Undiscounted Return")
    ax.set_title(
        f"Return vs Steps — {env_id}\n{_subtitle(env_id, seed, total_timesteps)}",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)


def plot_costs(
    df: pd.DataFrame,
    env_id: str,
    seed: int,
    total_timesteps: int,
    cost_limit: float,
    out_path: pathlib.Path,
    smooth_window: int = 20,
    label: str = "C-PPO",
) -> None:
    """
    Plot episode cost vs environment steps (C-PPO only).

    Draws a horizontal dashed line at `cost_limit` for reference.

    Args:
        df              : pandas DataFrame from metrics.csv.
        env_id          : Environment name.
        seed            : Training seed.
        total_timesteps : Training budget.
        cost_limit      : Safety threshold d (horizontal reference line).
        out_path        : Output PNG path.
        smooth_window   : Rolling-average window (applied to train costs).
        label           : Legend prefix.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Train episode costs (smoothed)
    if "episode_cost" in df.columns:
        cost_rows = df[["step", "episode_cost"]].copy()
        cost_rows["episode_cost"] = pd.to_numeric(
            cost_rows["episode_cost"], errors="coerce")
        cost_rows = cost_rows.dropna(subset=["episode_cost"])
        if not cost_rows.empty:
            smoothed = _smooth(cost_rows["episode_cost"], smooth_window)
            ax.plot(
                cost_rows["step"], smoothed,
                color="steelblue", alpha=0.85, linewidth=1.2,
                label=f"{label} train cost (smooth={smooth_window})",
            )
            ax.plot(
                cost_rows["step"], cost_rows["episode_cost"],
                color="steelblue", alpha=0.20, linewidth=0.6,
            )

    # Eval costs (mean ± std)
    if "eval_cost_mean" in df.columns:
        eval_rows = df[["step", "eval_cost_mean"]].copy()
        eval_rows["eval_cost_mean"] = pd.to_numeric(
            eval_rows["eval_cost_mean"], errors="coerce")
        eval_rows = eval_rows.dropna(subset=["eval_cost_mean"])
        if not eval_rows.empty:
            ax.plot(
                eval_rows["step"], eval_rows["eval_cost_mean"],
                color="crimson", linewidth=2.0, marker="o", markersize=4,
                label=f"{label} eval cost (mean)",
            )
            if "eval_cost_std" in df.columns:
                std_col = pd.to_numeric(df["eval_cost_std"], errors="coerce")
                eval_std = std_col.loc[eval_rows.index]
                if eval_std.notna().any():
                    ax.fill_between(
                        eval_rows["step"],
                        eval_rows["eval_cost_mean"] - eval_std,
                        eval_rows["eval_cost_mean"] + eval_std,
                        color="crimson", alpha=0.15,
                    )

    # Cost limit reference line
    ax.axhline(
        cost_limit, color="black", linestyle="--", linewidth=1.2,
        label=f"cost_limit={cost_limit}",
    )

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Episodic Cost")
    ax.set_title(
        f"Cost vs Steps — {env_id}\n{_subtitle(env_id, seed, total_timesteps)}",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)


def plot_lambda(
    df: pd.DataFrame,
    env_id: str,
    seed: int,
    total_timesteps: int,
    out_path: pathlib.Path,
) -> None:
    """
    Plot Lagrange multiplier λ vs environment steps (C-PPO only).

    Skips gracefully if no non-NaN lambda values are present.

    Args:
        df              : pandas DataFrame from metrics.csv.
        env_id          : Environment name.
        seed            : Training seed.
        total_timesteps : Training budget.
        out_path        : Output PNG path.
    """
    if "lambda" not in df.columns:
        print("[plotting] plot_lambda: 'lambda' column not found — skipping.")
        return

    lam_rows = df[["step", "lambda"]].copy()
    lam_rows["lambda"] = pd.to_numeric(lam_rows["lambda"], errors="coerce")
    lam_rows = lam_rows.dropna(subset=["lambda"])

    if lam_rows.empty:
        print("[plotting] plot_lambda: all lambda values are NaN — skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        lam_rows["step"], lam_rows["lambda"],
        color="purple", linewidth=1.8, marker="o", markersize=4,
        label="λ (Lagrange multiplier)",
    )
    ax.axhline(0.0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("λ (Lagrange multiplier)")
    ax.set_title(
        f"Lambda vs Steps — {env_id}\n{_subtitle(env_id, seed, total_timesteps)}",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)


def plot_losses(
    df: pd.DataFrame,
    env_id: str,
    seed: int,
    total_timesteps: int,
    out_path: pathlib.Path,
    label: str = "PPO",
) -> None:
    """
    Plot training loss components vs environment steps.

    Draws up to four sub-series (each only if column is present and non-empty):
        - policy_loss
        - value_loss
        - approx_kl
        - clip_fraction

    Args:
        df              : pandas DataFrame from metrics.csv.
        env_id          : Environment name.
        seed            : Training seed.
        total_timesteps : Training budget.
        out_path        : Output PNG path.
        label           : Legend prefix.
    """
    series_cfg = [
        ("policy_loss",   "Policy loss",      "steelblue",  "solid"),
        ("value_loss",    "Value loss",        "darkorange", "solid"),
        ("approx_kl",     "Approx KL",         "seagreen",   "dashed"),
        ("clip_fraction", "Clip fraction",     "crimson",    "dotted"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted_any = False

    for col, col_label, color, ls in series_cfg:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        valid = df[["step"]].copy()
        valid["y"] = series
        valid = valid.dropna(subset=["y"])
        if valid.empty:
            continue
        ax.plot(
            valid["step"], valid["y"],
            color=color, linestyle=ls, linewidth=1.4, alpha=0.85,
            label=f"{label} {col_label}",
        )
        plotted_any = True

    if not plotted_any:
        print("[plotting] plot_losses: no loss columns found — skipping.")
        plt.close(fig)
        return

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Value")
    ax.set_title(
        f"Training Losses vs Steps — {env_id}\n{_subtitle(env_id, seed, total_timesteps)}",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)


def plot_comparison(
    ppo_df: pd.DataFrame,
    cppo_df: pd.DataFrame,
    env_id: str,
    seed: int,
    total_timesteps: int,
    cost_limit: float,
    out_dir: pathlib.Path,
) -> None:
    """
    Side-by-side comparison of PPO vs C-PPO eval metrics.

    Produces one figure with two subplots:
        (1) eval_return_mean for PPO and C-PPO
        (2) eval_cost_mean for C-PPO (PPO has no cost column)

    Saved as ``<out_dir>/comparison.png``.

    Args:
        ppo_df          : Metrics DataFrame for baseline PPO run.
        cppo_df         : Metrics DataFrame for constrained PPO run.
        env_id          : Environment name.
        seed            : Seed used for both runs.
        total_timesteps : Training budget.
        cost_limit      : Safety threshold (drawn on cost subplot).
        out_dir         : Directory to save the comparison PNG.
    """
    out_dir = pathlib.Path(out_dir)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    subtitle = _subtitle(env_id, seed, total_timesteps)

    # ---- subplot 1: eval return ----
    ax = axes[0]
    for run_df, run_label, color in [
        (ppo_df, "PPO", "steelblue"),
        (cppo_df, "C-PPO", "darkorange"),
    ]:
        if run_df is None or "eval_return_mean" not in run_df.columns:
            continue
        er = run_df[["step", "eval_return_mean"]].copy()
        er["eval_return_mean"] = pd.to_numeric(er["eval_return_mean"], errors="coerce")
        er = er.dropna(subset=["eval_return_mean"])
        if er.empty:
            continue
        ax.plot(er["step"], er["eval_return_mean"],
                color=color, linewidth=2.0, marker="o", markersize=4,
                label=run_label)
        if "eval_return_std" in run_df.columns:
            std_col = pd.to_numeric(run_df["eval_return_std"], errors="coerce")
            std_vals = std_col.loc[er.index]
            if std_vals.notna().any():
                ax.fill_between(er["step"],
                                er["eval_return_mean"] - std_vals,
                                er["eval_return_mean"] + std_vals,
                                color=color, alpha=0.12)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Eval Return (mean)")
    ax.set_title(f"Return — {env_id}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- subplot 2: eval cost (C-PPO only) ----
    ax = axes[1]
    if cppo_df is not None and "eval_cost_mean" in cppo_df.columns:
        ec = cppo_df[["step", "eval_cost_mean"]].copy()
        ec["eval_cost_mean"] = pd.to_numeric(ec["eval_cost_mean"], errors="coerce")
        ec = ec.dropna(subset=["eval_cost_mean"])
        if not ec.empty:
            ax.plot(ec["step"], ec["eval_cost_mean"],
                    color="crimson", linewidth=2.0, marker="o", markersize=4,
                    label="C-PPO eval cost (mean)")
            if "eval_cost_std" in cppo_df.columns:
                std_col = pd.to_numeric(cppo_df["eval_cost_std"], errors="coerce")
                std_vals = std_col.loc[ec.index]
                if std_vals.notna().any():
                    ax.fill_between(ec["step"],
                                    ec["eval_cost_mean"] - std_vals,
                                    ec["eval_cost_mean"] + std_vals,
                                    color="crimson", alpha=0.12)
    ax.axhline(cost_limit, color="black", linestyle="--", linewidth=1.2,
               label=f"cost_limit={cost_limit}")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Eval Cost (mean)")
    ax.set_title(f"Cost — {env_id}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"PPO vs C-PPO Comparison\n{subtitle}", fontsize=9, y=1.01)
    fig.tight_layout()
    _save(fig, out_dir / "comparison.png")
