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
    1. Episode return vs environment steps (smoothed with rolling window).
    2. Episode cost vs environment steps (C-PPO only).
    3. Lambda (λ) vs environment steps (C-PPO only).
    4. Comparison overlay: PPO vs C-PPO return and cost on the same axes.

Design notes:
    - Uses matplotlib only; no seaborn dependency.
    - All plots saved to PNG at 150 dpi by default.
    - Functions accept a pandas DataFrame (loaded from metrics.csv) as input
      so they can be unit-tested independently of training.

Status: STUB — plotting logic not yet implemented.
"""

from __future__ import annotations

import pathlib
# import datetime
# import pandas as pd
# import matplotlib.pyplot as plt


def plot_returns(df, env_id: str, seed: int, total_timesteps: int,
                 out_path: pathlib.Path, smooth_window: int = 20) -> None:
    """
    Plot episode return vs environment steps.

    Args:
        df              : pandas DataFrame loaded from metrics.csv.
                          Expected columns: ``step``, ``episode_return``.
        env_id          : Environment name string for plot title.
        seed            : Training seed for plot annotation.
        total_timesteps : Total training budget for plot annotation.
        out_path        : File path to save the PNG.
        smooth_window   : Rolling-average window size (in rows).
    """
    raise NotImplementedError


def plot_costs(df, env_id: str, seed: int, total_timesteps: int,
               cost_limit: float, out_path: pathlib.Path,
               smooth_window: int = 20) -> None:
    """
    Plot episode cost vs environment steps (C-PPO only).

    Draws a horizontal dashed line at `cost_limit` for reference.

    Args:
        df              : pandas DataFrame from metrics.csv.
                          Expected columns: ``step``, ``episode_cost``.
        env_id          : Environment name string.
        seed            : Training seed.
        total_timesteps : Training budget.
        cost_limit      : Safety threshold d (drawn as reference line).
        out_path        : Output PNG path.
        smooth_window   : Rolling-average window size.
    """
    raise NotImplementedError


def plot_lambda(df, env_id: str, seed: int, total_timesteps: int,
                out_path: pathlib.Path) -> None:
    """
    Plot Lagrange multiplier λ vs environment steps (C-PPO only).

    Args:
        df              : pandas DataFrame from metrics.csv.
                          Expected column: ``step``, ``lambda``.
        env_id          : Environment name.
        seed            : Training seed.
        total_timesteps : Training budget.
        out_path        : Output PNG path.
    """
    raise NotImplementedError


def plot_comparison(ppo_df, cppo_df, env_id: str, seed: int,
                    total_timesteps: int, cost_limit: float,
                    out_dir: pathlib.Path) -> None:
    """
    Side-by-side comparison of PPO vs C-PPO.

    Produces two subplots: (1) episode return, (2) episode cost.

    Args:
        ppo_df          : Metrics DataFrame for baseline PPO run.
        cppo_df         : Metrics DataFrame for constrained PPO run.
        env_id          : Environment name.
        seed            : Seed used for both runs.
        total_timesteps : Training budget.
        cost_limit      : Safety threshold (drawn on cost subplot).
        out_dir         : Directory to save the comparison PNG.
    """
    raise NotImplementedError
