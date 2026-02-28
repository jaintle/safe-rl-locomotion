"""
aggregate_results.py
====================
Aggregate multi-seed training results into summary figures and a markdown table.

Reads metrics.csv from a list of PPO and C-PPO run directories, extracts
eval rows, computes final-step metrics and mean±std across seeds, generates
publication-style aggregated plots, and prints a markdown table snippet.

Dependencies: pandas, matplotlib (same as the rest of the repo).

Usage example::

    python scripts/aggregate_results.py \\
        --ppo_dirs  runs/ppo_hopper_v4_t025_seed0_500k \\
                    runs/ppo_hopper_v4_t025_seed1_500k \\
                    runs/ppo_hopper_v4_t025_seed2_500k \\
        --cppo_dirs runs/cppo_hopper_v4_t025_limit80_seed0_500k \\
                    runs/cppo_hopper_v4_t025_limit80_seed1_500k \\
                    runs/cppo_hopper_v4_t025_limit80_seed2_500k \\
        --env_id Hopper-v4 \\
        --budget 500000 \\
        --cost_limit 80.0 \\
        --out_dir reports/figures/hopper_v4 \\
        --tag 500k

Outputs written to <out_dir>/:
    return_overlay_<tag>.png   — PPO vs C-PPO eval_return_mean (mean ± std band)
    cost_overlay_<tag>.png     — PPO vs C-PPO eval_cost_mean (mean ± std band)
    lambda_curves_<tag>.png    — C-PPO lambda per seed (+ mean)
    pareto_<tag>.png           — final return vs final cost scatter with error bars
    summary_<tag>.md           — markdown table snippet (paste into results report)
"""

from __future__ import annotations

import argparse
import csv
import datetime
import pathlib
import sys
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate multi-seed results into summary figures and tables."
    )
    p.add_argument(
        "--ppo_dirs", nargs="+", required=True,
        help="One or more PPO run directories (each contains metrics.csv).",
    )
    p.add_argument(
        "--cppo_dirs", nargs="+", required=True,
        help="One or more C-PPO run directories (each contains metrics.csv).",
    )
    p.add_argument("--env_id",     type=str,   default="Hopper-v4")
    p.add_argument("--budget",     type=int,   default=500_000,
                   help="Training budget (for plot annotations).")
    p.add_argument("--cost_limit", type=float, default=80.0,
                   help="Cost constraint limit (drawn as reference line).")
    p.add_argument(
        "--out_dir", type=pathlib.Path,
        default=pathlib.Path("reports/figures/hopper_v4"),
        help="Directory to write summary PNGs and markdown snippet.",
    )
    p.add_argument("--tag", type=str, default="",
                   help="Short tag appended to output filenames (e.g. '500k', '1m').")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_eval_df(run_dir: pathlib.Path) -> pd.DataFrame:
    """
    Load metrics.csv from run_dir, keep only rows where eval_return_mean
    is non-NaN, and coerce all columns to numeric.
    """
    csv_path = pathlib.Path(run_dir) / "metrics.csv"
    if not csv_path.exists():
        print(f"  [WARNING] metrics.csv not found in {run_dir} — skipping.",
              file=sys.stderr)
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    eval_df = df[df["eval_return_mean"].notna()].copy()
    if eval_df.empty:
        print(f"  [WARNING] No eval rows in {run_dir}/metrics.csv — skipping.",
              file=sys.stderr)
    return eval_df


def _load_all(dirs: List[str]) -> List[pd.DataFrame]:
    dfs = []
    for d in dirs:
        df = _load_eval_df(pathlib.Path(d))
        if not df.empty:
            dfs.append(df)
    return dfs


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _align_on_step(dfs: List[pd.DataFrame], col: str) -> pd.DataFrame:
    """
    Given a list of per-seed DataFrames, pivot on 'step' and return a
    DataFrame with one column per seed (NaN where a seed has no row for
    that step).
    """
    aligned = {}
    for i, df in enumerate(dfs):
        if col not in df.columns:
            continue
        aligned[f"seed_{i}"] = df.set_index("step")[col]
    if not aligned:
        return pd.DataFrame()
    return pd.DataFrame(aligned).sort_index()


def _final_metric(dfs: List[pd.DataFrame], col: str) -> np.ndarray:
    """Extract the last non-NaN value of `col` from each seed DataFrame."""
    vals = []
    for df in dfs:
        if col in df.columns:
            s = df[col].dropna()
            if not s.empty:
                vals.append(float(s.iloc[-1]))
    return np.array(vals)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

SEED_COLORS = ["#4C72B0", "#DD8452", "#55A868"]  # blue, orange, green

def _subtitle(env_id: str, budget: int, tag: str) -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"env={env_id}  budget={budget:,}  tag={tag}  generated={ts}"


def _save(fig: plt.Figure, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def _fname(out_dir: pathlib.Path, name: str, tag: str) -> pathlib.Path:
    suffix = f"_{tag}" if tag else ""
    return out_dir / f"{name}{suffix}.png"


# ---------------------------------------------------------------------------
# 1. Overlay: eval_return_mean or eval_cost_mean (mean ± std band)
# ---------------------------------------------------------------------------

def plot_overlay(
    ppo_dfs: List[pd.DataFrame],
    cppo_dfs: List[pd.DataFrame],
    col: str,
    ylabel: str,
    title: str,
    out_path: pathlib.Path,
    cost_limit: Optional[float] = None,
    subtitle: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for run_dfs, label, color in [
        (ppo_dfs,  "PPO",   "#4C72B0"),
        (cppo_dfs, "C-PPO", "#DD8452"),
    ]:
        aligned = _align_on_step(run_dfs, col)
        if aligned.empty:
            continue
        steps = aligned.index.values
        mean  = aligned.mean(axis=1).values
        std   = aligned.std(axis=1).values

        ax.plot(steps, mean, color=color, linewidth=2.0, label=f"{label} (mean)")
        ax.fill_between(steps, mean - std, mean + std,
                        color=color, alpha=0.18, label=f"{label} ±1 std")

        # Individual seed traces (thin, semi-transparent)
        for j, scol in enumerate(aligned.columns):
            ax.plot(steps, aligned[scol].values,
                    color=color, linewidth=0.6, alpha=0.35,
                    linestyle="--")

    if cost_limit is not None:
        ax.axhline(cost_limit, color="black", linestyle="--",
                   linewidth=1.2, label=f"cost_limit={cost_limit}")

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n{subtitle}", fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 2. Lambda curves (C-PPO only, per seed + mean)
# ---------------------------------------------------------------------------

def plot_lambda_curves(
    cppo_dfs: List[pd.DataFrame],
    out_path: pathlib.Path,
    cost_limit: float,
    subtitle: str = "",
) -> None:
    if not cppo_dfs or not any("lambda" in df.columns for df in cppo_dfs):
        print("  [INFO] No lambda column found — skipping lambda_curves plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    aligned = _align_on_step(cppo_dfs, "lambda")
    steps = aligned.index.values

    for j, scol in enumerate(aligned.columns):
        color = SEED_COLORS[j % len(SEED_COLORS)]
        ax.plot(steps, aligned[scol].values,
                color=color, linewidth=1.0, alpha=0.6,
                label=f"seed {j}")

    if len(aligned.columns) > 1:
        mean_lam = aligned.mean(axis=1).values
        ax.plot(steps, mean_lam,
                color="black", linewidth=2.2, label="mean λ", zorder=5)

    ax.axhline(0.0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("λ (Lagrange multiplier)")
    ax.set_title(f"C-PPO Lambda vs Steps\n{subtitle}", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 3. Pareto scatter: final return vs final cost (with error bars)
# ---------------------------------------------------------------------------

def plot_pareto(
    ppo_dfs: List[pd.DataFrame],
    cppo_dfs: List[pd.DataFrame],
    out_path: pathlib.Path,
    cost_limit: float,
    subtitle: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))

    for run_dfs, label, color, marker in [
        (ppo_dfs,  "PPO",   "#4C72B0", "o"),
        (cppo_dfs, "C-PPO", "#DD8452", "s"),
    ]:
        ret_vals  = _final_metric(run_dfs, "eval_return_mean")
        cost_vals = _final_metric(run_dfs, "eval_cost_mean")

        if len(ret_vals) == 0:
            continue

        mean_ret  = ret_vals.mean()
        mean_cost = cost_vals.mean() if len(cost_vals) else float("nan")
        std_ret   = ret_vals.std()   if len(ret_vals)  > 1 else 0.0
        std_cost  = cost_vals.std()  if len(cost_vals) > 1 else 0.0

        # Individual seed points
        for i, (r, c) in enumerate(zip(ret_vals,
                                       cost_vals if len(cost_vals) else [float("nan")] * len(ret_vals))):
            ax.scatter(c, r, color=color, marker=marker,
                       alpha=0.5, s=40, zorder=4)

        # Mean with error bars
        ax.errorbar(
            mean_cost, mean_ret,
            xerr=std_cost, yerr=std_ret,
            color=color, marker=marker, markersize=10,
            linewidth=1.5, capsize=4, label=label, zorder=5,
        )

    ax.axvline(cost_limit, color="black", linestyle="--",
               linewidth=1.2, label=f"cost_limit={cost_limit}")
    ax.set_xlabel("Final Eval Cost (mean)")
    ax.set_ylabel("Final Eval Return (mean)")
    ax.set_title(f"Return–Cost Tradeoff (Pareto)\n{subtitle}", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 4. Markdown table snippet
# ---------------------------------------------------------------------------

def write_markdown_table(
    ppo_dfs: List[pd.DataFrame],
    cppo_dfs: List[pd.DataFrame],
    budget: int,
    cost_limit: float,
    tag: str,
    out_path: pathlib.Path,
) -> None:
    rows = []
    for run_dfs, algo in [(ppo_dfs, "PPO"), (cppo_dfs, "C-PPO")]:
        ret_vals  = _final_metric(run_dfs, "eval_return_mean")
        cost_vals = _final_metric(run_dfs, "eval_cost_mean")
        n = len(ret_vals)

        if n == 0:
            rows.append({
                "Algorithm": algo,
                "Budget": f"{budget:,}",
                "Seeds": "—",
                "Return (mean±std)": "—",
                "Cost (mean±std)": "—",
                "Constraint met?": "—",
            })
            continue

        mean_ret  = ret_vals.mean()
        std_ret   = ret_vals.std() if n > 1 else 0.0
        mean_cost = cost_vals.mean() if len(cost_vals) else float("nan")
        std_cost  = cost_vals.std()  if len(cost_vals) > 1 else 0.0

        constraint_met = (
            f"{sum(c <= cost_limit for c in cost_vals)}/{n}"
            if len(cost_vals) else "—"
        )

        rows.append({
            "Algorithm": algo,
            "Budget": f"{budget:,}",
            "Seeds": str(n),
            "Return (mean±std)": f"{mean_ret:.1f} ± {std_ret:.1f}",
            "Cost (mean±std)": f"{mean_cost:.2f} ± {std_cost:.2f}" if len(cost_vals) else "—",
            "Constraint met?": f"{constraint_met} seeds ≤ {cost_limit}",
        })

    lines = [
        f"<!-- auto-generated by aggregate_results.py  tag={tag} -->",
        "",
        f"### {tag.upper()} Steps — Summary Table",
        "",
        "| Algorithm | Budget | Seeds | Return (mean±std) | Cost (mean±std) | Constraint met? |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['Algorithm']} | {row['Budget']} | {row['Seeds']} "
            f"| {row['Return (mean±std)']} | {row['Cost (mean±std)']} "
            f"| {row['Constraint met?']} |"
        )
    lines += ["", ""]
    md = "\n".join(lines)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"  saved → {out_path}")
    # Also print to stdout so it can be captured in CI logs
    print("\n" + md)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag      = args.tag
    subtitle = _subtitle(args.env_id, args.budget, tag)

    print(f"[aggregate_results] Loading PPO runs ({len(args.ppo_dirs)} dirs)…")
    ppo_dfs = _load_all(args.ppo_dirs)
    print(f"  Loaded {len(ppo_dfs)} valid PPO run(s).")

    print(f"[aggregate_results] Loading C-PPO runs ({len(args.cppo_dirs)} dirs)…")
    cppo_dfs = _load_all(args.cppo_dirs)
    print(f"  Loaded {len(cppo_dfs)} valid C-PPO run(s).")

    if not ppo_dfs and not cppo_dfs:
        print("[aggregate_results] ERROR: No valid run data found.", file=sys.stderr)
        sys.exit(1)

    print(f"[aggregate_results] Writing plots to {out_dir}/…")

    # 1. Return overlay
    plot_overlay(
        ppo_dfs, cppo_dfs,
        col="eval_return_mean",
        ylabel="Eval Return (mean)",
        title=f"Return vs Steps — {args.env_id}",
        out_path=_fname(out_dir, "return_overlay", tag),
        subtitle=subtitle,
    )

    # 2. Cost overlay
    plot_overlay(
        ppo_dfs, cppo_dfs,
        col="eval_cost_mean",
        ylabel="Eval Cost (mean)",
        title=f"Cost vs Steps — {args.env_id}",
        out_path=_fname(out_dir, "cost_overlay", tag),
        cost_limit=args.cost_limit,
        subtitle=subtitle,
    )

    # 3. Lambda curves (C-PPO only)
    plot_lambda_curves(
        cppo_dfs,
        out_path=_fname(out_dir, "lambda_curves", tag),
        cost_limit=args.cost_limit,
        subtitle=subtitle,
    )

    # 4. Pareto scatter
    plot_pareto(
        ppo_dfs, cppo_dfs,
        out_path=_fname(out_dir, "pareto", tag),
        cost_limit=args.cost_limit,
        subtitle=subtitle,
    )

    # 5. Markdown table
    write_markdown_table(
        ppo_dfs, cppo_dfs,
        budget=args.budget,
        cost_limit=args.cost_limit,
        tag=tag,
        out_path=out_dir / f"summary_{tag}.md",
    )

    print(f"\n[aggregate_results] Done. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
