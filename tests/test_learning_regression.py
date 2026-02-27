"""
test_learning_regression.py
============================
Learning regression test layer (slow tests — not run in fast CI).

Verifies that both PPO and C-PPO agents actually learn: the final
evaluation return must exceed the initial evaluation return by a
minimum threshold, demonstrating that gradient updates are effective.

This test is marked ``@pytest.mark.slow`` and is excluded from the
default pytest run.  To run it explicitly::

    pytest -m slow -v tests/test_learning_regression.py

Design notes:
    - Uses PPO for the primary regression; C-PPO uses PPO as its backbone
      so a C-PPO regression is also meaningful.
    - Training budget: 50 000 steps (fast enough on CPU, ~5–15 min).
    - Threshold: +10 return points over the initial eval.  Hopper-v4 starts
      at roughly 15–50 (random policy); +10 is intentionally conservative
      to avoid flakiness due to seed variance while still catching gradient
      sign bugs or broken value function updates.
    - High variance across seeds is expected and acknowledged; a single-seed
      test is a necessary but not sufficient condition for correctness.
    - C-PPO regression also performs a soft constraint check: the mean
      episode cost over the last 20 % of eval rows must be ≤ 1.5 × cost_limit.
      This is lenient by design — the Lagrangian method can be slow to converge.

Status: ACTIVE (Phase 4 — both algorithms implemented).
"""

from __future__ import annotations

import csv
import pathlib
import subprocess
import sys
from typing import List

import pytest

REPO_ROOT = pathlib.Path(__file__).parent.parent

# Mark all tests in this file as slow.
pytestmark = pytest.mark.slow

# Training budget for regression tests.
# 50k steps takes ~5–15 min on CPU; enough to show early learning on Hopper-v4.
REGRESSION_TIMESTEPS = 50_000
REGRESSION_EVAL_EVERY = 10_000

# Minimum improvement threshold (conservative to avoid flakiness).
# Hopper-v4 random-policy return ≈ 15–50; even weak learning exceeds +10.
MIN_IMPROVEMENT_THRESHOLD = 10.0


# ---------------------------------------------------------------------------
# CSV parsing helpers
# ---------------------------------------------------------------------------

def _load_eval_returns(metrics_csv: pathlib.Path) -> List[float]:
    """
    Extract all eval_return_mean values (non-NaN rows) from metrics.csv.

    Returns:
        List of floats in chronological order.
    """
    returns: List[float] = []
    with open(metrics_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get("eval_return_mean", "").strip()
            if val not in ("", "nan", "None"):
                returns.append(float(val))
    return returns


def _load_eval_costs(metrics_csv: pathlib.Path) -> List[float]:
    """
    Extract all eval_cost_mean values (non-NaN rows) from metrics.csv.
    """
    costs: List[float] = []
    with open(metrics_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get("eval_cost_mean", "").strip()
            if val not in ("", "nan", "None"):
                costs.append(float(val))
    return costs


# ---------------------------------------------------------------------------
# PPO learning regression
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_ppo_learning_regression(tmp_path: pathlib.Path) -> None:
    """
    Train PPO for REGRESSION_TIMESTEPS steps; verify final > initial + threshold.

    Failure modes this catches:
        - Gradient sign errors (policy degrades instead of improving).
        - Advantage normalisation bugs (no learning signal).
        - Incorrect log-prob computation (ratio always ≈ 0 / clipping always triggers).
        - Broken value function (no credit assignment).
    """
    save_dir = tmp_path / "ppo_regression"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_ppo.py"),
            "--env_id", "Hopper-v4",
            "--seed", "0",
            "--total_timesteps", str(REGRESSION_TIMESTEPS),
            "--save_dir", str(save_dir),
            "--eval_every", str(REGRESSION_EVAL_EVERY),
        ],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert result.returncode == 0, (
        f"train_ppo.py failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )

    returns = _load_eval_returns(save_dir / "metrics.csv")
    assert len(returns) >= 2, (
        f"Too few eval data points ({len(returns)}) — "
        f"check eval_every={REGRESSION_EVAL_EVERY} vs "
        f"total_timesteps={REGRESSION_TIMESTEPS}."
    )

    initial_return = returns[0]
    final_return   = returns[-1]
    improvement    = final_return - initial_return

    assert improvement >= MIN_IMPROVEMENT_THRESHOLD, (
        f"PPO learning regression FAILED:\n"
        f"  initial eval return : {initial_return:.1f}\n"
        f"  final eval return   : {final_return:.1f}\n"
        f"  improvement         : {improvement:.1f}\n"
        f"  threshold           : {MIN_IMPROVEMENT_THRESHOLD}\n"
        f"All eval returns: {returns}"
    )


# ---------------------------------------------------------------------------
# C-PPO learning regression
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_cppo_learning_regression(tmp_path: pathlib.Path) -> None:
    """
    Train C-PPO for REGRESSION_TIMESTEPS steps; verify final > initial + threshold.

    Also checks that the constraint is approximately satisfied: the mean
    eval episode cost over the last 20 % of eval rows must be ≤ cost_limit × 1.5
    (lenient — the Lagrangian method is slow to converge on short budgets).

    The threshold is the same conservative delta as PPO (+10 return points).
    """
    save_dir   = tmp_path / "cppo_regression"
    cost_limit = 0.1

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_cppo.py"),
            "--env_id", "Hopper-v4",
            "--seed", "0",
            "--total_timesteps", str(REGRESSION_TIMESTEPS),
            "--save_dir", str(save_dir),
            "--eval_every", str(REGRESSION_EVAL_EVERY),
            "--cost_limit", str(cost_limit),
        ],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert result.returncode == 0, (
        f"train_cppo.py failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )

    returns = _load_eval_returns(save_dir / "metrics.csv")
    assert len(returns) >= 2, (
        f"Too few eval data points ({len(returns)}) — "
        f"check eval_every / total_timesteps settings."
    )

    initial_return = returns[0]
    final_return   = returns[-1]
    improvement    = final_return - initial_return

    assert improvement >= MIN_IMPROVEMENT_THRESHOLD, (
        f"C-PPO learning regression FAILED:\n"
        f"  initial eval return : {initial_return:.1f}\n"
        f"  final eval return   : {final_return:.1f}\n"
        f"  improvement         : {improvement:.1f}\n"
        f"  threshold           : {MIN_IMPROVEMENT_THRESHOLD}\n"
        f"All eval returns: {returns}"
    )

    # Soft constraint satisfaction check (lenient — training is stochastic).
    costs = _load_eval_costs(save_dir / "metrics.csv")
    if costs:
        tail_costs     = costs[len(costs) * 4 // 5:]  # last 20 % of evals
        mean_tail_cost = sum(tail_costs) / len(tail_costs)
        lenient_limit  = cost_limit * 1.5
        assert mean_tail_cost <= lenient_limit, (
            f"C-PPO constraint check FAILED:\n"
            f"  mean tail eval cost : {mean_tail_cost:.4f}\n"
            f"  lenient limit (1.5×) : {lenient_limit:.4f}\n"
            f"  cost_limit           : {cost_limit}\n"
            f"  All eval costs: {costs}"
        )
