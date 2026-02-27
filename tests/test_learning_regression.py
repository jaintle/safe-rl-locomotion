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
    - Uses a short-but-meaningful training budget (not 5k steps).
    - Threshold is intentionally conservative to avoid flakiness.
    - High variance across seeds is expected and acknowledged; a single-seed
      test is a necessary but not sufficient condition for correctness.
    - These tests will be expanded with multi-seed runs once the implementation
      is stable.

Status: STUB — training loop not yet implemented.
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

# Training budget for regression tests.  Long enough to show learning on
# Hopper-v4 without taking hours on CPU.
REGRESSION_TIMESTEPS = 200_000
REGRESSION_EVAL_EVERY = 20_000

# Minimum improvement: final eval return must exceed initial by this amount.
# Hopper-v4 random-policy return ≈ 15–40; a learning policy should reach
# several hundred by 200k steps.
MIN_IMPROVEMENT_THRESHOLD = 100.0


def _load_eval_returns(metrics_csv: pathlib.Path) -> List[float]:
    """
    Extract all eval_return_mean values (non-NaN rows) from metrics.csv.

    Returns:
        List of floats in chronological order.
    """
    returns = []
    with open(metrics_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get("eval_return_mean", "").strip()
            if val not in ("", "nan", "None"):
                returns.append(float(val))
    return returns


@pytest.mark.slow
@pytest.mark.skip(reason="PPO training loop not yet implemented (scaffold phase).")
def test_ppo_learning_regression(tmp_path: pathlib.Path) -> None:
    """
    Train PPO for REGRESSION_TIMESTEPS steps; verify final > initial + threshold.

    Failure modes this catches:
        - Gradient sign errors (policy degrades).
        - Advantage normalisation bugs (no learning signal).
        - Incorrect log-prob computation (clipping always triggers).
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
        f"Too few eval data points ({len(returns)}) — check eval_every setting."
    )

    initial_return = returns[0]
    final_return = returns[-1]
    improvement = final_return - initial_return

    assert improvement >= MIN_IMPROVEMENT_THRESHOLD, (
        f"PPO learning regression FAILED: initial={initial_return:.1f}, "
        f"final={final_return:.1f}, improvement={improvement:.1f} < "
        f"threshold={MIN_IMPROVEMENT_THRESHOLD}"
    )


@pytest.mark.slow
@pytest.mark.skip(reason="C-PPO training loop not yet implemented (scaffold phase).")
def test_cppo_learning_regression(tmp_path: pathlib.Path) -> None:
    """
    Train C-PPO for REGRESSION_TIMESTEPS steps; verify final > initial + threshold.

    Also checks that the constraint is approximately satisfied: the mean
    episode cost across the last 20% of training should be ≤ cost_limit * 1.5
    (a lenient check given training-time variance).
    """
    save_dir = tmp_path / "cppo_regression"
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
    assert len(returns) >= 2, "Too few eval data points."

    initial_return = returns[0]
    final_return = returns[-1]
    improvement = final_return - initial_return

    assert improvement >= MIN_IMPROVEMENT_THRESHOLD, (
        f"C-PPO learning regression FAILED: initial={initial_return:.1f}, "
        f"final={final_return:.1f}, improvement={improvement:.1f} < "
        f"threshold={MIN_IMPROVEMENT_THRESHOLD}"
    )

    # Soft constraint satisfaction check (lenient — training is stochastic).
    costs = []
    with open(save_dir / "metrics.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get("eval_cost_mean", "").strip()
            if val not in ("", "nan", "None"):
                costs.append(float(val))

    if costs:
        tail_costs = costs[len(costs) * 4 // 5:]  # last 20% of evals
        mean_tail_cost = sum(tail_costs) / len(tail_costs)
        assert mean_tail_cost <= cost_limit * 1.5, (
            f"C-PPO constraint violation: mean tail cost={mean_tail_cost:.4f} "
            f"> 1.5 * cost_limit={cost_limit * 1.5:.4f}"
        )
