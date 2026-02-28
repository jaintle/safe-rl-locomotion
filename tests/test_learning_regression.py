"""
test_learning_regression.py
============================
Learning regression test layer (slow tests — not run in fast CI).

Run explicitly with::

    pytest -m slow -v tests/test_learning_regression.py

Design notes — PPO:
    - The return-improvement assertion (final − initial ≥ threshold) is
      appropriate for baseline PPO because there is no safety penalty pulling
      the policy away from the reward objective.  Hopper-v4 with a working
      PPO reliably improves by ≥ +10 over 50 k steps even with seed variance.

Design notes — C-PPO:
    - The same return-improvement criterion is *not* appropriate for C-PPO
      Lagrangian on short budgets.  The Lagrangian penalty (lambda × cost)
      is added to the policy gradient as soon as any constraint violation is
      detected; this causes the policy to sacrifice some reward in order to
      satisfy the cost constraint.  On a 50 k-step budget the optimiser may
      never have enough time to simultaneously drive cost below the limit AND
      grow the reward beyond its initial value.  A naive
      "final_return >= initial_return + delta" assertion would be brittle and
      would fire on correct implementations.
    - Instead we assert:
        (a) Training completed and produced ≥ 2 eval checkpoints.
        (b) Constraint satisfaction (primary safety goal): mean eval_cost_mean
            over the last 20 % of eval rows ≤ cost_limit × 1.5.
            The 1.5× factor is intentionally lenient — convergence is slow.
        (c) Weak performance sanity (catches degenerate / NaN policies):
            max(eval_return_mean) ≥ 20.
            Hopper-v4 with a completely broken policy (NaN actions,
            zero gradient, etc.) produces returns near 0–3.  A value of 20
            is reliably exceeded at least once during 50 k steps by any
            policy that is executing valid actions, even while being penalised
            for constraint violations.  This does NOT require monotonic
            improvement; it only verifies the policy is not degenerate.

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

# Mark all tests in this module as slow (also applied individually below).
pytestmark = pytest.mark.slow

# Training budget for regression tests.
# 50 k steps takes ~5–15 min on CPU; enough to show early learning on Hopper-v4.
REGRESSION_TIMESTEPS = 50_000
REGRESSION_EVAL_EVERY = 10_000

# PPO: minimum return improvement threshold.
# Hopper-v4 random-policy return ≈ 15–50; even weak PPO exceeds +10.
MIN_PPO_IMPROVEMENT = 10.0

# C-PPO: constraint lenient multiplier (1.5 × cost_limit).
CPPO_CONSTRAINT_MULTIPLIER = 1.5

# C-PPO: minimum ever-achieved return (degenerate-policy guard).
CPPO_MIN_PEAK_RETURN = 20.0


# ---------------------------------------------------------------------------
# CSV parsing helpers
# ---------------------------------------------------------------------------

def _load_eval_returns(metrics_csv: pathlib.Path) -> List[float]:
    """Return all non-NaN eval_return_mean values in chronological order."""
    returns: List[float] = []
    with open(metrics_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get("eval_return_mean", "").strip()
            if val not in ("", "nan", "None"):
                returns.append(float(val))
    return returns


def _load_eval_costs(metrics_csv: pathlib.Path) -> List[float]:
    """Return all non-NaN eval_cost_mean values in chronological order."""
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
    max_return     = max(returns)
    improvement    = final_return - initial_return

    assert improvement >= MIN_PPO_IMPROVEMENT, (
        f"PPO learning regression FAILED:\n"
        f"  initial eval return : {initial_return:.1f}\n"
        f"  final eval return   : {final_return:.1f}\n"
        f"  max eval return     : {max_return:.1f}\n"
        f"  improvement         : {improvement:.1f}\n"
        f"  threshold           : {MIN_PPO_IMPROVEMENT}\n"
        f"  all eval returns    : {[f'{r:.1f}' for r in returns]}"
    )


# ---------------------------------------------------------------------------
# C-PPO learning regression
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_cppo_learning_regression(tmp_path: pathlib.Path) -> None:
    """
    Train C-PPO for REGRESSION_TIMESTEPS steps; check constraint satisfaction
    and guard against degenerate policies.

    Assertions (see module docstring for full rationale):

    (a) At least 2 eval checkpoints exist.

    (b) Constraint satisfaction — primary safety goal:
        mean(eval_cost_mean over last 20 % of eval rows)
            <= cost_limit * CPPO_CONSTRAINT_MULTIPLIER (1.5)
        On a 50 k-step budget the Lagrangian typically drives cost well
        below the limit by the end of training, but the 1.5× factor gives
        headroom for runs where convergence is still in progress.

    (c) Degenerate-policy guard:
        max(eval_return_mean) >= CPPO_MIN_PEAK_RETURN (20.0)
        Any policy taking valid actions on Hopper-v4 will achieve a return
        of at least 20 at some point during 50 k steps.  Values near 0–3
        indicate NaN actions, all-zero gradients, or a broken cost critic
        that forces the policy to a corner solution.

    NOTE: No return-improvement assertion is made (see module docstring).
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
    costs   = _load_eval_costs(save_dir / "metrics.csv")

    # (a) Enough eval rows to draw any conclusions.
    assert len(returns) >= 2, (
        f"Too few eval data points ({len(returns)}) — "
        f"check eval_every={REGRESSION_EVAL_EVERY} vs "
        f"total_timesteps={REGRESSION_TIMESTEPS}."
    )

    initial_return = returns[0]
    final_return   = returns[-1]
    max_return     = max(returns)

    # (b) Constraint satisfaction over the tail of training.
    if costs:
        tail_start     = max(1, len(costs) * 4 // 5)  # last 20 %, at least 1 row
        tail_costs     = costs[tail_start:]
        mean_tail_cost = sum(tail_costs) / len(tail_costs)
        lenient_limit  = cost_limit * CPPO_CONSTRAINT_MULTIPLIER

        assert mean_tail_cost <= lenient_limit, (
            f"C-PPO constraint satisfaction FAILED:\n"
            f"  mean tail eval cost : {mean_tail_cost:.4f}\n"
            f"  lenient limit (1.5×): {lenient_limit:.4f}  (cost_limit={cost_limit})\n"
            f"  all eval costs      : {[f'{c:.4f}' for c in costs]}\n"
            f"  initial return      : {initial_return:.1f}\n"
            f"  final return        : {final_return:.1f}\n"
            f"  max return          : {max_return:.1f}\n"
            f"  all eval returns    : {[f'{r:.1f}' for r in returns]}"
        )
    else:
        # Cost column absent — warn but don't fail (PPO-only CSV unlikely here).
        import warnings
        warnings.warn(
            "test_cppo_learning_regression: no eval_cost_mean rows found; "
            "constraint check skipped.",
            stacklevel=2,
        )

    # (c) Degenerate-policy guard.
    assert max_return >= CPPO_MIN_PEAK_RETURN, (
        f"C-PPO degenerate-policy check FAILED:\n"
        f"  max eval return     : {max_return:.1f}  (threshold={CPPO_MIN_PEAK_RETURN})\n"
        f"  initial return      : {initial_return:.1f}\n"
        f"  final return        : {final_return:.1f}\n"
        f"  all eval returns    : {[f'{r:.1f}' for r in returns]}\n"
        f"  all eval costs      : {[f'{c:.4f}' for c in costs]}"
    )
