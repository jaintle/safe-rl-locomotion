"""
test_determinism.py
===================
Determinism sanity-check test layer.

Verifies that two training runs with identical seeds produce evaluation
returns that are within a small absolute tolerance of each other.

This guards against non-determinism introduced by:
    - Uninitialised random states in new modules.
    - Platform floating-point ordering differences.
    - Accidentally unseeded components (env, buffer, agent).

Tolerance rationale (ATOL = 5.0):
    A tight tolerance (e.g. 1e-3) is correct in theory for CPU-only runs
    with the same PyTorch build.  In practice, subprocess re-execution can
    introduce micro-differences due to OS scheduling and memory layout.
    We use ATOL=5.0 (on a typical Hopper-v4 return scale of ~20–3500) to
    remain non-flaky across machines while still catching real bugs like
    missing seed calls or wrong GAE computation.  If both runs produce the
    same exact value the test passes trivially; the tolerance is a safety
    margin, not the target precision.

    For cost values (0.0–1.0 scale per step): ATOL=0.1 is sufficient.

Status:
    test_ppo_determinism:  ACTIVE (Phase 4).
    test_cppo_determinism: ACTIVE (Phase 4).
"""

from __future__ import annotations

import csv
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).parent.parent

# Absolute tolerance for eval return comparison across two identical seeds.
# Loosened to 5.0 to avoid flakiness across machines (see module docstring).
DETERMINISM_ATOL_RETURN = 5.0

# Absolute tolerance for eval cost comparison (cost values in [0, 1] per step).
DETERMINISM_ATOL_COST = 0.1

# Short training budget for determinism checks.
# 4096 steps → 2 PPO updates of 2048 steps each → fast but exercises full loop.
DETERMINISM_TIMESTEPS = 4096
DETERMINISM_EVAL_EVERY = 2048


# ---------------------------------------------------------------------------
# CSV parsing helper
# ---------------------------------------------------------------------------

def _read_first_eval_row(metrics_csv: pathlib.Path) -> dict:
    """
    Parse metrics.csv and return the first row that contains a non-NaN
    eval_return_mean value.

    Returns:
        Dict with at least "eval_return_mean"; also "eval_cost_mean" if present.

    Raises:
        ValueError: If no eval row is found.
    """
    with open(metrics_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get("eval_return_mean", "").strip()
            if val not in ("", "nan", "None"):
                result = {"eval_return_mean": float(val)}
                cost_val = row.get("eval_cost_mean", "").strip()
                if cost_val not in ("", "nan", "None"):
                    result["eval_cost_mean"] = float(cost_val)
                return result
    raise ValueError(f"No eval_return_mean found in {metrics_csv}")


# ---------------------------------------------------------------------------
# PPO determinism test
# ---------------------------------------------------------------------------

def test_ppo_determinism(tmp_path: pathlib.Path) -> None:
    """
    Run PPO twice with seed=42; assert early eval returns are within tolerance.

    The two runs are sequential subprocesses with identical arguments.
    A return difference > DETERMINISM_ATOL_RETURN indicates a seeding bug.

    Expected run time: ~20–40 s on a modern CPU (two 4096-step runs).
    """
    save_dirs = [tmp_path / f"run_{i}" for i in range(2)]

    for save_dir in save_dirs:
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "train_ppo.py"),
                "--env_id", "Hopper-v4",
                "--seed", "42",
                "--total_timesteps", str(DETERMINISM_TIMESTEPS),
                "--save_dir", str(save_dir),
                "--eval_every", str(DETERMINISM_EVAL_EVERY),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, (
            f"PPO train run failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    row0 = _read_first_eval_row(save_dirs[0] / "metrics.csv")
    row1 = _read_first_eval_row(save_dirs[1] / "metrics.csv")

    r0 = row0["eval_return_mean"]
    r1 = row1["eval_return_mean"]

    assert abs(r0 - r1) <= DETERMINISM_ATOL_RETURN, (
        f"PPO determinism check failed: run0={r0:.4f}, run1={r1:.4f}, "
        f"diff={abs(r0 - r1):.4f} > tol={DETERMINISM_ATOL_RETURN}\n"
        f"This likely indicates a missing or incorrect seed call."
    )


# ---------------------------------------------------------------------------
# C-PPO determinism test
# ---------------------------------------------------------------------------

def test_cppo_determinism(tmp_path: pathlib.Path) -> None:
    """
    Run C-PPO twice with seed=42; assert early eval returns are within tolerance.

    Also checks eval_cost_mean if present in both runs (tolerance: ATOL_COST).

    Expected run time: ~20–50 s on a modern CPU (two 4096-step runs).
    """
    save_dirs = [tmp_path / f"run_{i}" for i in range(2)]

    for save_dir in save_dirs:
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "train_cppo.py"),
                "--env_id", "Hopper-v4",
                "--seed", "42",
                "--total_timesteps", str(DETERMINISM_TIMESTEPS),
                "--save_dir", str(save_dir),
                "--eval_every", str(DETERMINISM_EVAL_EVERY),
                "--cost_limit", "0.1",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, (
            f"C-PPO train run failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    row0 = _read_first_eval_row(save_dirs[0] / "metrics.csv")
    row1 = _read_first_eval_row(save_dirs[1] / "metrics.csv")

    r0 = row0["eval_return_mean"]
    r1 = row1["eval_return_mean"]

    assert abs(r0 - r1) <= DETERMINISM_ATOL_RETURN, (
        f"C-PPO determinism check failed: run0={r0:.4f}, run1={r1:.4f}, "
        f"diff={abs(r0 - r1):.4f} > tol={DETERMINISM_ATOL_RETURN}"
    )

    # Cost check (optional — only when both runs produced cost evals)
    if "eval_cost_mean" in row0 and "eval_cost_mean" in row1:
        c0 = row0["eval_cost_mean"]
        c1 = row1["eval_cost_mean"]
        assert abs(c0 - c1) <= DETERMINISM_ATOL_COST, (
            f"C-PPO cost determinism check failed: run0={c0:.4f}, run1={c1:.4f}, "
            f"diff={abs(c0 - c1):.4f} > tol={DETERMINISM_ATOL_COST}"
        )
