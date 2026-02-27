"""
test_determinism.py
===================
Determinism sanity-check test layer.

Verifies that two training runs with identical seeds produce evaluation
returns that are within a small absolute tolerance of each other.

This guards against non-determinism introduced by:
    - Uninitialised random states in new modules.
    - Non-deterministic CUDA ops (if GPU is used).
    - Floating-point ordering differences across OS/hardware.

Tolerance note:
    A tight tolerance (e.g. 1e-3) is appropriate for CPU-only runs with the
    same PyTorch version.  GPU runs may require a looser tolerance due to
    non-deterministic CUDA kernels.

Status:
    test_ppo_determinism: STUB (training loop not yet implemented).
    test_cppo_determinism: STUB (training loop not yet implemented).
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).parent.parent

# Absolute tolerance for eval return comparison across two identical seeds.
DETERMINISM_ATOL = 1e-3

# Short training budget for determinism checks (fast but long enough to
# exercise the full update pipeline).
DETERMINISM_TIMESTEPS = 2_000
DETERMINISM_EVAL_EVERY = 1_000


def _read_first_eval_return(metrics_csv: pathlib.Path) -> float:
    """
    Parse metrics.csv and return the first non-NaN eval_return_mean value.

    Args:
        metrics_csv: Path to the metrics CSV produced by a training run.

    Returns:
        First evaluation return as a float.

    Raises:
        ValueError: If no eval_return_mean column or value is found.
    """
    import csv
    with open(metrics_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get("eval_return_mean", "").strip()
            if val not in ("", "nan", "None"):
                return float(val)
    raise ValueError(f"No eval_return_mean found in {metrics_csv}")


@pytest.mark.skip(reason="PPO training loop not yet implemented (scaffold phase).")
def test_ppo_determinism(tmp_path: pathlib.Path) -> None:
    """
    Run PPO twice with seed=42; assert early eval returns are within tolerance.
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
            timeout=120,
        )
        assert result.returncode == 0, (
            f"Train run failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    r0 = _read_first_eval_return(save_dirs[0] / "metrics.csv")
    r1 = _read_first_eval_return(save_dirs[1] / "metrics.csv")

    assert abs(r0 - r1) <= DETERMINISM_ATOL, (
        f"PPO determinism check failed: run0={r0:.6f}, run1={r1:.6f}, "
        f"diff={abs(r0-r1):.6f} > tol={DETERMINISM_ATOL}"
    )


@pytest.mark.skip(reason="C-PPO training loop not yet implemented (scaffold phase).")
def test_cppo_determinism(tmp_path: pathlib.Path) -> None:
    """
    Run C-PPO twice with seed=42; assert early eval returns are within tolerance.
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
            timeout=120,
        )
        assert result.returncode == 0, (
            f"C-PPO train run failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    r0 = _read_first_eval_return(save_dirs[0] / "metrics.csv")
    r1 = _read_first_eval_return(save_dirs[1] / "metrics.csv")

    assert abs(r0 - r1) <= DETERMINISM_ATOL, (
        f"C-PPO determinism check failed: run0={r0:.6f}, run1={r1:.6f}, "
        f"diff={abs(r0-r1):.6f} > tol={DETERMINISM_ATOL}"
    )
