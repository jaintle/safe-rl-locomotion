"""
test_smoke.py
=============
Smoke test layer (fast, always run in CI).

Verifies that:
    1. The Gymnasium + MuJoCo environment can be created and stepped (via
       the existing smoke_env.py script).
    2. After a short training run (5 000 steps), metrics.csv exists in the
       designated save directory.
    3. At least one checkpoint file has been saved.

These tests are intentionally lightweight — they exercise the full
training pipeline but do not check learning quality.

Status:
    - test_smoke_env: ACTIVE (uses existing scripts/smoke_env.py).
    - test_ppo_smoke_train: STUB (training loop not yet implemented).
    - test_cppo_smoke_train: STUB (training loop not yet implemented).
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

# Path to the repository root (tests/ is one level down from root).
REPO_ROOT = pathlib.Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# 1. Environment smoke test
# ---------------------------------------------------------------------------

def test_smoke_env() -> None:
    """
    Verify that the MuJoCo environment loads and runs 1 000 random steps.

    Runs scripts/smoke_env.py as a subprocess and checks for "SMOKE_OK" in
    stdout.  This ensures the gymnasium[mujoco] installation is healthy
    independently of any algorithm code.
    """
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "smoke_env.py")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"smoke_env.py exited with code {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "SMOKE_OK" in result.stdout, (
        f"Expected 'SMOKE_OK' in smoke_env.py output.\n"
        f"stdout: {result.stdout}"
    )


# ---------------------------------------------------------------------------
# 2. PPO smoke train (stub — will be enabled once training loop exists)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="PPO training loop not yet implemented (scaffold phase).")
def test_ppo_smoke_train(tmp_path: pathlib.Path) -> None:
    """
    Train PPO for 5 000 steps and verify artefacts are written.

    Checks:
        - <save_dir>/metrics.csv exists and is non-empty.
        - At least one file in <save_dir>/checkpoints/.
    """
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_ppo.py"),
            "--env_id", "Hopper-v4",
            "--seed", "0",
            "--total_timesteps", "5000",
            "--save_dir", str(tmp_path),
            "--eval_every", "2500",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"train_ppo.py failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    metrics_csv = tmp_path / "metrics.csv"
    assert metrics_csv.exists(), "metrics.csv not found after smoke train."
    assert metrics_csv.stat().st_size > 0, "metrics.csv is empty."

    checkpoints_dir = tmp_path / "checkpoints"
    ckpts = list(checkpoints_dir.glob("*.pt")) if checkpoints_dir.exists() else []
    assert len(ckpts) >= 1, "No checkpoint files found after smoke train."


# ---------------------------------------------------------------------------
# 3. C-PPO smoke train (stub)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="C-PPO training loop not yet implemented (scaffold phase).")
def test_cppo_smoke_train(tmp_path: pathlib.Path) -> None:
    """
    Train C-PPO for 5 000 steps and verify artefacts are written.

    Same checks as PPO plus verifies that the lambda column appears in
    metrics.csv.
    """
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_cppo.py"),
            "--env_id", "Hopper-v4",
            "--seed", "0",
            "--total_timesteps", "5000",
            "--save_dir", str(tmp_path),
            "--eval_every", "2500",
            "--cost_limit", "0.1",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"train_cppo.py failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    metrics_csv = tmp_path / "metrics.csv"
    assert metrics_csv.exists(), "metrics.csv not found after C-PPO smoke train."

    import csv
    with open(metrics_csv) as f:
        header = next(csv.reader(f))
    assert "lambda" in header, f"'lambda' column missing from metrics.csv. Header: {header}"
