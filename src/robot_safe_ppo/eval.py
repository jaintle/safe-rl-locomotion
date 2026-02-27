"""
eval.py
=======
Deterministic policy evaluation utilities.

Responsibilities:
    - Run a fixed number of evaluation episodes with the *mean* action
      (no sampling noise) to obtain unbiased performance estimates.
    - For C-PPO: additionally accumulate per-episode cost totals.
    - Return summary statistics (mean, std, min, max) over evaluation episodes.

Separation of concerns:
    - This module contains no training logic.
    - It is called by both training scripts (periodic evaluation) and by the
      standalone `scripts/evaluate.py` script for post-hoc analysis.

Determinism contract:
    Evaluation episodes are seeded deterministically starting from a fixed
    eval_seed so that results are comparable across runs and checkpoints.

Status: STUB — evaluation logic not yet implemented.
"""

from __future__ import annotations

# import numpy as np


def evaluate_policy(agent, env_id: str, n_episodes: int = 10,
                    eval_seed: int = 1000,
                    compute_cost: bool = False,
                    cost_fn=None) -> dict:
    """
    Evaluate a trained policy over several deterministic episodes.

    Uses the *mean* action (mu) from the policy distribution rather than
    sampling, ensuring evaluation is noise-free.

    Args:
        agent         : Trained agent with a `select_action(obs, deterministic=True)`
                        method.
        env_id        : Gymnasium environment ID string (e.g. ``"Hopper-v4"``).
        n_episodes    : Number of full episodes to run.
        eval_seed     : Base random seed; episode i uses seed `eval_seed + i`.
        compute_cost  : If True, accumulate per-step cost using `cost_fn`.
        cost_fn       : Callable ``(obs, action, next_obs) -> float``; required
                        when `compute_cost=True`.

    Returns:
        Dictionary with keys:
            - ``eval_return_mean`` : Mean undiscounted episode return.
            - ``eval_return_std``  : Standard deviation of episode returns.
            - ``eval_cost_mean``   : Mean episode cost (only if compute_cost).
            - ``eval_cost_std``    : Std of episode costs (only if compute_cost).
    """
    raise NotImplementedError("evaluate_policy not yet implemented.")
