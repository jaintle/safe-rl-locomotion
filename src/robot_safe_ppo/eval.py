"""
eval.py
=======
Deterministic policy evaluation utilities.

Responsibilities:
    - Run a fixed number of evaluation episodes with the *mean* action
      (no sampling noise) to obtain unbiased performance estimates.
    - For C-PPO: additionally accumulate per-episode cost totals.
    - Return summary statistics (mean, std) over evaluation episodes.

Separation of concerns:
    - This module contains no training logic.
    - It is called by both training scripts (periodic evaluation) and by the
      standalone ``scripts/evaluate.py`` script for post-hoc analysis.

Determinism contract:
    Episode i uses seed ``eval_seed + i``, so results are identical across
    runs and checkpoints as long as the same seed range is used.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import gymnasium as gym


def evaluate_policy(
    agent,
    env_id: str,
    n_episodes: int = 10,
    eval_seed: int = 1000,
    compute_cost: bool = False,
    cost_fn: Optional[Callable] = None,
) -> dict:
    """
    Evaluate a trained policy over several deterministic episodes.

    Uses the *mean* action (``deterministic=True``) from the policy
    distribution — no sampling noise — so evaluation results are stable.

    Each evaluation episode is seeded independently:
    episode *i* uses ``env.reset(seed=eval_seed + i)``.

    Args:
        agent         : Agent with a ``select_action(obs, deterministic=True)``
                        method returning ``(action, log_prob, value)``.
        env_id        : Gymnasium environment ID, e.g. ``"Hopper-v4"``.
        n_episodes    : Number of full episodes to run.
        eval_seed     : Base seed; episode *i* uses ``eval_seed + i``.
        compute_cost  : If True, accumulate per-step cost using ``cost_fn``.
        cost_fn       : Callable ``(obs, action, next_obs) -> float``;
                        required when ``compute_cost=True``.

    Returns:
        Dictionary with keys:
            - ``eval_return_mean`` : Mean undiscounted episode return.
            - ``eval_return_std``  : Std-dev of episode returns.
            - ``eval_cost_mean``   : Mean episode cost (only if compute_cost).
            - ``eval_cost_std``    : Std-dev of episode costs (only if compute_cost).
    """
    if compute_cost and cost_fn is None:
        raise ValueError("cost_fn must be provided when compute_cost=True.")

    env = gym.make(env_id)
    episode_returns = []
    episode_costs = []

    for i in range(n_episodes):
        obs, _ = env.reset(seed=eval_seed + i)
        episode_return = 0.0
        episode_cost = 0.0
        done = False

        while not done:
            action, _, _ = agent.select_action(obs, deterministic=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += float(reward)
            if compute_cost:
                episode_cost += float(cost_fn(obs, action, next_obs))
            obs = next_obs

        episode_returns.append(episode_return)
        if compute_cost:
            episode_costs.append(episode_cost)

    env.close()

    result: dict = {
        "eval_return_mean": float(np.mean(episode_returns)),
        "eval_return_std": float(np.std(episode_returns)),
    }
    if compute_cost:
        result["eval_cost_mean"] = float(np.mean(episode_costs))
        result["eval_cost_std"] = float(np.std(episode_costs))

    return result
