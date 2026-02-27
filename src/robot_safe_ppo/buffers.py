"""
buffers.py
==========
Rollout buffer used by both PPO and C-PPO training loops.

Responsibilities:
    - Store (obs, action, reward, done, log_prob, value) tuples collected
      during environment interaction.
    - For C-PPO: additionally store per-step cost signals.
    - Compute Generalised Advantage Estimates (GAE-λ) and returns on demand.
    - Provide minibatch iterators for the PPO update epochs.

Design decisions:
    - Pre-allocated numpy arrays (no dynamic appending) for speed.
    - GAE computed in-place after rollout collection is complete.
    - Buffer is reset at the start of each new rollout.

Status: STUB — data structures and GAE logic not yet implemented.
"""

from __future__ import annotations

# import numpy as np


class RolloutBuffer:
    """
    Fixed-size rollout buffer for on-policy PPO training.

    Args:
        buffer_size : Number of environment steps to collect per rollout.
        obs_dim     : Dimensionality of the observation space.
        act_dim     : Dimensionality of the action space.
        gamma       : Discount factor γ for return computation.
        gae_lambda  : GAE λ parameter (0 = TD, 1 = MC).
        store_costs : If True, allocate an additional cost buffer (for C-PPO).
    """

    def __init__(self, buffer_size: int, obs_dim: int, act_dim: int,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 store_costs: bool = False) -> None:
        raise NotImplementedError("RolloutBuffer.__init__ not yet implemented.")

    def reset(self) -> None:
        """Zero out all buffers and reset the write pointer."""
        raise NotImplementedError

    def add(self, obs, action, reward, done, log_prob, value,
            cost: float = 0.0) -> None:
        """
        Store one transition.

        Args:
            obs      : Observation array of shape (obs_dim,).
            action   : Action array of shape (act_dim,).
            reward   : Scalar reward.
            done     : Boolean episode-termination flag.
            log_prob : Log-probability of `action` under the current policy.
            value    : V(obs) estimate from the critic.
            cost     : Per-step cost (C-PPO only; ignored if store_costs=False).
        """
        raise NotImplementedError

    def compute_advantages(self, last_value: float) -> None:
        """
        Compute GAE advantages and discounted returns in-place.

        Must be called once after the rollout is complete and before
        iterating over minibatches.

        Args:
            last_value: Critic's value estimate for the state *after* the
                        last stored transition (bootstrapping for non-terminal).
        """
        raise NotImplementedError

    def get_minibatches(self, batch_size: int):
        """
        Yield shuffled minibatches for PPO update epochs.

        Yields:
            Dict with keys: obs, actions, log_probs_old, advantages, returns,
                            (and costs if store_costs=True).
        """
        raise NotImplementedError
