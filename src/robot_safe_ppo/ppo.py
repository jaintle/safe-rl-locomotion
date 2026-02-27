"""
ppo.py
======
Baseline Proximal Policy Optimisation (PPO) implementation.

Reference:
    Schulman et al., "Proximal Policy Optimization Algorithms", 2017.
    https://arxiv.org/abs/1707.06347

Design notes:
    - CleanRL-style: single-file, self-contained agent class.
    - Continuous action spaces only (MuJoCo Hopper-v4, Walker2d-v4).
    - Uses Generalised Advantage Estimation (GAE-lambda).
    - Advantage normalisation applied before each update epoch.
    - Actor and critic share a MLP torso (optional) or are separate networks.
    - All logging is delegated to utils.MetricLogger; this module is
      algorithm-only with no I/O side-effects.

Status: STUB — algorithm logic not yet implemented.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Imports (will be populated when algorithm logic is added)
# ---------------------------------------------------------------------------
# import torch
# import torch.nn as nn
# import numpy as np
# from robot_safe_ppo.buffers import RolloutBuffer
# from robot_safe_ppo.utils import make_mlp, set_seeds


class PPOAgent:
    """
    Placeholder for the PPO agent.

    Attributes (to be implemented):
        actor  : stochastic policy network (Gaussian head for continuous ctrl)
        critic : state-value network V(s)
        optimizer : Adam optimiser shared across actor + critic parameters
    """

    def __init__(self, obs_dim: int, act_dim: int, cfg: dict) -> None:
        """
        Initialise actor, critic, and optimiser.

        Args:
            obs_dim: Dimensionality of the observation space.
            act_dim: Dimensionality of the continuous action space.
            cfg:     Hyperparameter dictionary (loaded from configs/ppo.yaml).
        """
        raise NotImplementedError("PPOAgent.__init__ not yet implemented.")

    def select_action(self, obs):
        """
        Sample an action from the current policy.

        Args:
            obs: Numpy array of shape (obs_dim,) — a single observation.

        Returns:
            action      : numpy array of shape (act_dim,)
            log_prob    : scalar log-probability of the sampled action
            value       : scalar state-value estimate V(obs)
        """
        raise NotImplementedError

    def update(self, buffer) -> dict:
        """
        Perform one PPO update epoch over the collected rollout buffer.

        Args:
            buffer: RolloutBuffer containing transitions from the last rollout.

        Returns:
            Dictionary of scalar training metrics (losses, entropy, etc.)
        """
        raise NotImplementedError
