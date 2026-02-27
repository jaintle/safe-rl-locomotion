"""
cppo_lagrangian.py
==================
Safety-Constrained PPO via Lagrangian relaxation (C-PPO).

Reference inspiration:
    Achiam et al., "Constrained Policy Optimisation (CPO)", 2017.
    https://arxiv.org/abs/1705.10528

    The implementation here uses a simpler Lagrangian dual ascent update
    rather than the trust-region projection of CPO proper.  This is standard
    practice in the safe-RL empirical literature and is computationally
    tractable without second-order methods.

Constrained objective:
    maximise  E[reward] − λ · (E[cost] − d)
    subject to  λ ≥ 0

Dual (lambda) update rule:
    λ ← max(0,  λ + α_λ · (avg_episode_cost − cost_limit))

Cost function contract:
    The cost function receives (obs, action, next_obs) as numpy arrays and
    returns a non-negative scalar cost for that transition.  It must be
    computable from publicly-observable state; MuJoCo internals are not used.

Example cost definitions (placeholder, not yet wired up):
    1. Mean absolute action exceeds threshold τ:
           cost = float(np.mean(np.abs(action)) > tau)
    2. Torso height (obs[0] for Hopper) below safe threshold:
           cost = float(obs[0] < height_threshold)

Status: STUB — algorithm logic not yet implemented.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Cost functions
# ---------------------------------------------------------------------------

def cost_action_magnitude(obs: np.ndarray, action: np.ndarray,
                           next_obs: np.ndarray, threshold: float = 0.8) -> float:
    """
    Placeholder cost: penalise high-magnitude actions.

    A transition is considered unsafe when the mean absolute value of the
    action vector exceeds `threshold`.

    Args:
        obs       : Current observation (unused in this cost definition).
        action    : Action taken at this step.
        next_obs  : Next observation (unused in this cost definition).
        threshold : Safety threshold on mean |action|.

    Returns:
        1.0 if unsafe, 0.0 otherwise.
    """
    # TODO: wire into C-PPO rollout collection
    return float(np.mean(np.abs(action)) > threshold)


def cost_torso_angle(obs: np.ndarray, action: np.ndarray,
                     next_obs: np.ndarray, threshold: float = 0.2) -> float:
    """
    Placeholder cost: penalise large torso angles (Hopper-v4).

    For Hopper-v4, obs[1] is the torso angle (in radians).  The robot is
    considered unsafe when |torso_angle| exceeds `threshold`.

    Args:
        obs       : Current observation; obs[1] is torso angle for Hopper.
        action    : Action taken (unused).
        next_obs  : Next observation (unused).
        threshold : Angular deviation threshold (radians).

    Returns:
        1.0 if unsafe, 0.0 otherwise.
    """
    # TODO: verify obs index against Hopper-v4 observation spec
    torso_angle = obs[1] if len(obs) > 1 else 0.0
    return float(abs(torso_angle) > threshold)


# ---------------------------------------------------------------------------
# Lambda (dual variable) manager
# ---------------------------------------------------------------------------

class LagrangianMultiplier:
    """
    Manages the Lagrange multiplier λ for the safety constraint.

    λ is updated via dual ascent after each rollout batch.

    Status: STUB — update logic placeholder only.
    """

    def __init__(self, init_lambda: float = 0.0, lr_lambda: float = 1e-2,
                 cost_limit: float = 0.1) -> None:
        """
        Args:
            init_lambda : Initial value of λ (must be ≥ 0).
            lr_lambda   : Step size for the dual ascent update.
            cost_limit  : Safety threshold d; constraint is E[cost] ≤ d.
        """
        self.lam = max(0.0, init_lambda)
        self.lr = lr_lambda
        self.cost_limit = cost_limit

    def update(self, avg_episode_cost: float) -> float:
        """
        Apply one dual-ascent step given the observed average episode cost.

        λ ← max(0,  λ + α_λ · (avg_cost − d))

        Args:
            avg_episode_cost: Mean cost per episode over the last rollout batch.

        Returns:
            Updated value of λ.
        """
        # TODO: implement update and call from training loop
        self.lam = max(0.0, self.lam + self.lr * (avg_episode_cost - self.cost_limit))
        return self.lam

    @property
    def value(self) -> float:
        """Current λ value."""
        return self.lam


# ---------------------------------------------------------------------------
# C-PPO Agent (stub)
# ---------------------------------------------------------------------------

class CPPOLagrangianAgent:
    """
    Constrained PPO agent with Lagrangian dual update.

    Extends PPOAgent by:
    - maintaining a LagrangianMultiplier instance
    - logging per-step costs and per-episode cost sums
    - adjusting the policy loss with the penalty term −λ·cost

    Status: STUB — not yet implemented.
    """

    def __init__(self, obs_dim: int, act_dim: int, cfg: dict) -> None:
        """
        Args:
            obs_dim: Observation dimensionality.
            act_dim: Action dimensionality.
            cfg    : Hyperparameter dict (from configs/cppo.yaml).
        """
        raise NotImplementedError("CPPOLagrangianAgent.__init__ not yet implemented.")

    def select_action(self, obs):
        """Same interface as PPOAgent.select_action."""
        raise NotImplementedError

    def update(self, buffer, avg_episode_cost: float) -> dict:
        """
        Perform one C-PPO update.

        In addition to the standard PPO update, applies the Lagrangian
        penalty and updates λ.

        Args:
            buffer           : RolloutBuffer with transitions and per-step costs.
            avg_episode_cost : Mean cost per episode for the last batch.

        Returns:
            Dict of training metrics including lambda value.
        """
        raise NotImplementedError
