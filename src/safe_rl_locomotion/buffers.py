"""
buffers.py
==========
Rollout buffer used by both PPO and C-PPO training loops.

Responsibilities:
    - Store (obs, action, reward, done, log_prob, value) tuples collected
      during environment interaction.
    - For C-PPO: additionally store per-step costs, cost-critic values,
      cost advantages, and cost returns.
    - Compute Generalised Advantage Estimates (GAE-λ) for both the reward
      stream and (optionally) the cost stream.
    - Provide shuffled minibatch iterators for the PPO update epochs.

Design decisions:
    - Pre-allocated numpy arrays (no dynamic appending) for speed.
    - GAE is computed in-place after rollout collection is complete.
    - Buffer is reset at the start of each new rollout.
    - ``done[t] = True`` means the episode *terminated or was truncated* at
      step t, so ``V(s_{t+1})`` should not be bootstrapped.

GAE formula (Schulman et al., 2016) applied to both reward and cost streams:
    δ_t  = r_t + γ · V(s_{t+1}) · (1 − d_t) − V(s_t)
    Â_t  = δ_t + γλ · (1 − d_t) · Â_{t+1}
"""

from __future__ import annotations

from typing import Iterator

import numpy as np


class RolloutBuffer:
    """
    Fixed-size on-policy rollout buffer for PPO / C-PPO.

    When ``store_costs=True`` (C-PPO mode), the buffer also allocates:
        - ``costs``           : per-step cost signal c_t.
        - ``cost_values``     : cost-critic estimates V_C(s_t).
        - ``cost_advantages`` : GAE-λ over the cost stream.
        - ``cost_returns``    : cost_advantages + cost_values (critic targets).

    Both ``compute_advantages`` (reward stream) and
    ``compute_cost_advantages`` (cost stream) must be called before
    iterating over minibatches in C-PPO.

    Args:
        buffer_size : Number of environment steps per rollout.
        obs_dim     : Observation space dimensionality.
        act_dim     : Action space dimensionality.
        gamma       : Discount factor γ (shared for reward and cost streams).
        gae_lambda  : GAE λ (shared for reward and cost streams).
        store_costs : If True, allocate cost arrays (for C-PPO).
    """

    def __init__(self, buffer_size: int, obs_dim: int, act_dim: int,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 store_costs: bool = False) -> None:
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.store_costs = store_costs
        self._alloc()
        self.reset()

    # ------------------------------------------------------------------
    # Allocation helpers
    # ------------------------------------------------------------------

    def _alloc(self) -> None:
        """Allocate numpy arrays for all buffer fields."""
        n = self.buffer_size
        # Core fields (PPO + C-PPO)
        self.obs = np.zeros((n, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((n, self.act_dim), dtype=np.float32)
        self.rewards = np.zeros(n, dtype=np.float32)
        self.dones = np.zeros(n, dtype=np.float32)
        self.log_probs = np.zeros(n, dtype=np.float32)
        self.values = np.zeros(n, dtype=np.float32)
        self.advantages = np.zeros(n, dtype=np.float32)
        self.returns = np.zeros(n, dtype=np.float32)
        # Cost fields (C-PPO only)
        if self.store_costs:
            self.costs = np.zeros(n, dtype=np.float32)
            self.cost_values = np.zeros(n, dtype=np.float32)
            self.cost_advantages = np.zeros(n, dtype=np.float32)
            self.cost_returns = np.zeros(n, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Zero all arrays and reset the write pointer to 0."""
        self.obs[:] = 0.0
        self.actions[:] = 0.0
        self.rewards[:] = 0.0
        self.dones[:] = 0.0
        self.log_probs[:] = 0.0
        self.values[:] = 0.0
        self.advantages[:] = 0.0
        self.returns[:] = 0.0
        if self.store_costs:
            self.costs[:] = 0.0
            self.cost_values[:] = 0.0
            self.cost_advantages[:] = 0.0
            self.cost_returns[:] = 0.0
        self.ptr = 0

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float,
            done: bool, log_prob: float, value: float,
            cost: float = 0.0, cost_value: float = 0.0) -> None:
        """
        Store one transition at the current write position.

        Args:
            obs        : Observation array, shape (obs_dim,).
            action     : Action array, shape (act_dim,).
            reward     : Scalar reward for this step.
            done       : True if the episode terminated or was truncated.
            log_prob   : Log-probability of ``action`` under the current policy.
            value      : Reward-critic estimate V(s).
            cost       : Per-step safety cost (C-PPO only; ignored otherwise).
            cost_value : Cost-critic estimate V_C(s) (C-PPO only).
        """
        assert self.ptr < self.buffer_size, (
            f"Buffer overflow: ptr={self.ptr} >= buffer_size={self.buffer_size}. "
            "Call reset() before adding more transitions."
        )
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        if self.store_costs:
            self.costs[self.ptr] = cost
            self.cost_values[self.ptr] = cost_value
        self.ptr += 1

    def compute_advantages(self, last_value: float) -> None:
        """
        Compute GAE-λ reward advantages and discounted returns in-place.

        Must be called *once* after the rollout is fully collected and
        *before* iterating over minibatches.

        Args:
            last_value: Reward-critic estimate V(s) for the state immediately
                        *after* the last stored transition.
        """
        gae = 0.0
        for t in reversed(range(self.buffer_size)):
            non_terminal = 1.0 - self.dones[t]
            next_val = last_value if t == self.buffer_size - 1 else self.values[t + 1]
            delta = self.rewards[t] + self.gamma * next_val * non_terminal - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            self.advantages[t] = gae
        self.returns[:] = self.advantages + self.values

    def compute_cost_advantages(self, last_cost_value: float) -> None:
        """
        Compute GAE-λ cost advantages and cost returns in-place (C-PPO only).

        Applies the identical GAE formula to the cost stream using
        ``cost_values`` as the cost-critic baseline.

        Must be called *after* ``compute_advantages`` and *before* iterating
        over minibatches.

        Args:
            last_cost_value: Cost-critic estimate V_C(s) for the state
                             immediately after the last stored transition.

        Raises:
            RuntimeError: If called when ``store_costs=False``.
        """
        if not self.store_costs:
            raise RuntimeError(
                "compute_cost_advantages requires store_costs=True."
            )
        gae = 0.0
        for t in reversed(range(self.buffer_size)):
            non_terminal = 1.0 - self.dones[t]
            next_cval = (
                last_cost_value
                if t == self.buffer_size - 1
                else self.cost_values[t + 1]
            )
            delta = (
                self.costs[t]
                + self.gamma * next_cval * non_terminal
                - self.cost_values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            self.cost_advantages[t] = gae
        self.cost_returns[:] = self.cost_advantages + self.cost_values

    def get_minibatches(self, batch_size: int) -> Iterator[dict]:
        """
        Yield randomly shuffled minibatches over the full buffer.

        Advantages are yielded *raw* (not normalised); normalisation is
        applied inside the agent's update loop over the full buffer.

        Args:
            batch_size: Number of transitions per minibatch.

        Yields:
            Dict with keys:
                ``obs``, ``actions``, ``log_probs_old``, ``advantages``,
                ``returns``
                and, when ``store_costs=True``:
                ``costs``, ``cost_advantages``, ``cost_returns``.
                All values are numpy float32 arrays.
        """
        indices = np.random.permutation(self.buffer_size)
        for start in range(0, self.buffer_size, batch_size):
            idx = indices[start : start + batch_size]
            batch: dict = {
                "obs": self.obs[idx],
                "actions": self.actions[idx],
                "log_probs_old": self.log_probs[idx],
                "advantages": self.advantages[idx],
                "returns": self.returns[idx],
            }
            if self.store_costs:
                batch["costs"] = self.costs[idx]
                batch["cost_advantages"] = self.cost_advantages[idx]
                batch["cost_returns"] = self.cost_returns[idx]
            yield batch
