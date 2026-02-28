"""
ppo.py
======
Baseline Proximal Policy Optimisation (PPO) — CleanRL-style implementation.

Reference:
    Schulman et al., "Proximal Policy Optimization Algorithms", 2017.
    https://arxiv.org/abs/1707.06347

Design:
    - Separate actor and critic MLPs (no shared torso).
    - Diagonal Gaussian policy: mean from actor network; log-std is a
      learnable parameter vector independent of state.
    - Orthogonal weight initialisation (standard for MuJoCo PPO).
    - Single Adam optimiser over actor + critic + log_std parameters.
    - Advantage normalisation over the *full* rollout buffer before updates.
    - Clipped surrogate objective; unclipped value function loss.
    - Optional entropy bonus (ent_coef, default 0.0).

Key cfg keys consumed by PPOAgent:
    hidden_sizes  : list of int, e.g. [64, 64]
    activation    : "tanh" | "relu"
    lr            : Adam learning rate
    n_epochs      : update epochs per rollout
    batch_size    : minibatch size
    clip_coef     : PPO clip epsilon ε
    vf_coef       : value-function loss coefficient
    ent_coef      : entropy bonus coefficient
    max_grad_norm : gradient-clipping threshold
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from safe_rl_locomotion.utils import make_mlp


class PPOAgent:
    """
    PPO agent for continuous-action MuJoCo environments.

    The actor outputs the mean of a diagonal Gaussian.
    The critic outputs a scalar state-value estimate V(s).
    The log-std is a learnable vector (not state-dependent).

    Args:
        obs_dim: Dimensionality of the observation space.
        act_dim: Dimensionality of the continuous action space.
        cfg    : Hyperparameter dict (from configs/ppo.yaml or merged config).
    """

    def __init__(self, obs_dim: int, act_dim: int, cfg: dict) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cfg = cfg

        hidden = cfg.get("hidden_sizes", [64, 64])
        activation = cfg.get("activation", "tanh")

        # --- Networks ---
        # Actor: obs → mean action  (final layer uses gain=0.01 per CleanRL)
        self.actor = make_mlp(obs_dim, act_dim, hidden, activation)
        # Critic: obs → scalar V(s)  (final layer uses gain=1.0)
        self.critic = make_mlp(obs_dim, 1, hidden, activation)

        # Log-std: single learnable vector, initialised to 0 (std ≈ 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # Orthogonal initialisation
        self._init_weights()

        # Single optimiser over all trainable parameters
        lr = float(cfg.get("lr", 3e-4))
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters())
            + list(self.critic.parameters())
            + [self.log_std],
            lr=lr,
            eps=1e-5,
        )

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """
        Orthogonal initialisation following CleanRL convention.

        Hidden layers: gain = sqrt(2).
        Actor output: gain = 0.01 (small for near-zero initial actions).
        Critic output: gain = 1.0.
        """
        def _ortho(module: nn.Linear, gain: float) -> None:
            nn.init.orthogonal_(module.weight, gain=gain)
            nn.init.zeros_(module.bias)

        hidden_gain = np.sqrt(2)

        # Actor hidden layers
        linears = [m for m in self.actor.modules() if isinstance(m, nn.Linear)]
        for layer in linears[:-1]:
            _ortho(layer, hidden_gain)
        _ortho(linears[-1], gain=0.01)

        # Critic hidden layers
        linears = [m for m in self.critic.modules() if isinstance(m, nn.Linear)]
        for layer in linears[:-1]:
            _ortho(layer, hidden_gain)
        _ortho(linears[-1], gain=1.0)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def _get_dist(self, obs_t: torch.Tensor) -> torch.distributions.Normal:
        """Return the diagonal Gaussian policy distribution for obs_t."""
        mean = self.actor(obs_t)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select an action from the current policy.

        Args:
            obs          : Observation array, shape (obs_dim,).
            deterministic: If True, use the distribution mean (no noise).
                           Use during evaluation; use False during training.

        Returns:
            action   : numpy array, shape (act_dim,).
            log_prob : Scalar log-probability of the action (sum over dims).
            value    : Scalar critic estimate V(obs).
        """
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            dist = self._get_dist(obs_t)
            action_t = dist.mean if deterministic else dist.sample()
            log_prob = dist.log_prob(action_t).sum(dim=-1)    # sum over action dims
            value = self.critic(obs_t).squeeze(-1)

        return (
            action_t.squeeze(0).numpy(),
            float(log_prob.item()),
            float(value.item()),
        )

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self, buffer) -> dict:
        """
        Run PPO update epochs over a completed rollout buffer.

        Advantage normalisation is applied once over the full buffer *before*
        the update epochs begin and is un-done afterwards to keep the buffer
        state consistent with what was collected.

        Args:
            buffer: Fully-populated ``RolloutBuffer`` with computed advantages.

        Returns:
            Dict of mean scalar metrics averaged across all minibatch updates:
                ``policy_loss``, ``value_loss``, ``entropy``,
                ``approx_kl``, ``clip_fraction``.
        """
        cfg = self.cfg
        n_epochs: int = int(cfg.get("n_epochs", 10))
        batch_size: int = int(cfg.get("batch_size", 64))
        clip_coef: float = float(cfg.get("clip_coef", 0.2))
        vf_coef: float = float(cfg.get("vf_coef", 0.5))
        ent_coef: float = float(cfg.get("ent_coef", 0.0))
        max_grad_norm: float = float(cfg.get("max_grad_norm", 0.5))

        # ------------------------------------------------------------------
        # Normalise advantages over the full buffer (in-place).
        # Save mean/std so we can restore after the update.
        # ------------------------------------------------------------------
        adv_raw = buffer.advantages.copy()
        adv_mean = float(buffer.advantages.mean())
        adv_std = float(buffer.advantages.std()) + 1e-8
        buffer.advantages = (buffer.advantages - adv_mean) / adv_std

        # Accumulate metrics across all minibatch steps
        sum_metrics = dict(
            policy_loss=0.0,
            value_loss=0.0,
            entropy=0.0,
            approx_kl=0.0,
            clip_fraction=0.0,
        )
        n_updates = 0

        for _ in range(n_epochs):
            for batch in buffer.get_minibatches(batch_size):
                obs_t = torch.as_tensor(batch["obs"], dtype=torch.float32)
                act_t = torch.as_tensor(batch["actions"], dtype=torch.float32)
                lp_old = torch.as_tensor(batch["log_probs_old"], dtype=torch.float32)
                adv_t = torch.as_tensor(batch["advantages"], dtype=torch.float32)
                ret_t = torch.as_tensor(batch["returns"], dtype=torch.float32)

                # ---- Policy loss ----------------------------------------
                dist = self._get_dist(obs_t)
                lp_new = dist.log_prob(act_t).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(lp_new - lp_old)
                pg_loss1 = -adv_t * ratio
                pg_loss2 = -adv_t * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # ---- Value loss ------------------------------------------
                value_pred = self.critic(obs_t).squeeze(-1)
                value_loss = 0.5 * ((value_pred - ret_t) ** 2).mean()

                # ---- Total loss -----------------------------------------
                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters())
                    + list(self.critic.parameters())
                    + [self.log_std],
                    max_grad_norm,
                )
                self.optimizer.step()

                # ---- Diagnostics (no grad needed) -----------------------
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()

                sum_metrics["policy_loss"] += policy_loss.item()
                sum_metrics["value_loss"] += value_loss.item()
                sum_metrics["entropy"] += entropy.item()
                sum_metrics["approx_kl"] += approx_kl
                sum_metrics["clip_fraction"] += clip_frac
                n_updates += 1

        # Restore original advantages
        buffer.advantages = adv_raw

        # Return per-update averages
        return {k: v / max(n_updates, 1) for k, v in sum_metrics.items()}
