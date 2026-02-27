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

Dual (lambda) update rule (applied per rollout):
    λ ← clip(λ + α_λ · (avg_episode_cost − cost_limit), 0, lambda_max)

Policy gradient:
    The Lagrangian modifies the PPO policy loss by adding a penalised cost
    policy term:
        L_policy = L_reward_ppo + λ · L_cost_ppo
    where L_cost_ppo is the clipped surrogate loss over cost advantages.

Cost critic:
    A separate value network V_C(s) is maintained to produce cost-baseline
    estimates.  It is trained with an MSE regression loss against
    cost GAE returns.  This critic is entirely separate from the reward
    critic V(s); they share no weights.

Cost function contract:
    Cost functions receive ``(obs, action, next_obs)`` as numpy arrays and
    return a non-negative scalar for that transition.  No MuJoCo internals
    are accessed.

Implemented cost functions:
    - ``cost_action_magnitude``: indicator cost when mean |action| > threshold.
    - ``cost_torso_angle``     : indicator cost when |torso_angle| > threshold
                                 (uses obs[1] for Hopper-v4).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from robot_safe_ppo.ppo import PPOAgent
from robot_safe_ppo.utils import make_mlp


# ---------------------------------------------------------------------------
# Cost functions
# ---------------------------------------------------------------------------

def cost_action_magnitude(obs: np.ndarray, action: np.ndarray,
                           next_obs: np.ndarray,
                           threshold: float = 0.8) -> float:
    """
    Binary indicator cost: penalise high-magnitude actions.

    Returns 1.0 when mean |action| > threshold, else 0.0.

    Args:
        obs       : Current observation (unused).
        action    : Action taken at this step.
        next_obs  : Next observation (unused).
        threshold : Safety threshold on mean |action|.

    Returns:
        1.0 if unsafe, 0.0 otherwise.
    """
    return float(np.mean(np.abs(action)) > threshold)


def cost_torso_angle(obs: np.ndarray, action: np.ndarray,
                     next_obs: np.ndarray,
                     threshold: float = 0.2) -> float:
    """
    Binary indicator cost: penalise large torso angles (Hopper-v4).

    For Hopper-v4, ``obs[1]`` is the torso angle in radians.  Returns 1.0
    when |torso_angle| > threshold.

    Args:
        obs       : Current observation.  obs[1] is used as torso angle.
        action    : Action taken (unused).
        next_obs  : Next observation (unused).
        threshold : Angular deviation threshold (radians).

    Returns:
        1.0 if unsafe, 0.0 otherwise.
    """
    torso_angle = float(obs[1]) if len(obs) > 1 else 0.0
    return float(abs(torso_angle) > threshold)


def get_cost_fn(name: str, cfg: dict):
    """
    Return a cost function callable from a name string and config dict.

    The returned callable has signature: ``(obs, action, next_obs) -> float``.

    Args:
        name: ``"action_magnitude"`` or ``"torso_angle"``.
        cfg : Config dict; threshold values are read from here.

    Returns:
        Bound cost function with threshold already applied.

    Raises:
        ValueError: If ``name`` is not a recognised cost function.
    """
    if name == "action_magnitude":
        thresh = float(cfg.get("cost_action_magnitude_threshold", 0.8))
        return lambda obs, act, nobs: cost_action_magnitude(obs, act, nobs, thresh)
    elif name == "torso_angle":
        thresh = float(cfg.get("cost_torso_angle_threshold", 0.2))
        return lambda obs, act, nobs: cost_torso_angle(obs, act, nobs, thresh)
    else:
        raise ValueError(
            f"Unknown cost function '{name}'. Choose 'action_magnitude' or 'torso_angle'."
        )


# ---------------------------------------------------------------------------
# Lagrangian multiplier
# ---------------------------------------------------------------------------

class LagrangianMultiplier:
    """
    Manages the Lagrange multiplier λ for the safety constraint.

    λ is updated via dual ascent after each rollout:
        λ ← clip(λ + α_λ · (avg_episode_cost − d),  0,  lambda_max)

    Args:
        init_lambda : Initial value of λ (clamped to [0, lambda_max]).
        lr_lambda   : Dual ascent step size α_λ.
        cost_limit  : Safety threshold d; constraint is E[cost_episode] ≤ d.
        lambda_max  : Upper bound on λ to prevent runaway growth.
    """

    def __init__(self, init_lambda: float = 0.0, lr_lambda: float = 1e-2,
                 cost_limit: float = 0.1, lambda_max: float = 10.0) -> None:
        self.cost_limit = cost_limit
        self.lr = lr_lambda
        self.lambda_max = lambda_max
        self.lam = float(np.clip(init_lambda, 0.0, lambda_max))

    def update(self, avg_episode_cost: float) -> float:
        """
        Apply one dual-ascent step.

        Args:
            avg_episode_cost: Mean cost per episode over the last rollout.

        Returns:
            Updated λ value.
        """
        self.lam = float(np.clip(
            self.lam + self.lr * (avg_episode_cost - self.cost_limit),
            0.0,
            self.lambda_max,
        ))
        return self.lam

    @property
    def value(self) -> float:
        """Current λ value."""
        return self.lam


# ---------------------------------------------------------------------------
# C-PPO Lagrangian Agent
# ---------------------------------------------------------------------------

def _save_cppo_checkpoint(agent: "CPPOLagrangianAgent",
                          path, metadata: dict) -> None:
    """
    Save C-PPO agent state to a ``.pt`` checkpoint.

    Extends the base PPO checkpoint with cost_critic weights and lambda.

    Args:
        agent   : CPPOLagrangianAgent instance.
        path    : Destination file path.
        metadata: Scalar metadata dict (step, seed, eval metrics, lambda).
    """
    import pathlib
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_state_dict": agent.actor.state_dict(),
            "critic_state_dict": agent.critic.state_dict(),
            "cost_critic_state_dict": agent.cost_critic.state_dict(),
            "log_std": agent.log_std.data.clone(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "lambda": agent.lagrangian.value,
            "metadata": metadata,
        },
        path,
    )


class CPPOLagrangianAgent(PPOAgent):
    """
    Constrained PPO agent with Lagrangian dual variable.

    Extends ``PPOAgent`` by:
    - Adding a cost critic network V_C(s).
    - Modifying the policy loss:
          L_policy = L_reward_ppo + λ · L_cost_ppo
      where L_cost_ppo uses clipped surrogates over cost advantages.
    - Adding a cost-critic regression loss alongside the reward-critic loss.
    - Maintaining a ``LagrangianMultiplier`` updated after each rollout.

    The PPO backbone (actor, reward critic, log_std) is identical to
    ``PPOAgent``.  The single optimiser covers all parameters including the
    cost critic.

    Additional cfg keys consumed:
        lambda_init  : float, initial λ (default 0.0).
        lr_lambda    : float, dual ascent step size (default 0.01).
        cost_limit   : float, safety threshold d (default 0.1).
        lambda_max   : float, upper clamp on λ (default 10.0).
    """

    def __init__(self, obs_dim: int, act_dim: int, cfg: dict) -> None:
        # Initialise PPO backbone (sets self.actor, self.critic, self.log_std,
        # self._init_weights(), and a temporary self.optimizer).
        super().__init__(obs_dim, act_dim, cfg)

        hidden = cfg.get("hidden_sizes", [64, 64])
        activation = cfg.get("activation", "tanh")

        # Cost critic: V_C(s) → scalar cost-value estimate
        self.cost_critic = make_mlp(obs_dim, 1, hidden, activation)
        self._init_cost_critic_weights()

        # Rebuild optimizer to include cost critic parameters.
        # (Replaces the one created by PPOAgent.__init__.)
        lr = float(cfg.get("lr", 3e-4))
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters())
            + list(self.critic.parameters())
            + list(self.cost_critic.parameters())
            + [self.log_std],
            lr=lr,
            eps=1e-5,
        )

        # Lagrangian multiplier
        self.lagrangian = LagrangianMultiplier(
            init_lambda=float(cfg.get("lambda_init", 0.0)),
            lr_lambda=float(cfg.get("lr_lambda", 1e-2)),
            cost_limit=float(cfg.get("cost_limit", 0.1)),
            lambda_max=float(cfg.get("lambda_max", 10.0)),
        )

    def _init_cost_critic_weights(self) -> None:
        """Orthogonal init for the cost critic (same convention as reward critic)."""
        linears = [m for m in self.cost_critic.modules() if isinstance(m, nn.Linear)]
        hidden_gain = float(np.sqrt(2))
        for layer in linears[:-1]:
            nn.init.orthogonal_(layer.weight, gain=hidden_gain)
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(linears[-1].weight, gain=1.0)
        nn.init.zeros_(linears[-1].bias)

    def get_cost_value(self, obs: np.ndarray) -> float:
        """
        Return the cost-critic's estimate V_C(obs).

        Args:
            obs: Observation array, shape (obs_dim,).

        Returns:
            Scalar cost-value estimate.
        """
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            return float(self.cost_critic(obs_t).squeeze(-1).item())

    # ------------------------------------------------------------------
    # C-PPO update
    # ------------------------------------------------------------------

    def update(self, buffer, avg_episode_cost: float) -> dict:  # type: ignore[override]
        """
        Run C-PPO update epochs and update λ.

        The policy loss includes a Lagrangian penalty over cost advantages:
            L_policy = L_reward_ppo + λ · L_cost_ppo

        Both reward and cost critics are updated with MSE regression losses.
        λ is updated *once* after all gradient epochs using ``avg_episode_cost``.

        Args:
            buffer           : Fully-populated ``RolloutBuffer`` (store_costs=True)
                               with reward and cost advantages computed.
            avg_episode_cost : Mean episode cost over the last rollout batch.
                               Used to update the Lagrange multiplier.

        Returns:
            Dict of mean scalar metrics:
                ``policy_loss``, ``cost_policy_loss``, ``value_loss``,
                ``cost_value_loss``, ``entropy``, ``approx_kl``,
                ``clip_fraction``, ``lambda``.
        """
        cfg = self.cfg
        n_epochs: int = int(cfg.get("n_epochs", 10))
        batch_size: int = int(cfg.get("batch_size", 64))
        clip_coef: float = float(cfg.get("clip_coef", 0.2))
        vf_coef: float = float(cfg.get("vf_coef", 0.5))
        ent_coef: float = float(cfg.get("ent_coef", 0.0))
        max_grad_norm: float = float(cfg.get("max_grad_norm", 0.5))

        lam = self.lagrangian.value

        # ------------------------------------------------------------------
        # Normalise reward advantages over the full buffer.
        # ------------------------------------------------------------------
        adv_raw = buffer.advantages.copy()
        adv_mean = float(buffer.advantages.mean())
        adv_std = float(buffer.advantages.std()) + 1e-8
        buffer.advantages = (buffer.advantages - adv_mean) / adv_std

        # Normalise cost advantages over the full buffer.
        cadv_raw = buffer.cost_advantages.copy()
        cadv_mean = float(buffer.cost_advantages.mean())
        cadv_std = float(buffer.cost_advantages.std()) + 1e-8
        buffer.cost_advantages = (buffer.cost_advantages - cadv_mean) / cadv_std

        sum_metrics = dict(
            policy_loss=0.0,
            cost_policy_loss=0.0,
            value_loss=0.0,
            cost_value_loss=0.0,
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
                cadv_t = torch.as_tensor(batch["cost_advantages"], dtype=torch.float32)
                cret_t = torch.as_tensor(batch["cost_returns"], dtype=torch.float32)

                # ---- Policy and entropy ----------------------------------
                dist = self._get_dist(obs_t)
                lp_new = dist.log_prob(act_t).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                ratio = torch.exp(lp_new - lp_old)

                # ---- Reward policy loss (clipped surrogate) --------------
                pg_r1 = -adv_t * ratio
                pg_r2 = -adv_t * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                reward_policy_loss = torch.max(pg_r1, pg_r2).mean()

                # ---- Cost policy loss (clipped surrogate) ----------------
                pg_c1 = cadv_t * ratio
                pg_c2 = cadv_t * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                cost_policy_loss = torch.max(pg_c1, pg_c2).mean()

                # ---- Value losses ----------------------------------------
                reward_value_pred = self.critic(obs_t).squeeze(-1)
                reward_vf_loss = 0.5 * ((reward_value_pred - ret_t) ** 2).mean()

                cost_value_pred = self.cost_critic(obs_t).squeeze(-1)
                cost_vf_loss = 0.5 * ((cost_value_pred - cret_t) ** 2).mean()

                # ---- Combined loss ---------------------------------------
                # policy: reward - ent_bonus + lambda * cost_penalty
                # critic: vf_coef * (reward_vf + cost_vf)
                loss = (
                    reward_policy_loss
                    + lam * cost_policy_loss
                    - ent_coef * entropy
                    + vf_coef * (reward_vf_loss + cost_vf_loss)
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters())
                    + list(self.critic.parameters())
                    + list(self.cost_critic.parameters())
                    + [self.log_std],
                    max_grad_norm,
                )
                self.optimizer.step()

                # ---- Diagnostics -----------------------------------------
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()

                sum_metrics["policy_loss"] += reward_policy_loss.item()
                sum_metrics["cost_policy_loss"] += cost_policy_loss.item()
                sum_metrics["value_loss"] += reward_vf_loss.item()
                sum_metrics["cost_value_loss"] += cost_vf_loss.item()
                sum_metrics["entropy"] += entropy.item()
                sum_metrics["approx_kl"] += approx_kl
                sum_metrics["clip_fraction"] += clip_frac
                n_updates += 1

        # Restore normalised buffers
        buffer.advantages = adv_raw
        buffer.cost_advantages = cadv_raw

        # Lambda update (once per rollout, after gradient updates)
        new_lam = self.lagrangian.update(avg_episode_cost)

        metrics = {k: v / max(n_updates, 1) for k, v in sum_metrics.items()}
        metrics["lambda"] = new_lam
        return metrics
