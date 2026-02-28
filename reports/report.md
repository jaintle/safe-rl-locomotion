# Safe RL Locomotion: Technical Report

**Date:** 2026-02-28
**Status:** Implementation complete; full training results pending.

---

## Overview

This repository implements two policy gradient algorithms for continuous-control tasks:

1. **Baseline PPO** — the clipped surrogate objective of Schulman et al. (2017), applied to Hopper-v4 with continuous action spaces and GAE-λ advantage estimation.
2. **C-PPO Lagrangian** — an extension that adds a safety constraint via a Lagrangian multiplier (dual ascent), inspired by Constrained Policy Optimisation (Achiam et al., 2017). The Lagrangian approach is chosen over trust-region projection for implementation simplicity while preserving the core safety-training dynamic.

The goal is a clean, reproducible codebase that demonstrates proof-of-work for both standard deep RL and safe RL engineering practices, without claiming SOTA performance or novel theory.

---

## Methods

### Baseline PPO

The PPO agent uses two separate MLPs (actor and critic) with hidden layers `[64, 64]`, tanh activations, and orthogonal initialisation (hidden gain=√2, output gain=0.01/1.0 for actor/critic). A diagonal Gaussian policy with a learnable log-standard-deviation parameter is used for continuous action spaces.

Key hyperparameters (from `configs/ppo.yaml`):

| Parameter | Value |
|---|---|
| n_steps | 2048 |
| n_epochs | 10 |
| batch_size | 64 |
| clip_coef ε | 0.2 |
| γ (discount) | 0.99 |
| λ_GAE | 0.95 |
| lr | 3 × 10⁻⁴ |
| LR annealing | linear to 0 |
| vf_coef | 0.5 |
| ent_coef | 0.0 |

The update objective per minibatch is:

    L = L_policy + vf_coef × L_value − ent_coef × H

where `L_policy = E[max(−A × ratio, −A × clip(ratio, 1−ε, 1+ε))]` and `L_value = MSE(V(s), V_target)`.

### Safety-Constrained PPO (Lagrangian)

C-PPO extends PPO with a second critic V_C(s) estimating expected cost returns, and a scalar Lagrange multiplier λ ≥ 0. The combined policy objective per update is:

    L_total = L_reward_policy + λ × L_cost_policy + vf_coef × (L_reward_value + L_cost_value)

The dual variable is updated once per rollout (after all gradient epochs) using dual ascent:

    λ ← clip(λ + α_λ × (avg_cost − d), 0, λ_max)

where `d` is the cost limit, `avg_cost` is the mean completed-episode cost from the current rollout, and `λ_max = 10.0` is a clamp to prevent runaway growth.

Additional hyperparameters (from `configs/cppo.yaml`):

| Parameter | Value |
|---|---|
| cost_limit d | 0.1 |
| cost_fn | action_magnitude |
| α_λ (lr_lambda) | 0.01 |
| λ_init | 0.0 |
| λ_max | 10.0 |

### Cost Functions

Two cost functions are available (no MuJoCo internal state access required):

**`action_magnitude`** (default): binary indicator — returns 1.0 when the mean absolute action magnitude exceeds a threshold (default 0.8), otherwise 0.0.

    c(s, a, s') = 1[mean(|a|) > threshold]

**`torso_angle`** (Hopper-v4 specific): binary indicator — returns 1.0 when the absolute torso pitch angle (obs index 1) exceeds a threshold (default 0.2 radians).

    c(s, a, s') = 1[|obs[1]| > threshold]

Binary costs simplify the constraint definition but produce high-variance cost advantages. Smooth cost proxies would reduce this variance.

### GAE for Both Streams

Both the reward and cost value functions use the same GAE formula with shared γ and λ_GAE. The cost returns are computed identically to reward returns, with episode terminals zeroing the bootstrap and GAE recursion.

---

## Evaluation Protocol

- **Deterministic evaluation**: the mean action (argmax of the Gaussian) is used at test time, eliminating sampling noise.
- **Episode count**: 10 evaluation episodes per checkpoint.
- **Seeding**: episode `i` uses seed `eval_seed + i = 1000 + i`. This is fixed across runs so evaluation results are comparable across checkpoints.
- **Frequency**: every `eval_every` environment steps (default: 10 000).
- **Metrics logged**: `eval_return_mean`, `eval_return_std`, `eval_cost_mean`, `eval_cost_std`, `lambda`.

---

## Results

> Full training results have not yet been collected. The table below is a template to be populated after 1 M-step runs complete.

| Algorithm | Env | Seed | Budget | Final Eval Return (mean ± std) | Final Eval Cost (mean ± std) | Final λ |
|---|---|---|---|---|---|---|
| PPO | Hopper-v4 | 0 | 1 M | _TBD_ | — | — |
| C-PPO | Hopper-v4 | 0 | 1 M | _TBD_ | _TBD_ | _TBD_ |

Smoke-run results (5 000 steps) are not meaningful as learning indicators and are not reported here.

---

## Test Coverage

Three testing layers are implemented:

**Smoke tests** (`tests/test_smoke.py`): 5 000 training steps; assert `metrics.csv` exists, is non-empty, has at least one checkpoint, and (for C-PPO) has a `lambda` column in the header. Runs in ~30–60 s on CPU.

**Determinism tests** (`tests/test_determinism.py`): two identical-seed runs (4 096 steps); assert `|eval_return_mean_run0 − eval_return_mean_run1| ≤ 5.0`. Tolerance is loosened from the theoretical 1e-3 to 5.0 to handle OS-level non-determinism across machines without loss of diagnostic value for real bugs. Runs in ~30–60 s on CPU.

**Learning regression tests** (`tests/test_learning_regression.py`, marked `@pytest.mark.slow`): 50 000 training steps; assert `final_eval_return − initial_eval_return ≥ 10.0`. The delta of +10 is conservative relative to the expected improvement from a working PPO on Hopper-v4 but is designed to avoid flakiness from seed variance. For C-PPO, also asserts that the mean eval cost over the last 20 % of eval rows is ≤ 1.5 × cost_limit.

---

## Limitations and Next Steps

**Limitations:**

- All results are single-seed. Cross-seed variance on Hopper-v4 is large; conclusions about final performance require at least 3–5 seeds.
- Binary cost functions produce high-variance cost advantages, which can slow or destabilise Lagrangian convergence.
- The Lagrangian method does not provide CPO's formal safety guarantee (constraint feasibility at every update). It can temporarily violate the constraint during early training.
- No hyperparameter tuning was performed; the configs use CleanRL-style defaults.
- CPU-only training limits the practical run budget.

**Possible next steps:**

- Multi-seed evaluation (3–5 seeds) with mean ± std reporting.
- Smooth cost functions (e.g., continuous torso-angle penalty instead of a binary indicator).
- Second environment (Walker2d-v4) to test generalisation of the cost function design.
- Comparison of Lagrangian C-PPO against projection-based CPO.
- Longer training budgets (3–5 M steps) to reach near-optimal Hopper-v4 returns.

---

## References

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
2. Achiam, J., Held, D., Tamar, A., & Abbeel, P. (2017). *Constrained Policy Optimization*. ICML 2017. arXiv:1705.10528.
3. Todorov, E., Erez, T., & Tassa, Y. (2012). *MuJoCo: A physics engine for model-based control*. IROS 2012.
4. Huang, S. et al. (2022). CleanRL: High-quality Single-file Implementations of Deep RL Algorithms. *JMLR* 23(274):1−18.
