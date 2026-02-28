# Release: safe-rl-locomotion v1.0.0

**Date:** 2026-03-01
**DOI:** *(assigned upon Zenodo upload)*
**Repository:** https://github.com/jaintle/safe-rl-locomotion

---

## Research Summary

This release is the initial public artifact for a controlled empirical study
of constrained policy optimization on a continuous locomotion benchmark.
The study implements baseline Proximal Policy Optimization (PPO; Schulman
et al., 2017) and a Lagrangian-penalized variant (C-PPO; following Achiam
et al., 2017), evaluates both on Hopper-v4 across three random seeds and two
training budgets (500k and 1M steps), and quantifies the empirical
reward–constraint tradeoff under a fixed action-magnitude safety constraint.
All results are reported as mean ± standard deviation across seeds with no
cherry-picking; constraint satisfaction is assessed quantitatively at the
final checkpoint.

---

## What Is Included

- **Baseline PPO** — CleanRL-style implementation: separate actor/critic MLPs,
  diagonal Gaussian policy, learnable log-std, GAE-λ, advantage normalization,
  clipped surrogate objective, linear learning-rate annealing.
- **Lagrangian C-PPO** — extends PPO with a separate cost critic V_C(s), cost
  GAE computed identically to reward GAE, and a dual variable λ updated once
  per rollout via gradient ascent on the Lagrangian.
- **Dual rollout buffer** — simultaneous reward and cost advantage estimation
  (GAE-λ) over a single rollout.
- **Deterministic evaluation pipeline** — mean action (no sampling noise),
  fixed evaluation seeds, periodic checkpointing.
- **Reproduction scripts** — `scripts/reproduce_hopper_v4_500k.sh` and
  `scripts/reproduce_hopper_v4_1m.sh` run all 6 training jobs sequentially
  and invoke multi-seed aggregation automatically.
- **Multi-seed aggregation** — `scripts/aggregate_results.py` produces return
  overlays, cost overlays, lambda trajectories, and Pareto scatter plots.
- **Committed figures** — all summary plots for 500k and 1M budgets are
  committed under `reports/figures/hopper_v4/`.
- **Three-layer test suite** — smoke (5k steps), determinism (same-seed
  reproducibility, ATOL = 5.0), and learning regression (50k steps).
- **GitHub Actions CI** — runs smoke and determinism tests on every push and
  pull request.
- **Structured experiment log** — `reports/experiment_log.md` documents every
  implementation phase and experiment run.

---

## Experimental Protocol

| Parameter | Value |
|-----------|-------|
| Environment | Hopper-v4 (Gymnasium 1.x, MuJoCo backend) |
| Seeds | 0, 1, 2 |
| Training budgets | 500k steps, 1M steps |
| Evaluation | Deterministic (mean action), 10 episodes, every 10k steps |
| Cost function | Binary action-magnitude indicator: c = 1 if mean\|a\| > 0.25 |
| Cost limit | d = 80.0 |
| Network | MLP [64, 64], tanh, separate actor and critic |
| Optimiser | Adam, lr = 3×10⁻⁴, linear annealing |
| n_steps | 2048 |
| n_epochs | 10 |
| clip ε | 0.2 |
| GAE λ | 0.95 |
| γ | 0.99 |
| C-PPO lr_lambda | 0.01 |
| C-PPO lambda_max | 10.0 |

Hyperparameters follow CleanRL defaults for MuJoCo continuous control.
No hyperparameter tuning was performed.

---

## Results at 1M Steps (Seeds 0, 1, 2)

| Algorithm | Return (mean ± std) | Cost (mean ± std) | Seeds feasible |
|-----------|---------------------|-------------------|----------------|
| PPO       | 2507 ± 864          | 565 ± 228         | 0 / 3          |
| C-PPO     | 570 ± 294           | 35 ± 23           | 3 / 3          |

Cost limit: d = 80.0. Feasibility assessed at final checkpoint
(eval_cost_mean ≤ d). Full per-budget results are in
[`reports/results_hopper_v4.md`](../reports/results_hopper_v4.md).

---

## Reproducibility Instructions

**Requirements:** Python 3.11, MuJoCo (installed via pip).

```bash
git clone https://github.com/jaintle/safe-rl-locomotion.git
cd safe-rl-locomotion

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Verify environment
python scripts/smoke_env.py  # Expected: SMOKE_OK

# Run fast test suite (~2–4 min on CPU)
pytest -q

# Full 1M benchmark: 3 seeds × 2 algorithms (~6–12 hours on CPU)
bash scripts/reproduce_hopper_v4_1m.sh
```

Per-run plots are written to `runs/<run_name>/plots/`. Aggregated
multi-seed figures are written to `reports/figures/hopper_v4/`.

---

## Zenodo DOI

This release corresponds to the Zenodo-archived version of the repository.
The DOI should be cited when referencing the artifact in academic work.
See the Citation section of `README.md` for a BibTeX entry.

---

## References

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. (2017).
   *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
2. Achiam, J., Held, D., Tamar, A., and Abbeel, P. (2017).
   *Constrained Policy Optimization*. ICML 2017. arXiv:1705.10528.
