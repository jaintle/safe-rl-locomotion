# safe-rl-locomotion: Constrained Reinforcement Learning on Hopper-v4
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18818384.svg)](https://doi.org/10.5281/zenodo.18818384) ![CI](https://github.com/jaintle/safe-rl-locomotion/actions/workflows/ci.yml/badge.svg)


This repository presents a controlled empirical study of constrained policy optimization, implementing baseline PPO (Schulman et al., 2017) and a Lagrangian-penalized variant (C-PPO, following Achiam et al., 2017) on the Hopper-v4 continuous-control benchmark. The study is designed around multi-seed evaluation, honest reporting of reward–constraint tradeoffs, and full reproducibility: every run saves its resolved configuration, seeds all random state, and produces a fixed-schema metrics log. All reported results are mean ± standard deviation across three independent seeds; no cherry-picking or best-of-N selection is applied.

---

## Research Objective

This project investigates the reward–constraint tradeoff in constrained reinforcement learning using the Lagrangian relaxation approach. Specifically, we:

1. Implement baseline Proximal Policy Optimization (Schulman et al., 2017) as an unconstrained reference policy.
2. Implement a Lagrangian-penalized variant of PPO (C-PPO) in which a dual variable penalizes constraint violations during policy optimization, following the framework of Achiam et al. (2017).
3. Evaluate both algorithms on Hopper-v4 across multiple random seeds and training budgets to characterize long-horizon convergence and constraint satisfaction reliability.
4. Quantify the empirical tradeoff between expected episodic return and episodic constraint cost under a fixed cost limit.

The primary research questions are: (a) does the Lagrangian dual mechanism reliably enforce the constraint at scale, (b) what is the magnitude of the return penalty incurred by constraint satisfaction, and (c) how does this tradeoff evolve with training budget?

---

## Empirical Snapshot (Hopper-v4, 1M Steps, 3 Seeds)

| Algorithm | Return (mean ± std) | Cost (mean ± std) | Seeds feasible |
|-----------|---------------------|-------------------|----------------|
| PPO       | 2507 ± 864          | 565 ± 228         | 0 / 3          |
| C-PPO     | 570 ± 294           | 35 ± 23           | 3 / 3          |

<p align="center">
  <img src="reports/figures/hopper_v4/pareto_1m.png" width="550">
</p>

*Final return vs final cost (mean ± std across seeds). Dashed line indicates constraint limit (d = 80).*

Baseline PPO maximizes episodic return without any constraint awareness, achieving a mean return of 2507 across three seeds. However, it systematically violates the action-magnitude constraint at every checkpoint, with mean episodic cost of 565 — more than seven times the limit of 80. No seed satisfies the constraint at the final checkpoint.

C-PPO, equipped with a Lagrangian dual variable updated once per rollout, reliably enforces the constraint throughout training. At 1M steps, all three seeds are feasible (mean cost 35, well below the limit of 80), while maintaining a mean episodic return of 570. The dual multiplier converges stably across seeds with no oscillatory or divergent behavior observed.

These results empirically demonstrate that the Lagrangian relaxation mechanism produces a stable and consistent reward–constraint tradeoff under the training conditions studied. The tradeoff is substantial: constraint satisfaction is achieved at a cost of approximately 77% of baseline PPO return. This magnitude is consistent with the tightness of the constraint relative to unconstrained PPO behavior and is reported here without qualification.

---

## Experimental Protocol

**Environment.** Hopper-v4 as provided by Gymnasium 0.29 with the MuJoCo physics backend. Observations are 11-dimensional joint state vectors; actions are 3-dimensional continuous joint torques in [-1, 1].

**Algorithms.** Baseline PPO with clipped surrogate objective, generalized advantage estimation (GAE-λ), advantage normalization, and a diagonal Gaussian policy with learnable log-standard deviation. C-PPO extends this with a separate cost critic V_C(s), cost GAE computed identically to reward GAE, and a dual variable λ updated once per rollout via gradient ascent on the Lagrangian: λ ← clip(λ + α_λ · (avg_cost − d), 0, λ_max).

**Cost function.** Per-step cost is a binary indicator: c(s, a) = 1 if mean|a| > 0.25, else 0. Episodic cost is the sum of per-step costs. The cost limit is d = 80.0, calibrated against observed PPO costs (~158 at 200k steps) to define a constraint that is meaningful but achievable within a moderate training budget.

**Seeds.** Three independent runs per algorithm: seeds 0, 1, 2. Seeds are set for Python, NumPy, PyTorch, and Gymnasium environment resets.

**Training budgets.** 200k steps (pilot), 500k steps, 1M steps.

**Evaluation.** Deterministic evaluation (mean action, no sampling) over 10 episodes at every 10,000 training steps. Evaluation episodes use fixed seeds (1000 + i for episode i) to reduce stochasticity. Reported metrics are eval_return_mean and eval_cost_mean per evaluation point.

**Aggregation.** Multi-seed mean ± standard deviation computed by aligning runs on the global step index. Constraint satisfaction is assessed at the final checkpoint: a run is considered feasible if its final eval_cost_mean ≤ d.

**Hyperparameters.** All runs use CleanRL-style defaults unless noted: n_steps = 2048, lr = 3×10⁻⁴ (linear annealing), clip ε = 0.2, GAE λ = 0.95, discount γ = 0.99, 10 minibatch epochs per update. C-PPO additionally uses lr_lambda = 0.01, lambda_max = 10.0.

---

## Key Findings

At 1M training steps across three seeds:

- PPO achieves mean episodic return of 2507 ± 864 but incurs mean episodic cost of 565 ± 228. Zero of three seeds satisfy the cost constraint (cost ≤ 80) at the final checkpoint.
- C-PPO achieves mean episodic return of 570 ± 294, a substantial reduction relative to PPO, but incurs mean episodic cost of 35 ± 23. All three seeds satisfy the cost constraint at the final checkpoint.
- The reward–cost tradeoff is consistent across training budgets: C-PPO return improves from 500k to 1M steps while remaining feasible, indicating that the Lagrangian mechanism does not prevent further policy improvement once the constraint is satisfied.
- The dual multiplier λ converges stably under the update rule used; no oscillatory or divergent behavior is observed across seeds.
- PPO exhibits substantially higher cost variance than C-PPO, reflecting the absence of any regularization on action magnitude.

These findings are consistent with the theoretical predictions of the Lagrangian constrained RL framework: the dual mechanism trades reward for constraint compliance, and the tradeoff is stable at the training scales studied here.

---

## Setup

### Requirements

- Python 3.11
- MuJoCo physics engine (installed automatically via the `mujoco` pip package)

### Install

```bash
git clone <repo-url>
cd safe-rl-locomotion

python3.11 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# Optional: install the package itself in editable mode so
# `import safe_rl_locomotion` works without PYTHONPATH manipulation.
# The training scripts insert src/ into sys.path automatically,
# so this step is only needed for interactive use or external tooling.
pip install -e .
```

### Verify Installation

```bash
python scripts/smoke_env.py
# Expected output: SMOKE_OK
```

---

## Usage

### Train Baseline PPO

```bash
python scripts/train_ppo.py \
    --env_id Hopper-v4 \
    --seed 0 \
    --total_timesteps 1000000 \
    --save_dir runs/ppo_hopper_s0 \
    --eval_every 10000
```

PPO training computes `eval_cost_mean` and `eval_cost_std` at every evaluation interval using the same cost function as C-PPO. The training objective is not modified; cost is monitored for comparison purposes only.

### Train Constrained PPO (C-PPO)

```bash
python scripts/train_cppo.py \
    --env_id Hopper-v4 \
    --seed 0 \
    --total_timesteps 1000000 \
    --save_dir runs/cppo_hopper_s0 \
    --eval_every 10000 \
    --cost_limit 80.0 \
    --cost_fn action_magnitude
```

### Evaluate a Saved Checkpoint

```bash
python scripts/evaluate.py \
    --checkpoint runs/ppo_hopper_s0/checkpoints/step_01000000.pt \
    --env_id Hopper-v4 \
    --n_episodes 20
```

### Generate Per-Run Plots

```bash
# Single run
python scripts/make_plots.py \
    --run_dir runs/ppo_hopper_s0 \
    --env_id Hopper-v4 --seed 0 --total_timesteps 1000000 \
    --out_dir reports/figures

# Side-by-side comparison
python scripts/make_plots.py \
    --ppo_csv  runs/ppo_hopper_s0/metrics.csv \
    --cppo_csv runs/cppo_hopper_s0/metrics.csv \
    --env_id Hopper-v4 --seed 0 --total_timesteps 1000000 \
    --cost_limit 80.0 \
    --out_dir reports/figures
```

---

## Reproduction

### Fast verification (~2–4 minutes on CPU)

```bash
pytest -q
```

Runs smoke tests (5k-step training) and determinism checks. Slow learning-regression tests are excluded by default and can be run with `pytest -m slow`.

### Full multi-seed benchmarks

```bash
source .venv/bin/activate

# 500k benchmark: 3 seeds × 2 algorithms (≈3–6 hours on CPU)
bash scripts/reproduce_hopper_v4_500k.sh

# 1M benchmark: 3 seeds × 2 algorithms (≈6–12 hours on CPU)
bash scripts/reproduce_hopper_v4_1m.sh
```

Each script runs all training jobs sequentially, generates per-run plots under `runs/<run_name>/plots/`, and invokes `scripts/aggregate_results.py` to produce multi-seed summary figures and a results table in `reports/figures/hopper_v4/`.

### Manual multi-seed aggregation

```bash
python scripts/aggregate_results.py \
    --ppo_dirs  runs/ppo_hopper_v4_t025_seed0_1m  runs/ppo_hopper_v4_t025_seed1_1m  runs/ppo_hopper_v4_t025_seed2_1m \
    --cppo_dirs runs/cppo_hopper_v4_t025_limit80_seed0_1m runs/cppo_hopper_v4_t025_limit80_seed1_1m runs/cppo_hopper_v4_t025_limit80_seed2_1m \
    --env_id Hopper-v4 --budget 1000000 --cost_limit 80.0 \
    --out_dir reports/figures/hopper_v4 --tag 1m
```

---

## Results (Hopper-v4)

Full results, per-budget analysis, and discussion are in [`reports/results_hopper_v4.md`](reports/results_hopper_v4.md).

### Summary at 1M steps (seeds 0, 1, 2)

| Algorithm | Return (mean ± std) | Cost (mean ± std) | Seeds feasible |
|---|---|---|---|
| PPO   | 2507 ± 864 | 565 ± 228 | 0 / 3 |
| C-PPO |  570 ± 294 |  35 ±  23 | 3 / 3 |

Cost limit: 80.0. Feasibility assessed at final checkpoint.

### Summary figures

| Figure | Description |
|---|---|
| `reports/figures/hopper_v4/return_overlay_1m.png` | PPO vs C-PPO return (mean ± std across seeds) |
| `reports/figures/hopper_v4/cost_overlay_1m.png` | PPO vs C-PPO cost with constraint threshold |
| `reports/figures/hopper_v4/lambda_curves_1m.png` | C-PPO dual variable per seed and mean |
| `reports/figures/hopper_v4/pareto_1m.png` | Final return vs final cost with error bars |

---

## Repository Structure

```
safe-rl-locomotion/
├── configs/
│   ├── ppo.yaml                         # PPO hyperparameters
│   └── cppo.yaml                        # C-PPO hyperparameters and safety parameters
├── reports/
│   ├── experiment_log.md                # Structured log: one entry per implementation phase
│   ├── report.md                        # Short technical report
│   ├── results_hopper_v4.md             # Per-budget results, analysis, and discussion
│   └── figures/
│       └── hopper_v4/                   # Auto-generated summary figures and tables
├── scripts/
│   ├── smoke_env.py                     # Gymnasium + MuJoCo installation check
│   ├── train_ppo.py                     # PPO training entry point
│   ├── train_cppo.py                    # C-PPO training entry point
│   ├── evaluate.py                      # Post-hoc deterministic policy evaluation
│   ├── make_plots.py                    # Per-run plot generation from metrics.csv
│   ├── aggregate_results.py             # Multi-seed aggregation, summary figures, markdown table
│   ├── reproduce_hopper_v4_500k.sh      # Full 500k benchmark (3 seeds, PPO + C-PPO)
│   └── reproduce_hopper_v4_1m.sh        # Full 1M benchmark (3 seeds, PPO + C-PPO)
├── src/
│   └── safe_rl_locomotion/
│       ├── ppo.py                       # PPO agent
│       ├── cppo_lagrangian.py           # C-PPO agent, cost functions, Lagrangian multiplier
│       ├── buffers.py                   # Rollout buffer with dual GAE (reward + cost)
│       ├── utils.py                     # Seeds, MLP factory, MetricLogger, checkpointing
│       ├── eval.py                      # Deterministic evaluation loop
│       └── plotting.py                  # Training curve visualisation
├── tests/
│   ├── test_smoke.py                    # 5k-step smoke tests for PPO and C-PPO
│   ├── test_determinism.py              # Same-seed reproducibility check (ATOL = 5.0)
│   └── test_learning_regression.py      # 50k-step learning regression (marked slow)
├── requirements.txt                     # Pinned dependencies
├── setup.py                             # Legacy editable install support (pip install -e .)
├── pyproject.toml                       # PEP 517 build configuration
└── README.md
```

---

## Design Notes

**No large RL framework dependencies.** Agents are implemented in the style of CleanRL: self-contained, readable, and free of abstraction layers that obscure algorithmic behavior.

**Reproducibility as a first-class property.** Every run saves its resolved config, seeds all random number generators, and produces a metrics.csv with a fixed schema. Two runs with the same seed and config produce evaluation returns within a tolerance of 5.0.

**Honest evaluation.** Deterministic evaluation (mean action) is used consistently. Results are reported as mean ± std over multiple seeds, not best-of-N. Constraint satisfaction is assessed quantitatively, not qualitatively.

**Cost design without simulator access.** The cost function operates on observations and actions only. No MuJoCo internal state (contact forces, constraint torques) is accessed.

---

## Future Research Directions

This repository is intended as a foundation for further research in safe reinforcement learning for locomotion. Three natural directions for extension are outlined below.

### A. Environment Generalization

The current evaluation is limited to Hopper-v4. Extending to Walker2d-v4 and HalfCheetah-v4 would allow characterization of how the reward–constraint tradeoff varies across locomotion tasks with different morphologies and action dimensionalities. It remains an open empirical question whether the Lagrangian mechanism's convergence properties and the magnitude of the return penalty generalize across these settings, or whether task structure significantly modulates constraint satisfaction rates.

### B. Alternative Safety Formulations

The binary action-magnitude cost used here produces high-variance cost advantages due to its indicator structure. Replacing it with a smooth alternative — such as a squared torque penalty or an energy-based constraint — would reduce gradient variance and may improve constraint satisfaction rates early in training, before the dual variable has converged. A direct comparison to projection-based methods such as CPO (Achiam et al., 2017) would clarify the practical tradeoff between the computational simplicity of Lagrangian relaxation and the formal per-update constraint guarantees of trust-region projection approaches.

### C. Vision-Based Control and Sim-to-Real Transfer

Replacing proprioceptive state observations with pixel inputs introduces a representation learning challenge that is directly relevant to real-world robotics deployment. Pretrained visual encoders could be integrated while keeping the constrained policy optimization objective unchanged, allowing the cost mechanism to operate in representation space rather than raw observation space. Additional considerations for sim-to-real transfer — including domain randomization, partial observability, actuation delays, and contact dynamics mismatch — would constitute a natural extension of the constrained RL framework studied here and are a prerequisite for deployment on physical locomotion hardware.

---

## Limitations

- Single environment (Hopper-v4); generalization to other locomotion tasks (e.g., Walker2d-v4) has not been verified.
- Binary per-step cost produces high-variance cost advantages. Smooth cost functions would reduce estimator variance.
- The Lagrangian method provides no formal constraint guarantee analogous to CPO's trust-region projection. Violations are possible during early training before the dual variable converges.
- Three seeds is sufficient to characterize a trend but insufficient for publication-quality confidence intervals.
- No hyperparameter tuning; results reflect CleanRL defaults applied uniformly to both algorithms.
- CPU-only training; wall-clock times reported are approximate and hardware-dependent.

---

## References

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
2. Achiam, J., Held, D., Tamar, A., and Abbeel, P. (2017). *Constrained Policy Optimization*. ICML 2017. arXiv:1705.10528.
3. Todorov, E., Erez, T., and Tassa, Y. (2012). *MuJoCo: A physics engine for model-based control*. IROS 2012.

---

## Citation

If this artifact is useful for your research or teaching, please consider citing it:

```bibtex
@software{jain2026safe_rl_locomotion,
  author       = {Jain, Abhinav},
  title        = {safe-rl-locomotion: Reproducible Evaluation of Lagrangian Constrained PPO on Hopper-v4},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18818384}
}
```
