# Robot-Safe PPO

Reproducible reinforcement learning repository implementing:

1. **Baseline PPO** — Schulman et al., 2017 ([arXiv:1707.06347](https://arxiv.org/abs/1707.06347))
2. **Safety-Constrained PPO (C-PPO Lagrangian)** — inspired by CPO, Achiam et al., 2017 ([arXiv:1705.10528](https://arxiv.org/abs/1705.10528))

Target environments: **Hopper-v4**, **Walker2d-v4** (optional second environment).

> **Status:** Phase 6 complete — benchmark scripts and aggregation tooling in place.
> PPO and C-PPO Lagrangian fully implemented. Three-layer test suite active.
> Cost calibrated: `cost_fn=action_magnitude`, `threshold=0.25`, `cost_limit=80.0`.
> Pilot results (200k, seed 0): PPO return ≈ 552, C-PPO return ≈ 358 / cost ≈ 90.
> Multi-seed 500k/1M benchmarks reproducible via shell scripts (see Reproduction).
> This repository is intended as credible proof-of-work for robotics and RL research labs.

---

## Project Overview

| Component | Description |
|---|---|
| `src/robot_safe_ppo/ppo.py` | Baseline PPO agent (CleanRL-style, continuous actions) |
| `src/robot_safe_ppo/cppo_lagrangian.py` | Constrained PPO with Lagrangian dual update |
| `src/robot_safe_ppo/buffers.py` | Rollout buffer with GAE (reward + cost streams) |
| `src/robot_safe_ppo/utils.py` | Seeds, MLP factory, MetricLogger, checkpointing |
| `src/robot_safe_ppo/eval.py` | Deterministic policy evaluation |
| `src/robot_safe_ppo/plotting.py` | Training-curve visualisation (matplotlib) |
| `scripts/train_ppo.py` | PPO training entry point |
| `scripts/train_cppo.py` | C-PPO training entry point |
| `scripts/evaluate.py` | Post-hoc deterministic evaluation |
| `scripts/make_plots.py` | Generate PNG plots from metrics.csv |
| `configs/ppo.yaml` | PPO hyperparameters |
| `configs/cppo.yaml` | C-PPO hyperparameters (includes safety parameters) |
| `tests/test_smoke.py` | Fast smoke tests (env + 5 k-step PPO/CPPO runs) |
| `tests/test_determinism.py` | Determinism checks (two same-seed runs, ATOL=5.0) |
| `tests/test_learning_regression.py` | Slow learning regression (50 k steps, delta≥+10) |
| `reports/experiment_log.md` | Structured log of all experiments |
| `reports/report.md` | Short technical report |

---

## Setup

### Requirements

- Python 3.11
- MuJoCo physics engine (installed automatically via the `mujoco` pip package)

### Install

```bash
# Clone the repository
git clone <repo-url>
cd robot-safe-ppo

# Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
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

> **Cost logging:** PPO now computes `eval_cost_mean` and `eval_cost_std` during
> every deterministic evaluation using the same cost function as C-PPO
> (`action_magnitude` by default, configurable via `configs/ppo.yaml`).
> The PPO **training objective is not modified** — cost is evaluated only, not
> optimised.  This allows direct apples-to-apples cost comparison between PPO
> and C-PPO in plots and the results table.

### Train Constrained PPO (C-PPO)

```bash
python scripts/train_cppo.py \
    --env_id Hopper-v4 \
    --seed 0 \
    --total_timesteps 1000000 \
    --save_dir runs/cppo_hopper_s0 \
    --eval_every 10000 \
    --cost_limit 0.1 \
    --cost_fn action_magnitude
```

### Evaluate a Saved Checkpoint

```bash
python scripts/evaluate.py \
    --checkpoint runs/ppo_hopper_s0/checkpoints/step_01000000.pt \
    --env_id Hopper-v4 \
    --n_episodes 20
```

### Generate Plots

```bash
# Single PPO run (shorthand)
python scripts/make_plots.py \
    --run_dir runs/ppo_hopper_s0 \
    --env_id Hopper-v4 --seed 0 --total_timesteps 1000000 \
    --out_dir reports/figures

# Comparison: PPO vs C-PPO
python scripts/make_plots.py \
    --ppo_csv  runs/ppo_hopper_s0/metrics.csv \
    --cppo_csv runs/cppo_hopper_s0/metrics.csv \
    --env_id Hopper-v4 --seed 0 --total_timesteps 1000000 \
    --cost_limit 0.1 \
    --out_dir reports/figures
```

Expected output artefacts per run directory:

```
<save_dir>/
├── metrics.csv          # per-episode and eval metrics (both algos log eval_cost_mean)
├── config.yaml          # resolved hyperparameters used for this run
└── checkpoints/
    └── step_*.pt        # periodic policy checkpoints

reports/figures/
├── returns_vs_steps.png
├── costs_vs_steps.png   # C-PPO runs only
├── lambda_vs_steps.png  # C-PPO runs only
├── losses_vs_steps.png
└── comparison.png       # when both --ppo_csv and --cppo_csv provided
```

---

## Running Tests

```bash
# Fast tests: smoke + determinism (~2–4 min on CPU)
pytest -q

# PPO pytest smoke test only
pytest -q tests/test_smoke.py::test_ppo_smoke_train

# C-PPO pytest smoke test only
pytest -q tests/test_smoke.py::test_cppo_smoke_train

# PPO determinism test only
pytest -q tests/test_determinism.py::test_ppo_determinism

# C-PPO determinism test only
pytest -q tests/test_determinism.py::test_cppo_determinism

# Slow learning-regression tests (PPO + C-PPO, ~10–30 min on CPU)
pytest -m slow -v tests/test_learning_regression.py
```

---

## Reproduction

### Smoke runs (fast, ~30–60 s each)

```bash
# PPO smoke
python scripts/train_ppo.py \
    --env_id Hopper-v4 --seed 0 \
    --total_timesteps 5000 \
    --save_dir runs/ppo_smoke \
    --eval_every 2500

# C-PPO smoke
python scripts/train_cppo.py \
    --env_id Hopper-v4 --seed 0 \
    --total_timesteps 5000 \
    --save_dir runs/cppo_smoke \
    --eval_every 2500 \
    --cost_limit 0.1

# Pytest smoke suite
pytest -q tests/test_smoke.py
```

### Full training runs (1 M steps — several hours on CPU)

```bash
python scripts/train_ppo.py \
    --env_id Hopper-v4 --seed 0 \
    --total_timesteps 1000000 \
    --save_dir runs/ppo_hopper_s0 \
    --eval_every 10000

python scripts/train_cppo.py \
    --env_id Hopper-v4 --seed 0 \
    --total_timesteps 1000000 \
    --save_dir runs/cppo_hopper_s0 \
    --eval_every 10000 \
    --cost_limit 0.1
```

> Final evaluation returns and constraint violation rates will be appended to
> `reports/report.md` after full training completes.

---

## Repository Structure

```
robot-safe-ppo/
├── configs/
│   ├── ppo.yaml              # Baseline PPO hyperparameters
│   └── cppo.yaml             # C-PPO (Lagrangian) hyperparameters
├── reports/
│   ├── experiment_log.md        # Structured experiment log (all phases)
│   ├── report.md                # Short technical report
│   ├── results_hopper_v4.md     # Benchmark results and discussion
│   └── figures/
│       └── hopper_v4/           # Summary figures (auto-generated)
├── scripts/
│   ├── smoke_env.py             # MuJoCo/Gymnasium health check
│   ├── train_ppo.py             # PPO training entry point
│   ├── train_cppo.py            # C-PPO training entry point
│   ├── evaluate.py              # Post-hoc evaluation
│   ├── make_plots.py            # Per-run plot generation
│   ├── aggregate_results.py     # Multi-seed aggregation + summary figures
│   ├── reproduce_hopper_v4_500k.sh  # Reproduce 500k benchmark
│   └── reproduce_hopper_v4_1m.sh   # Reproduce 1M benchmark
├── src/
│   └── robot_safe_ppo/
│       ├── __init__.py
│       ├── ppo.py
│       ├── cppo_lagrangian.py
│       ├── buffers.py
│       ├── utils.py
│       ├── eval.py
│       └── plotting.py
├── tests/
│   ├── test_smoke.py
│   ├── test_determinism.py
│   └── test_learning_regression.py
├── requirements.txt
└── README.md
```

---

## Results (Hopper-v4)

Full results are documented in **[`reports/results_hopper_v4.md`](reports/results_hopper_v4.md)**.

### Pilot (200k steps, seed 0)

| Algorithm | Return (eval) | Cost (eval) | Lambda |
|---|---|---|---|
| PPO   | 551.95 | 158.8 | — |
| C-PPO | 357.73 |  90.1 | 2.98 |

C-PPO reduces cost by ~43 % at a ~35 % return penalty, demonstrating the
reward–constraint tradeoff. The `cost_limit=80.0` was calibrated against
observed PPO costs (`~158`) to create a meaningful, achievable constraint.

### Multi-seed benchmarks (500k, 1M)

Run the reproduction scripts to populate `reports/figures/hopper_v4/`:

```bash
source .venv/bin/activate
bash scripts/reproduce_hopper_v4_500k.sh   # seeds 0,1,2 × PPO+CPPO
bash scripts/reproduce_hopper_v4_1m.sh    # seeds 0,1,2 × PPO+CPPO
```

Key summary figures (generated automatically):

| Figure | Description |
|---|---|
| `reports/figures/hopper_v4/return_overlay_500k.png` | PPO vs C-PPO return (mean ± std) |
| `reports/figures/hopper_v4/cost_overlay_500k.png` | PPO vs C-PPO cost (mean ± std) |
| `reports/figures/hopper_v4/lambda_curves_500k.png` | C-PPO lambda convergence |
| `reports/figures/hopper_v4/pareto_500k.png` | Return–cost Pareto scatter |

---

## Design Principles

- **Minimal and readable**: CleanRL-style single-file agents, no large RL framework dependencies.
- **Reproducible**: All runs set seeds for Python, NumPy, PyTorch, and Gymnasium. Config files are saved alongside outputs.
- **Honest evaluation**: Deterministic evaluation (mean action, no sampling noise) at fixed intervals. Results reported as mean ± std over N episodes.
- **Safety cost design**: Costs are computed purely from observations and/or actions — no MuJoCo internal state access required.
- **Three-layer test coverage**: Smoke (always fast), determinism (fast sanity), regression (slow but meaningful).

---

## Limitations

- Single-seed results have high variance; production-quality conclusions require multi-seed sweeps.
- Short CPU training budgets may not reach near-optimal policies.
- The Lagrangian method does not provide the formal safety guarantees of CPO's trust-region projection; it is an approximation that can temporarily violate the constraint.
- Binary cost indicators (0/1) produce high-variance cost advantages; smooth cost functions would reduce this.
- No sim-to-real transfer, pixel-based observations, or large-scale hyperparameter search in this project phase.

---

## References

1. Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
2. Achiam, J. et al. (2017). *Constrained Policy Optimization*. ICML 2017. arXiv:1705.10528.
3. Todorov, E. et al. (2012). *MuJoCo: A physics engine for model-based control*. IROS 2012.
