# Robot-Safe PPO

Reproducible reinforcement learning repository implementing:

1. **Baseline PPO** — Schulman et al., 2017 ([arXiv:1707.06347](https://arxiv.org/abs/1707.06347))
2. **Safety-Constrained PPO (C-PPO Lagrangian)** — inspired by CPO, Achiam et al., 2017 ([arXiv:1705.10528](https://arxiv.org/abs/1705.10528))

Target environments: **Hopper-v4**, **Walker2d-v4** (optional second environment).

> **Status:** Phase 3 complete — both PPO and C-PPO Lagrangian fully implemented.
> Smoke tests active for both algorithms. Ready for full training runs.
> This repository is intended as credible proof-of-work for robotics and RL research labs.

---

## Project Overview

| Component | Description |
|---|---|
| `src/robot_safe_ppo/ppo.py` | Baseline PPO agent (CleanRL-style, continuous actions) |
| `src/robot_safe_ppo/cppo_lagrangian.py` | Constrained PPO with Lagrangian dual update |
| `src/robot_safe_ppo/buffers.py` | Rollout buffer with GAE computation |
| `src/robot_safe_ppo/utils.py` | Seeds, MLP factory, MetricLogger, checkpointing |
| `src/robot_safe_ppo/eval.py` | Deterministic policy evaluation |
| `src/robot_safe_ppo/plotting.py` | Training-curve visualisation |
| `scripts/train_ppo.py` | PPO training entry point |
| `scripts/train_cppo.py` | C-PPO training entry point |
| `scripts/evaluate.py` | Post-hoc deterministic evaluation |
| `scripts/make_plots.py` | Generate PNG plots from metrics.csv |
| `configs/ppo.yaml` | PPO hyperparameters |
| `configs/cppo.yaml` | C-PPO hyperparameters (includes safety parameters) |
| `tests/` | Three-layer test suite (smoke, determinism, regression) |
| `reports/experiment_log.md` | Structured log of all experiments |

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

Run the smoke environment test to confirm MuJoCo and Gymnasium are working:

```bash
python scripts/smoke_env.py
```

Expected output: `SMOKE_OK`

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
    --checkpoint runs/ppo_hopper_s0/checkpoints/step_1000000.pt \
    --env_id Hopper-v4 \
    --n_episodes 20
```

### Generate Plots

```bash
# Single run
python scripts/make_plots.py \
    --ppo_csv runs/ppo_hopper_s0/metrics.csv \
    --env_id Hopper-v4 --seed 0 --total_timesteps 1000000 \
    --out_dir reports/figures

# Comparison: PPO vs C-PPO
python scripts/make_plots.py \
    --ppo_csv runs/ppo_hopper_s0/metrics.csv \
    --cppo_csv runs/cppo_hopper_s0/metrics.csv \
    --env_id Hopper-v4 --seed 0 --total_timesteps 1000000 \
    --cost_limit 0.1 \
    --out_dir reports/figures
```

---

## Running Tests

```bash
# Fast tests only (smoke + determinism stubs — always runnable)
pytest -q

# Slow learning-regression tests (requires full training run ~hours on CPU)
pytest -m slow -v tests/test_learning_regression.py
```

---

## Reproduction

### Smoke tests (fast, ~30–60 s each on CPU)

```bash
# PPO: 5 000-step smoke training run
python scripts/train_ppo.py \
    --env_id Hopper-v4 \
    --seed 0 \
    --total_timesteps 5000 \
    --save_dir runs/ppo_smoke \
    --eval_every 2500

# C-PPO: 5 000-step smoke training run
python scripts/train_cppo.py \
    --env_id Hopper-v4 \
    --seed 0 \
    --total_timesteps 5000 \
    --save_dir runs/cppo_smoke \
    --eval_every 2500 \
    --cost_limit 0.1

# Run all smoke tests via pytest
pytest -q tests/test_smoke.py
```

### Full training runs (1M steps — several hours on CPU)

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

> Final evaluation returns and constraint violation rates will be reported here after full training completes.

---

## Repository Structure

```
robot-safe-ppo/
├── configs/
│   ├── ppo.yaml              # Baseline PPO hyperparameters
│   └── cppo.yaml             # C-PPO (Lagrangian) hyperparameters
├── reports/
│   ├── experiment_log.md     # Structured experiment log
│   └── figures/              # Generated plots (populated after training)
├── scripts/
│   ├── smoke_env.py          # MuJoCo/Gymnasium health check
│   ├── train_ppo.py          # PPO training entry point
│   ├── train_cppo.py         # C-PPO training entry point
│   ├── evaluate.py           # Post-hoc evaluation
│   └── make_plots.py         # Plot generation
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

## Design Principles

- **Minimal and readable**: CleanRL-style single-file agents, no large RL framework dependencies.
- **Reproducible**: All runs set seeds for Python, NumPy, PyTorch, and the Gymnasium environment. Config files are saved alongside outputs.
- **Honest evaluation**: Deterministic evaluation (mean action, no sampling) at fixed intervals. Results reported as mean ± std over evaluation episodes.
- **Safety cost design**: Costs are computed purely from observations and/or actions — no MuJoCo internal state access required.

---

## Limitations

- Single-seed results have high variance; production-quality claims would require multi-seed sweeps.
- Short training budgets (CPU-only) may not reach near-optimal policies.
- The Lagrangian method does not provide the formal safety guarantees of CPO's trust-region projection.
- No sim-to-real transfer, pixel-based observations, or large-scale hyperparameter search in this project phase.

---

## References

1. Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
2. Achiam, J. et al. (2017). *Constrained Policy Optimization*. ICML 2017. arXiv:1705.10528.
3. Todorov, E. et al. (2012). *MuJoCo: A physics engine for model-based control*. IROS 2012.
