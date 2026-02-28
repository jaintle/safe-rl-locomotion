# Changelog

All notable changes to this project are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] - 2026-03-01

### Added

- Baseline PPO implementation for continuous-action MuJoCo environments.
  CleanRL-style: separate actor and critic MLPs, diagonal Gaussian policy,
  learnable log-std, GAE-λ, advantage normalization, clipped surrogate objective.
- Lagrangian Constrained PPO (C-PPO) with a separate cost critic V_C(s),
  dual GAE over cost returns, and per-rollout dual ascent update:
  λ ← clip(λ + α_λ · (avg_cost − d), 0, λ_max).
- Dual rollout buffer supporting simultaneous reward and cost GAE computation.
- Deterministic evaluation pipeline (mean action, fixed evaluation seeds).
- Binary action-magnitude cost function and torso-angle cost function,
  both operating on observations and actions only (no MuJoCo internals).
- Multi-seed aggregation script (`scripts/aggregate_results.py`) producing
  overlaid return/cost curves, lambda trajectories, and Pareto scatter plots.
- Reproduction shell scripts for 500k and 1M step benchmarks
  (3 seeds × 2 algorithms each).
- Multi-seed summary figures committed under `reports/figures/hopper_v4/`:
  return overlay, cost overlay, lambda curves, Pareto plot (500k and 1M).
- Three-layer test suite: smoke tests (5k steps), determinism checks
  (same-seed reproducibility, ATOL = 5.0), and learning regression test
  (50k steps, marked slow).
- Structured experiment log (`reports/experiment_log.md`) covering all
  implementation phases from scaffold through full benchmark runs.
- Per-run config saving, fixed-schema metrics CSV, and checkpoint saving
  at configurable evaluation intervals.
- `pyproject.toml` and `setup.py` for editable package installation.
- GitHub Actions CI workflow (`.github/workflows/ci.yml`) running smoke
  and determinism tests on push and pull request.

### Experimental Scope

- Environment: Hopper-v4 (Gymnasium 1.x, MuJoCo backend)
- Seeds: 0, 1, 2 (three independent runs per algorithm)
- Training budgets: 200k steps (pilot), 500k steps, 1M steps
- Cost function: binary action-magnitude indicator, threshold 0.25
- Cost limit: d = 80.0

### Notes

This is the initial public research artifact release. Results are documented
in `README.md` and `reports/results_hopper_v4.md`. All reported metrics are
mean ± standard deviation across three seeds evaluated at the final checkpoint.
No hyperparameter tuning was performed; CleanRL defaults were applied uniformly.

---

*For implementation details and experiment observations, see
`reports/experiment_log.md`.*
