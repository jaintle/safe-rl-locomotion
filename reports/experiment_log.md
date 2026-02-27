# Experiment Log — Robot-Safe PPO

Structured log of all implementation steps and experiment runs.
Each entry is appended in chronological order.
Used as the primary source for drafting the final technical report.

---

## Entry 001

**Date:** 2026-02-27
**Task:** Phase 1 — Repository scaffold
**Environment:** N/A (no training run)
**Timesteps:** N/A
**Seed(s):** N/A

### Observations

**What was done:**
- Created full directory layout: `src/robot_safe_ppo/`, `scripts/`, `configs/`, `tests/`, `reports/`.
- Wrote stub implementations with docstrings for all six source modules:
  `ppo.py`, `cppo_lagrangian.py`, `buffers.py`, `utils.py`, `eval.py`, `plotting.py`.
- Created training entry-point scripts with fully functional CLI parsers
  (`train_ppo.py`, `train_cppo.py`) that print resolved args and exit cleanly.
  Post-scaffold, these will be extended with the actual training loop.
- Created supporting scripts: `evaluate.py`, `make_plots.py` (CLI stubs).
- Wrote YAML config files for both algorithms with documented hyperparameter
  choices close to CleanRL MuJoCo defaults.
- Wrote three-layer test suite (smoke, determinism, regression) with all
  algorithm-dependent tests marked `@pytest.mark.skip` pending implementation.
  `test_smoke_env` (which wraps `scripts/smoke_env.py`) is immediately runnable.
- Updated `README.md` with project overview, setup instructions,
  CLI usage examples, and a placeholder Reproduction section.

**Design choices:**
- Chose separate `ppo.py` and `cppo_lagrangian.py` rather than a single file
  with flags, to keep each algorithm's logic isolated and reviewable.
- Placed cost functions in `cppo_lagrangian.py` alongside the agent so that
  the cost definition and the constrained objective are co-located.
- `LagrangianMultiplier` is a standalone class (not embedded in the agent) to
  allow independent unit testing of the dual update rule.
- `buffers.py` stores costs as an optional field (`store_costs=False` by
  default) so the same buffer implementation serves both PPO and C-PPO.
- `utils.MetricLogger` is the single write point for CSV output; both
  training scripts delegate all I/O here to maintain a consistent schema.
- Config YAML files deliberately do not use Hydra composition to keep the
  dependency graph flat; Hydra is available in `requirements.txt` but
  the plain `yaml.safe_load` path is simpler and more portable.

**What failed:** Nothing in this phase — scaffold is Python-syntax-valid and all
files import cleanly (modules with `NotImplementedError` bodies are expected).

**Surprising behaviour:** None.

### Quantitative Notes

- Initial return: N/A
- Final return: N/A
- Constraint violation rate: N/A
- Lambda behaviour: N/A

### Debugging Notes

- No bugs encountered during scaffold creation.
- Verified the existing `scripts/smoke_env.py` outputs `SMOKE_OK` against
  the pre-installed venv (gymnasium 1.2.3, mujoco 3.5.0, torch 2.10.0).

### Limitations Noted

- No algorithm logic exists yet; all `NotImplementedError` raises are
  intentional placeholders.
- The three slow test-layer tests cannot run until Phase 2 (PPO implementation)
  and Phase 3 (C-PPO implementation) are complete.
- Single-seed evaluation only in the test suite — this will be noted as a
  limitation in the final report.
- CPU-only development environment; GPU determinism may require additional
  `torch.use_deterministic_algorithms(True)` calls.
- `requirements.txt` pins exact versions; compatibility with newer package
  releases has not been tested.

---
