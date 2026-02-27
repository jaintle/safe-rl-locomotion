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

## Entry 002

**Date:** 2026-02-28
**Task:** Phase 2 — Baseline PPO implementation
**Environment:** Hopper-v4 (target; smoke test pending user run)
**Timesteps:** 5 000 (smoke test budget)
**Seed(s):** 0

### Observations

**What was done:**
- Implemented all five core modules required for baseline PPO training:
  `utils.py`, `buffers.py`, `eval.py`, `ppo.py`, `scripts/train_ppo.py`.
- Activated `test_ppo_smoke_train` in `tests/test_smoke.py` (removed `@pytest.mark.skip`).
  `test_cppo_smoke_train` remains skipped (Phase 3).

**Implementation decisions:**

`utils.py`:
- `set_seeds` seeds Python `random`, NumPy, PyTorch (CPU + all CUDA devices).
- `make_mlp` produces a `nn.Sequential` with configurable hidden sizes and
  tanh/relu activation.
- `MetricLogger` opens the CSV file at construction time and writes the header
  immediately when `fieldnames` is provided, so `metrics.csv` exists on disk
  before any training steps run. This is required for the smoke test assertion.
- `save_checkpoint` stores actor, critic, log_std, and optimiser state in a
  single `.pt` file to allow full training resumption.

`buffers.py`:
- Pre-allocated numpy arrays; reset by zeroing (no reallocation per rollout).
- GAE computed in a single reversed loop.
  Non-terminal mask `(1 - done_t)` correctly handles both terminated and
  truncated episodes.
- Verified with manual calculation: constant `reward=1`, `value=0.5`,
  `gamma=0.99`, `gae_lambda=0.95` produces positive, monotonically-decreasing
  advantages. Done-flag propagation also verified analytically.
- `get_minibatches` shuffles indices with `np.random.permutation` before each
  yield pass.

`ppo.py`:
- Separate actor and critic MLPs (no shared torso); log-std is a learnable
  parameter vector not conditioned on state — standard for MuJoCo PPO.
- Orthogonal initialisation: hidden layers gain=sqrt(2), actor output gain=0.01
  (near-zero initial actions), critic output gain=1.0.
- Single Adam optimiser over all parameters (lr=3e-4, eps=1e-5).
- `select_action` uses `torch.no_grad()` for inference efficiency.
  `deterministic=True` returns the distribution mean for evaluation.
- Advantage normalisation applied over the full buffer before update epochs;
  raw advantages restored after update so the buffer state is not permanently mutated.
- Unclipped value function loss (simpler; clipped VF loss deferred to a later phase).
- Approx-KL and clip-fraction computed as diagnostic metrics.

`eval.py`:
- Deterministic evaluation: `select_action(obs, deterministic=True)`.
- Each episode seeded independently: episode i uses `eval_seed + i`.
- Returns `eval_return_mean`, `eval_return_std` (and cost variants for C-PPO).

`train_ppo.py`:
- `num_updates = max(1, total_timesteps // n_steps)`.
  For the smoke test (5 000 steps, n_steps=2048): 2 updates = 4 096 steps total.
  First eval triggers after step 4 096 (>= eval_every=2 500) — checkpoint saved.
- `sys.path.insert(0, .../src)` at the top of the script ensures the package is
  importable without a pip install, so test subprocesses can invoke it directly.
- LR annealing: `lr *= max(0, 1 - step / total_timesteps)` applied before each update.
- CSV schema is fixed upfront via `CSV_FIELDNAMES`; episode and eval rows both
  use this schema, with `float("nan")` for absent fields.

**What was fixed during implementation:**
- During logic verification, `utils.py` imports `torch` at module level, which
  prevented testing in the Linux VM (no torch there). MetricLogger and buffer
  logic were verified independently via isolated snippets — both passed.
- GAE done-mask: confirmed that the single mask `non_terminal = 1 - done_t`
  correctly zeroes the bootstrap value and terminates the GAE recursion in
  one expression; no separate handling needed for terminated vs truncated.

**Surprising behaviour:** None.

### Quantitative Notes

- Initial return: not yet measured (smoke test pending on user's machine)
- Final return: not yet measured
- Constraint violation rate: N/A (no constraint in baseline PPO)
- Lambda behaviour: N/A

### Debugging Notes

- Pure-Python logic tests (RolloutBuffer GAE, MetricLogger) passed with manual
  analytical verification in the VM.
- All 7 changed files pass Python `py_compile` syntax check.
- Smoke test requires the macOS venv (torch 2.10.0, gymnasium 1.2.3,
  mujoco 3.5.0); cannot be run inside the Linux VM.
- Expected smoke test execution time: < 30 s on CPU
  (4 096 env steps + 2 PPO updates + 10 eval episodes).

### Limitations Noted

- `num_updates = total_timesteps // n_steps` means actual steps run can be
  slightly less than `total_timesteps` when they are not divisible.
  For 1M/2048: 488 * 2048 = 999 424 steps (< 0.1% shortfall).
- Unclipped value function loss used (simpler than PPO paper's clipped variant).
- No early stopping on KL divergence.
- Evaluation noise from 10-episode finite sample is non-zero.
- Single seed only; variance across seeds not characterised in this phase.
- CPU-only; no GPU determinism flags set.

---
