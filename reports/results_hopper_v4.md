# Hopper-v4 Benchmark Results

**Environment:** Hopper-v4 (MuJoCo continuous-control)
**Date:** 2026-02-28 (200k pilot); full 500k/1M results pending.
**Algorithms:** Baseline PPO · Safety-Constrained PPO (Lagrangian, C-PPO)

---

## Experiment Setup

### Cost Function Calibration

Initial runs with `cost_limit=0.1` caused C-PPO policy collapse: the lambda
multiplier spiked immediately because every episode incurred costs far above
the limit.  A 200k pilot run of unconstrained PPO revealed the actual cost
scale with `cost_fn=action_magnitude, threshold=0.25`:

> PPO @ 200k steps: `eval_return_mean ≈ 551.95`, `eval_cost_mean ≈ 158.8`

Based on this measurement `cost_limit=80.0` was chosen — approximately the
50th percentile of observed PPO costs — to create a meaningful constraint that
the Lagrangian can actually satisfy within a moderate training budget.

### Configuration

| Parameter | Value |
|---|---|
| Environment | Hopper-v4 |
| Cost function | `action_magnitude` |
| Cost threshold | 0.25 |
| Cost limit d (C-PPO) | 80.0 |
| PPO n_steps | 2048 |
| PPO lr | 3 × 10⁻⁴ (linear annealing) |
| PPO clip ε | 0.2 |
| GAE λ | 0.95 |
| Discount γ | 0.99 |
| C-PPO lr_lambda | 0.01 |
| C-PPO lambda_max | 10.0 |
| Eval episodes | 10 (deterministic, seed 1000+i) |
| Eval interval | 10 000 steps |
| Seeds | 0, 1, 2 |

### Reproduction Commands

```bash
source .venv/bin/activate

# 500k benchmark (all seeds, PPO + C-PPO)
bash scripts/reproduce_hopper_v4_500k.sh

# 1M benchmark (all seeds, PPO + C-PPO)
bash scripts/reproduce_hopper_v4_1m.sh
```

Each script runs all 6 training jobs sequentially, generates per-run plots
under `runs/<run_name>/plots/`, and then calls `scripts/aggregate_results.py`
to write summary figures to `reports/figures/hopper_v4/`.

---

## Pilot Results (200k Steps, Seed 0)

These results are from the initial calibration run; multi-seed 500k/1M results
will replace this table once the benchmark scripts have been executed.

| Algorithm | Steps | Seed | Return (eval mean) | Cost (eval mean) | Lambda (final) |
|---|---|---|---|---|---|
| PPO   | 200 000 | 0 | 551.95 | 158.8 | — |
| C-PPO | 200 000 | 0 | 357.73 |  90.1 | 2.98 |

**Observation:** C-PPO achieves a clear reward–cost tradeoff relative to
unconstrained PPO.  Cost is reduced by ~43 % (158.8 → 90.1) while return
drops by ~35 % (551.95 → 357.73).  The lambda value of 2.98 indicates the
constraint was active (cost still above limit) at 200k steps; longer training
is expected to drive lambda higher and further reduce cost at the expense of
additional return.

---

## 500k Steps — Summary Table

*Pending execution of `bash scripts/reproduce_hopper_v4_500k.sh`.*

<!-- auto-generated table will be inserted here by aggregate_results.py -->
<!-- copy from reports/figures/hopper_v4/summary_500k.md after runs complete -->

| Algorithm | Budget | Seeds | Return (mean±std) | Cost (mean±std) | Constraint met? |
|---|---|---|---|---|---|
| PPO   | 500,000 | — | TBD | TBD | — |
| C-PPO | 500,000 | — | TBD | TBD | — |

---

## 1M Steps — Summary Table

*Pending execution of `bash scripts/reproduce_hopper_v4_1m.sh`.*

<!-- auto-generated table will be inserted here by aggregate_results.py -->
<!-- copy from reports/figures/hopper_v4/summary_1m.md after runs complete -->

| Algorithm | Budget | Seeds | Return (mean±std) | Cost (mean±std) | Constraint met? |
|---|---|---|---|---|---|
| PPO   | 1,000,000 | — | TBD | TBD | — |
| C-PPO | 1,000,000 | — | TBD | TBD | — |

---

## Figures

All figures are generated automatically by the reproduce scripts and saved to
`reports/figures/hopper_v4/`.  The table below lists the expected files;
links will resolve once training runs complete.

### 500k Steps

| Figure | Description |
|---|---|
| [`return_overlay_500k.png`](figures/hopper_v4/return_overlay_500k.png) | PPO vs C-PPO `eval_return_mean` vs steps (mean ± std across seeds) |
| [`cost_overlay_500k.png`](figures/hopper_v4/cost_overlay_500k.png) | PPO vs C-PPO `eval_cost_mean` vs steps (mean ± std across seeds) |
| [`lambda_curves_500k.png`](figures/hopper_v4/lambda_curves_500k.png) | C-PPO lambda per seed + mean vs steps |
| [`pareto_500k.png`](figures/hopper_v4/pareto_500k.png) | Final return vs final cost scatter with error bars (Pareto view) |

### 1M Steps

| Figure | Description |
|---|---|
| [`return_overlay_1m.png`](figures/hopper_v4/return_overlay_1m.png) | PPO vs C-PPO `eval_return_mean` vs steps (mean ± std across seeds) |
| [`cost_overlay_1m.png`](figures/hopper_v4/cost_overlay_1m.png) | PPO vs C-PPO `eval_cost_mean` vs steps (mean ± std across seeds) |
| [`lambda_curves_1m.png`](figures/hopper_v4/lambda_curves_1m.png) | C-PPO lambda per seed + mean vs steps |
| [`pareto_1m.png`](figures/hopper_v4/pareto_1m.png) | Final return vs final cost scatter with error bars (Pareto view) |

---

## Discussion

**Reward–cost tradeoff.** The 200k pilot demonstrates the expected qualitative
behaviour: C-PPO reduces cost at the expense of reward.  The magnitude of the
tradeoff will vary across seeds and budgets.  At 500k/1M steps, lambda
convergence should reduce cost more reliably, but return may remain lower than
unconstrained PPO throughout training.

**Constraint satisfaction rate.** With `cost_limit=80.0`, C-PPO is not
guaranteed to satisfy the constraint at all times.  The Lagrangian method
performs dual ascent outside the policy gradient loop; violations are possible
during early training before lambda grows large enough.  The summary table
records how many seeds satisfy the constraint at the final checkpoint.

**Seed variance.** Hopper-v4 exhibits high variance across random seeds,
particularly for PPO.  Three seeds is sufficient to demonstrate a consistent
trend but insufficient for precise confidence intervals.  Results should be
interpreted with this caveat.

**Budget effects.** Hopper-v4 PPO typically reaches ~2 000–3 000 return by
1M steps.  C-PPO with a tight constraint may plateau earlier.  The 200k pilot
already shows meaningful learning; longer runs are expected to widen the
return gap between the two algorithms while potentially narrowing the cost gap
as lambda stabilises.

---

## Limitations

- Single environment (Hopper-v4); results may not transfer to Walker2d-v4.
- Binary cost indicator (0/1 per step × episode steps) produces high-variance
  cost estimates; smoother cost functions would reduce noise.
- No hyperparameter tuning; CleanRL defaults are used throughout.
- CPU-only training; GPU runs may give different timing characteristics.
- Three seeds is a minimum for variance estimation; more seeds are needed for
  publication-quality claims.
