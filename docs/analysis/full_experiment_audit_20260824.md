# Full experiment audit: ML versus seven-day mean — 2026-08-24

## Executive conclusion

The project did not miss a universally superior seven-day mean from the start.
On the original `01_baseline_8m` holdout (`2026-03-27..2026-04-02`), an
independent recalculation on the same 135,344 prediction rows gives:

| Method | Target | Global WAPE | Bias |
|---|---|---:|---:|
| LightGBM baseline | observed sales | **25.9120%** | -0.9050% |
| Previous 7 observations mean | observed sales | 31.3086% | -4.5439% |
| Previous 14 observations mean | observed sales | 30.9201% | -5.2002% |
| Previous 7 observations mean | reconstructed demand | 36.9453% | +9.7925% |

LightGBM therefore beat the causal seven-observation mean by 5.40 percentage
points on the original evaluation. The current result is a later reversal, not
evidence that the initial model selection was false.

The project nevertheless missed the reversal for too long. There was no
mandatory, immutable end-to-end baseline suite attached to every production
scenario and data-contract change. By August, the current production system
was materially worse than a direct seven-calendar-day SKU quantity mean:

| Current comparison (`2026-08-11..23`, 8 valid dates) | Production | Mean 7 | Delta |
|---|---:|---:|---:|
| Bakery-day WAPE, observed sales | 18.8090% | **12.4591%** | -6.3499 pp |
| SKU-day WAPE, observed sales | 68.3563% | **38.8468%** | -29.5095 pp |
| Bakery-day WAPE, strict demand | 18.9855% | **12.6066%** | -6.3789 pp |
| SKU-day WAPE, strict demand | 67.5230% | **39.0358%** | -28.4871 pp |

Mean 7 wins on all eight dates for both targets. The result is not explained
only by forecast rows with zero observed sales: on positive-sales rows,
production WAPE is 51.33% and mean-7 WAPE is 28.83%.

## Scope audited

- `src/experiments/` legacy experiment family;
- 55 directories under `src/experiments_v2/`;
- 83 local `reports/**/summary.json` artifacts;
- `reports/experiment_log.jsonl`;
- stockout, allocation, demand reconstruction, bakery-total and production
  parity reports under `docs/`;
- the two 2026-08-24 end-to-end allocation/current-production evaluations;
- stored prediction artifacts for the original baseline and SKU benchmark.

This is a read-only audit. Production state and forecast tables were not
changed.

## Experiment-system findings

### 1. The original ML decision was supported by its holdout

The original LightGBM result was not merely compared with another ML variant.
The stored prediction rows allow an independent causal baseline calculation.
On that exact holdout, the model wins against means of 3, 7, 14 and 28 prior
observations. The best of those simple sales baselines is mean 14 at 30.9201%
WAPE, still 5.01 pp worse than ML.

This rules out the strongest version of the concern: mean 7 was not already
better on the original April test.

### 2. Earlier naive baselines did not test the present hypothesis

Simple methods appeared in the research history, but they were materially
different:

- experiment 67 used a rolling 14-observation SKU mean on five selected
  bakeries and a seven-day holdout;
- experiments 73 and 74 tested `lag7`, meaning repetition of the same weekday,
  at bakery-day level;
- several models used rolling means as input features, which is not the same as
  evaluating the rolling mean as an end-to-end forecast;
- predictive allocation experiments preserved incumbent category totals, so
  they could not detect that a direct SKU quantity baseline produced better
  totals when aggregated upward.

The current winning baseline is a direct bakery/SKU quantity forecast from the
previous seven calendar days, not last-week repetition and not a share profile
renormalized to the production total.

### 3. The April SKU benchmark was too narrow and used the wrong headline WAPE

Experiment 67 reported the arithmetic mean of pair-level WAPE:

- reported ML `avg_wmape`: 35.79%;
- reported mean-14 `avg_wmape`: 47.27%.

Recalculation from the stored prediction rows gives the volume-weighted global
figures actually relevant to total units:

- ML: 26.3476%;
- mean 14: 32.5468%.

The direction remains correct, but the headline metric was not the business
WAPE. The test also covered only 3,247 rows, five bakeries and seven days.

### 4. The monthly SKU benchmark is labelled more strongly than it ran

`sku_local_monthly/metrics.json` records:

- only three test days;
- `min_train_rows=999999`;
- Prophet unavailable;
- identical Prophet and local-LightGBM predictions, caused by fallback logic.

It is a smoke/fallback run, not evidence from trained monthly local models.
Its direct global WAPE is 29.8177% for the fallback model and 28.8550% for the
14-day mean. The simple baseline actually wins this artifact, but the report's
unweighted pair-average WAPE says 44.39% versus 45.87% and obscures that fact.
This was an early warning that should have triggered a proper rerun.

### 5. Comparability across the experiment registry is weak

The experiment set mixes:

- observed sales and several reconstructed-demand targets;
- SKU-day, bakery-day, allocation-only and hourly-profile tasks;
- 7-, 14- and 30-day holdouts;
- global WAPE and unweighted averages of group WAPE;
- historical model predictions, oracle category totals and live production
  snapshots;
- selected bakeries/SKUs and full-network samples.

Consequently, WAPE values from different directories cannot be ranked as one
leaderboard. The experiment log contains only two April production-training
entries and does not represent the research history. `INDEX.md` is also stale:
it documents 36 directories, while 55 experiment directories now exist.

### 6. The production system changed after the original validation

The live forecast is no longer the original flat SKU-level LightGBM experiment.
It is a multi-stage system:

1. bakery-day model;
2. category/SKU allocation and assortment filtering;
3. recent corrections, profile fallbacks and safeguards;
4. hourly allocation and serving snapshots.

The current result measures this whole system. The April result measured the
old SKU model on a static historical dataset. The change from 25.91% to 68.36%
cannot be interpreted as pure LightGBM model drift; much of it is pipeline,
universe and allocation error.

Evidence supporting this diagnosis:

- current bakery-day WAPE is about 19%, while SKU-day WAPE is about 68%;
- 229,607 forecast units, 15.30% of production mass, land on SKU-days with zero
  observed sales;
- even after removing zero-sales rows, the SKU error remains very high;
- product 1071 alone has production WAPE 71.33% versus 17.87% for mean 7;
- the current production forecast has +11.26% bias against observed sales.

### 7. Allocation research optimized inside a biased total constraint

The predictive-choice experiment correctly improved allocation when all
methods were forced to the same incumbent category totals:

- historical production: 48.4674% WAPE;
- predictive choice: 47.5289%;
- best inspected blend: 47.3890%.

But the unconstrained seven-day quantity mean scored 41.1801% on the same
historical interval. The allocation gain was real, but smaller than the error
in inherited totals. Because the experimental contract fixed those totals, it
could not answer the more important end-to-end question until the simple
quantity control was added later.

## Why the reversal was missed

The failure was governance and evaluation design, not the absence of all naive
models:

1. Baselines were treated as experiment-specific controls rather than permanent
   production challengers.
2. `lag7`, mean 14, rolling features and constrained mean shares were implicitly
   treated as covering the same idea as unconstrained mean 7.
3. Most comparisons optimized a component while freezing upstream totals.
4. WAPE aggregation was inconsistent; some headline tables averaged per-pair
   ratios instead of recomputing the ratio of global sums.
5. Smoke/fallback artifacts could look completed and were not centrally
   distinguished from trained evaluations.
6. There was no prospective champion-versus-baseline dashboard across every
   production run and lead time.
7. Production data, assortment, correction and profile contracts changed much
   faster than the static benchmark suite.

## Confidence and limitations

The conclusion that mean 7 currently beats production is strong but not yet a
deployment decision:

- it wins all eight available valid August dates and wins under both observed
  sales and a stricter demand proxy;
- an earlier 18-day historical window also favors mean 7 on three target
  definitions;
- however, eight dates are not enough to cover holidays, planned launches,
  seasonal changes or long stockouts;
- strict demand is an auditable proxy, not directly observed unmet demand;
- the current comparison is evaluated on the production forecast universe, so
  planned assortment introductions outside that universe still need a separate
  coverage test;
- mean 7 needs explicit cold-start and missing-history policy before shadow use.

## Required next controls

Before any new model or allocation rollout, require one frozen evaluation
contract that reports, on identical rows and dates:

1. current production;
2. zero forecast;
3. lag 1 and lag 7;
4. causal means of 3, 7, 14 and 28 calendar days;
5. same-weekday mean;
6. current ML bakery total plus incumbent allocation;
7. direct SKU challenger aggregated upward without renormalization.

For every method report global WAPE, bias and forecast mass at SKU-day,
category-day and bakery-day level, plus metrics by date, lead time, maturity,
assortment status, sales-zero status and product. Do not use average group WAPE
as the headline figure.

The immediate safe action is a read-only prospective mean-7 shadow. A canary or
production switch should wait for additional blocked historical folds and a
prospective period, with explicit fallbacks for cold start, assortment launches,
holidays and insufficient history.

