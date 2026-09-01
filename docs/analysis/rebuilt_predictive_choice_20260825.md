# Rebuilt predictive-choice SKU allocation — 2026-08-25

## Design corrections

The forecast-conditioned SKU-share challenger was rebuilt rather than copied
from the earlier experiment:

- one explicit production run per forecast date; no `argMax` mixing;
- forecast SKU-universe and incumbent forecast shares are known at inference;
- features use sales strictly before each fold boundary;
- predicted shares are normalized once inside `date × bakery × category`;
- production category totals are preserved exactly; no oracle totals;
- forecast-only bakery-days are excluded from quality metrics as DQ.

Features combine bakery, product, category, weekday, incumbent forecast share,
same-weekday share, causal-trend share and category/quantity scale. The frozen
LightGBM configuration is fitted separately for each fold.

## Frozen folds

- Blocked fold: train through 2026-07-21, test 2026-07-22..2026-08-02.
- Current fold: train through 2026-08-10, test the eight completed
  `base_norm_recent` dates through 2026-08-23.

## Observed-sales results

| Method | Blocked WAPE | Current WAPE |
|---|---:|---:|
| Incumbent | 40.3878% | 56.6228% |
| Causal blend 25% | 40.2107% | 51.6530% |
| Predictive choice | **39.5393%** | **44.4550%** |
| Predictive blend 25% | 39.9743% | 52.1300% |

Predictive choice wins all 12 blocked dates and all eight current dates. It
improves 1,519 of 2,154 blocked bakery-days and 1,223 of 1,406 current
bakery-days. Predictive blend 25% improves more individual bakery-days but
captures much less aggregate WAPE reduction.

On current strict demand, predictive-choice WAPE is 44.1301% versus 56.1277%
for the incumbent. Bias remains identical at -3.3387% because category totals
are conserved.

## Concentration

| Method | Blocked p95 | Current p95 | Current >=30% | Current >=40% |
|---|---:|---:|---:|---:|
| Incumbent | 13.51% | 30.36% | 73 | 10 |
| Predictive choice | 13.56% | **19.28%** | **18** | **0** |

The model preserves the clean blocked-period concentration profile and
substantially reduces the current incident.

## Key diagnostics

- Current SKU 1071 observed-sales WAPE: 60.87% → 22.78%.
- Blocked SKU 1071 WAPE: 23.07% → 22.75%.
- Bakery 29 on 2026-08-23: 90.90% → 65.21%.
- Every evaluated date improves, but bakery 29 remains materially inaccurate.

## Decision

Predictive choice is now the primary shadow candidate. It has a larger and
more consistent blocked gain than causal blend 25%, while materially reducing
current concentration. It is not ready for production because:

1. only 12 blocked dates are available;
2. the current incident is still not fully solved for bakery 29;
3. operational missing-SKU and cold-start coverage need explicit reporting;
4. prospective shadow must verify exact daily conservation after hourly
   scheduling.

Artifacts:

- `scripts/backtest_rebuilt_predictive_choice.py`
- `reports/rebuilt_predictive_choice_20260825/summary.json`
- `reports/rebuilt_predictive_choice_20260825/predictions.parquet`
- `.codex_tmp/predictive_choice_rebuild_20260825/selected_runs.csv`

Production writes: none.
