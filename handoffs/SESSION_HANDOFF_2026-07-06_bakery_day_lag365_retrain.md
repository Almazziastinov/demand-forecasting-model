# Session Handoff — 2026-07-06 — Bakery-Day Model Retrain with lag365

## Scope

- Investigated systematic overforecast in the bakery-day model (bias ≈ −2.7%,
  model predicts more than actual) during summer seasonal transition.
- Confirmed root cause: lag30 / roll_mean30 reflect higher May values when
  predicting June; model has no direct YoY anchor.
- Ran YoY experiments (exp 82 YoY, exp 83) to find the best YoY signal.
- Added `bakery_sales_lag365` to the production model, retrained, updated
  bias table, deployed code + model to VM.
- Extended dataset history window from 7 months to 13 months so lag365
  is populated on the VM.

## Why Overforecast Is Desirable Here

Overforecast means "bake more than sold" which is acceptable (stale goes
to 70%-price resale). Underforecast means missed sales — the more costly
outcome. The model already overforecasts on 160/188 bakeries; 28 bakeries
still underforecast and are corrected via `bakery_day_bias.json`.

## Seasonal Bias Analysis

Monthly average bakery sales confirmed a recurring summer dip:

| Transition | 2025 | 2026 |
|---|---|---|
| May → Jun | −4.6% | −2.2% |
| Jun → Jul | −2.5% | −6.1% |

Holdout bias by DOW (Jun 2026): uniform overforecast, no structural DOW issue.
Worst overforecast: weekends (−38 units/day). Lowest: Monday (−1).

## Experiment Results

### Exp 82 YoY (CV, 3 folds)

Post-processing YoY correction factor (compute bakery CF from same month
last year, apply multiplicatively): all variants WORSE (+0.06..+0.09 MAE).
Model lags already capture the local trend; external CF doubles the correction.

`bakery_sales_lag365` as model feature: consistent improvement across all folds.

| Fold | Baseline MAE | +lag365 MAE | Delta |
|---|---|---|---|
| fold1_apr | 1.1723 | 1.1690 | −0.0033 |
| fold2_may | 1.1684 | 1.1651 | −0.0033 |
| fold3_jun | 1.1680 | 1.1647 | −0.0033 |

Feature importance (gain): 2.3–3.5%.

### Exp 83 — Additional YoY Features (lag364, roll_mean4w_yoy, yoy_month_mean)

All three variants NEGATIVE in aggregate (mean delta +0.53..+0.83):
- Coverage only 27–29% (dataset starts Jan 2025, so these features are NaN
  for most Jun 2025 training rows).
- Fold3 (Jun test) showed marginal improvement for lag364 (−0.59) but
  fold1–2 were worse.
- **Revisit in autumn 2026** when dataset covers ≥18 months and coverage
  for these features reaches ~65%+.

## Code Changes

### `src/experiments_v2/build_bakery_daily_dataset.py`

```python
# before
for lag in [1, 2, 3, 7, 14, 30]:
# after
for lag in [1, 2, 3, 7, 14, 30, 365]:
```

### `src/experiments_v2/bakery_day_forecast.py` (3 locations)

1. `BASE_FEATURES` list — added `"bakery_sales_lag365"` after `"bakery_sales_lag30"`
2. `add_derived_features()` `numeric_fill_cols` — added `"bakery_sales_lag365"`
3. `build_future_feature_rows()` — added `"bakery_sales_lag365": _safe_tail_value(sales, 365)`

### `pipelines/forecast_publish/production_dataset_refresh.py`

```python
# before
DEFAULT_HISTORY_START_DATE = "2025-12-01"
# after
DEFAULT_HISTORY_START_DATE = "2025-06-01"
```

Required so that the VM's nightly dataset refresh exports ≥13 months of
history, giving lag365 meaningful coverage for current forecast dates.

## Retrain Results

Dataset: `data/processed/stg_daily_v1/bakery_daily_sales.csv`
(stg_check_lines, Jan 2025–Jul 2026, 94 456 rows, 219 bakeries, 70 features)

Holdout: Jun 6 – Jul 5 2026 (5 585 rows, 188 bakeries)

| Metric | Value |
|---|---|
| MAE | 67.20 |
| WMAPE | 7.44% |
| Bias (pred − actual) | −22.2 (overforecast, −2.66%) |
| Bakeries overforecast (bias < 0) | 160 / 188 |
| Bakeries underforecast (bias > 0) | 28 / 188 |
| Max underforecast | Вокзальная 1 Курск +5.6% |

`models/bakery_day_bias.json` updated from new holdout (188 bakeries).

## Deploy Status

| Artifact | Status |
|---|---|
| `models/bakery_day_model.joblib` | ✅ SCP'd to VM |
| `models/bakery_day_meta.joblib` | ✅ SCP'd to VM |
| `models/bakery_day_bias.json` | ✅ SCP'd to VM |
| Code (`f828cfc`, `2c38e80`) | ✅ `git pull` on VM |
| Service run today | ⏸ Skipped — run_id conflict |

**Why today's service was skipped:** The nightly timer ran at 03:30 UTC with
the old model (run_id `prod_base_bakery_no_sku_uplift_20260706_h14`), loading
4.8M rows into `sku_forecast_hour_embedded`. Our afternoon redeploy tried the
same run_id; `wait_for_run_deleted` timed out trying to delete those 4.8M rows.
The morning run remains active today (old model, no lag365).

**Tomorrow (2026-07-07):** Nightly timer fires at 03:30 UTC with a fresh
run_id (`20260707`). VM will:
1. Rebuild dataset from ClickHouse starting `2025-06-01` (13 months)
2. `bakery_sales_lag365` will populate at ~50–60% coverage for Jul 2026 rows
3. Run forecast with new model (70 features incl. lag365)
4. Publish `prod_base_bakery_no_sku_uplift_20260707_h14`

## Pending Issues

- `verify_prod_deploy` ends with "no active run found in forecast_runs_embedded"
  even though summary shows `activated=True`. Pre-existing issue; needs
  investigation.
- `forecast-production.service` PermissionError on
  `reports/production_dataset_refresh_summary.json` mentioned in previous
  sessions — may be same root cause as verify failure.
- 28-day `analyze_variants_comparison.py --variants base_raw` produces broken
  numbers (bias% +216.7%) — not root-caused, do not trust.
- Exp 83 YoY features (lag364, roll_mean4w_yoy, yoy_month_mean): revisit
  autumn 2026 when dataset coverage reaches ~65%+.

## Commits

| Hash | Message |
|---|---|
| `f828cfc` | feat: add bakery_sales_lag365 to bakery-day model |
| `2c38e80` | fix: extend dataset history start to 2025-06-01 for lag365 coverage |
