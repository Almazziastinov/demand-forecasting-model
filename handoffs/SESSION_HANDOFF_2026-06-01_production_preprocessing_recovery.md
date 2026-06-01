# Session Handoff - 2026-06-01 - Production Preprocessing Recovery

## Why this handoff exists

Earlier handoffs are stale. The current source of truth is the git state plus
local generated artifacts inspected on `2026-06-01`.

Repository state at inspection time:

- branch: `master`
- local `master` matches `origin/master`
- latest commit: `62a8bc0 feat: add production preprocessing experiments`
- tracked working tree: clean
- `git status` may warn about `codex_tmp/` permission denied, but no tracked
  files are modified, staged, deleted, or locally ahead of origin.

## Latest committed code state

The latest meaningful work is already committed in:

```text
62a8bc0 feat: add production preprocessing experiments
```

Important committed files from that commit:

- `src/experiments_v2/PRODUCTION_PREPROCESSING_PLAN.md`
- `src/experiments_v2/sales_cleaning.py`
- `src/experiments_v2/planning_metrics.py`
- `scripts/add_rolling_quantile_cap.py`
- `src/analysis/audit_sales_cleaning.py`
- `src/experiments_v2/78_bakery_target_cleaning/run.py`
- `src/experiments_v2/79_structural_top_down_correction/run.py`
- `src/experiments_v2/80_event_features_lgbm_top_down/run.py`
- `src/experiments_v2/apply_bakery_profiles.py`
- `src/experiments_v2/bakery_day_forecast.py`
- `src/experiments_v2/build_bakery_daily_dataset.py`
- `src/experiments_v2/build_bakery_hour_profile.py`
- `src/experiments_v2/build_sku_hour_share_profile.py`
- related tests under `tests/`

## Current modeling decision

The selected production direction is:

```text
LGBM bakery-level top-down with enriched event features -> SKU/hour split
```

The active backbone remains:

1. forecast bakery-level daily demand with one LGBM model;
2. split bakery-day to bakery-hour using bakery hour profiles;
3. split bakery-hour to SKU-hour using SKU hour-share profiles;
4. evaluate with planning metrics, not only MAE.

Residual correction / ensemble architecture is not the selected production
direction right now. Experiment 79 showed that a global correction layer over
the current LGBM top-down baseline worsened MAE, WMAPE, and material-error
share.

## Production preprocessing direction

The production plan now separates factual sales from base-model target:

```text
observed_sales -> base_capped_sales
```

Important rule:

- `observed_sales` remains the factual audit column;
- `base_capped_sales` is only for regular base-model training;
- product moves are not subtracted from sales because check lines do not encode
  those moves as demand context.

The selected target-cleaning recipe is weekday-aware trailing rolling quantile
capping:

- estimate `q05` and `q95` by `bakery_id x dow`;
- use a trailing window of `26` same-weekday observations;
- require `min_periods=8`;
- shift by one row so today's value never defines its own cap;
- fall back to expanding quantile when history is thin;
- preserve contextual high outliers by default for a later correction layer.

Reusable implementation:

- `src/experiments_v2/sales_cleaning.py`
- `src/analysis/audit_sales_cleaning.py`

## Planning metrics

MAE is still diagnostic, but production comparisons should also report:

- WMAPE;
- aggregate bias by planning level;
- material-error share where both are true:
  - absolute error > `50`;
  - relative absolute error > `20%`;
- direction of material errors:
  - underforecast risk;
  - overforecast risk.

Reusable implementation:

- `src/experiments_v2/planning_metrics.py`

## Experiment 78 - Bakery target cleaning

Location:

- `src/experiments_v2/78_bakery_target_cleaning/`

Generated outputs are ignored by git but exist locally in:

- `run_7d_v2/`
- `run_14d_v2/`
- `run_30d_v2/`

### 7-day result

Best model:

- `rolling_quantile_capped_target_lgbm`

Metrics:

- MAE: `142.654168`
- WMAPE: `15.977115`
- bias: `-66.258387`

Baseline `raw_target_lgbm`:

- MAE: `150.6829`
- WMAPE: `16.876325`
- bias: `-75.690298`

Interpretation:

- rolling quantile capping materially improves short-horizon quality.

### 14-day result

Best aggregate model:

- `quantile_capped_target_lgbm_benchmark`

Metrics:

- MAE: `118.212493`
- WMAPE: `12.589204`
- bias: `-16.27558`

Rolling quantile target:

- MAE: `118.636878`
- WMAPE: `12.6344`
- bias: `-20.744034`

Interpretation:

- static quantile benchmark is slightly best on aggregate, but rolling quantile
  remains very close.

### 30-day result

Best aggregate model:

- `quantile_capped_target_lgbm_benchmark`

Metrics:

- MAE: `96.425334`
- WMAPE: `9.868779`
- bias: `-0.889987`

Rolling quantile target:

- MAE: `97.568983`
- WMAPE: `9.985827`
- bias: `-0.466703`

Interpretation:

- static quantile wins 30-day MAE slightly;
- rolling quantile has very small bias and is preferred in the production plan
  because it adapts to trend and slow seasonality.

## Experiment 79 - Structural top-down correction

Location:

- `src/experiments_v2/79_structural_top_down_correction/`

Generated outputs are ignored by git but exist locally in:

- `run_7d_v2/`
- `run_14d_v2/`
- `run_30d_v2/`

### Main result

The correction layer is not currently useful.

30-day metrics:

- `lgbm_top_down`
  - MAE: `96.733781`
  - WMAPE: `9.912769`
  - large error share: `0.116639`
- `lgbm_top_down_plus_correction_lgbm`
  - MAE: `99.822973`
  - WMAPE: `10.229333`
  - large error share: `0.125115`
- `structural_plus_correction_lgbm`
  - MAE: `104.56409`
  - WMAPE: `10.715178`
  - large error share: `0.135618`
- `structural_top_down`
  - MAE: `114.178871`
  - WMAPE: `11.70045`
  - large error share: `0.161415`

Interpretation:

- the single LGBM top-down model remains the best of these options;
- correction layers should stay in research backlog only, and should be gated
  before any future production use.

## Experiment 80 - Enriched event features

Location:

- `src/experiments_v2/80_event_features_lgbm_top_down/`

Generated outputs are ignored by git but exist locally in:

- `run_7d/`
- `run_14d/`
- `run_30d/`

The experiment compares:

- `lgbm_top_down_base_events`
- `lgbm_top_down_enriched_events`

Enriched features add explicit event/payday context:

- holiday name;
- event-window type;
- event distance bin;
- current / nearest event city;
- event x weekday interaction;
- pre/post event flags;
- payday distance and payday-window features.

### 7-day result

- enriched events:
  - MAE: `139.474559`
  - WMAPE: `15.637319`
  - bias pct: `-7.042932`
- base events:
  - MAE: `150.246264`
  - WMAPE: `16.844998`
  - bias pct: `-8.457362`

### 14-day result

- enriched events:
  - MAE: `114.042928`
  - WMAPE: `12.162006`
  - bias pct: `-1.358046`
- base events:
  - MAE: `119.219408`
  - WMAPE: `12.714047`
  - bias pct: `-2.054066`

### 30-day result

- enriched events:
  - MAE: `96.736796`
  - WMAPE: `9.913078`
  - bias pct: `-0.499318`
- base events:
  - MAE: `97.571711`
  - WMAPE: `9.998636`
  - bias pct: `-0.50786`

Interpretation:

- enriched event/payday features improve 7-day and 14-day quality materially;
- 30-day improvement is smaller but still positive on MAE/WMAPE;
- this supports strengthening the single top-down LGBM with calendar/event
  context instead of adding a correction model.

## Exp80 smoke production artifacts

Local ignored outputs:

- `reports/bakery_day_model_exp80_smoke_summary.json`
- `reports/bakery_day_model_exp80_smoke_holdout_predictions.csv`
- `reports/bakery_day_model_exp80_smoke_bias_by_bakery.csv`
- `models/bakery_day_model_exp80_smoke.joblib`
- `models/bakery_day_meta_exp80_smoke.joblib`

Smoke summary:

- rows_total: `84238`
- rows_train: `78811`
- rows_test: `5427`
- bakeries: `212`
- date_min: `2025-01-15`
- date_max: `2026-05-12`
- test_start: `2026-04-13`
- feature_count: `56`
- MAE: `96.736796`
- WMAPE: `9.913078`
- bias: `-4.872596`

Largest underforecast bakery in smoke summary:

- `bakery_id = 245`
- bias: `106.552669`
- MAE: `176.116767`

This is the same bakery that was already notable in older bakery-day work.

## Exp80 allocation artifacts

Local ignored outputs:

- `data/processed/bakery_day_forecast_exp80.csv`
- `data/processed/sku_day_forecast_exp80.csv`
- `data/processed/sku_hour_forecast_exp80.csv`
- `data/processed/apply_bakery_profiles_summary_exp80.json`

Allocation summary:

- bakery_day_rows: `2968`
- bakery_hour_rows: `47574`
- sku_hour_rows: `2983936`
- sku_day_rows: `411548`
- dates: `14`
- bakeries: `212`
- products: `987`
- bakery_forecast_total: `2588929.757193`
- sku_forecast_total: `2588929.757193`
- allocation_ratio: `1.0`
- date_min: `2026-05-13`
- date_max: `2026-05-26`

Source mix:

- `exact`: `83.509%` rows, `95.929%` forecast share
- `bakery_hour_fallback_thin`: `16.0804%` rows, `3.2208%` forecast share
- `bakery_hour_fallback`: `0.4106%` rows, `0.8504%` forecast share

## SKU share profile local rebuild

Local ignored outputs updated on `2026-05-29`:

- `data/processed/sku_hour_share_profile.csv`
- `data/processed/sku_hour_share_profile_daily.csv`
- `data/processed/sku_hour_share_profile_summary.json`

These are generated artifacts and are not committed.

## What is not in git but matters

The current full computed state depends on ignored local artifacts:

- `data/raw/sales_hrs_all_clickhouse.csv`
- `data/processed/bakery_daily_sales.csv`
- `data/processed/bakery_hour_profile.csv`
- `data/processed/sku_hour_share_profile*.csv`
- exp78/79/80 generated run folders
- exp80 smoke model and report files
- exp80 forecast allocation CSVs

If restoring on another machine, code can be restored with git, but generated
state must be copied from this workspace or rebuilt from raw ClickHouse exports.

## Recommended next step

Continue from production preprocessing, not from the old Kazan normative
handoff.

Practical next work:

1. decide whether to train production bakery-day model on:
   - rolling quantile capped target; or
   - raw target with enriched event features only;
2. integrate enriched event features into the regular `bakery_day_forecast.py`
   production path if not already fully wired for train/forecast;
3. rebuild future bakery-day forecast with the chosen model;
4. run allocation and verify:
   - `allocation_ratio = 1.0`;
   - fallback share is acceptable;
   - planning metrics by `city x category` and `city x product`;
5. only after base layer is stable, revisit correction/event uplift as an
   auditable second layer.

## Commands useful for resuming

Check git state:

```powershell
git status -sb --untracked-files=no
git log --oneline -8
```

Run relevant tests:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_sales_cleaning.py tests/test_planning_metrics.py tests/test_bakery_target_cleaning_exp78.py tests/test_structural_top_down_correction_exp79.py tests/test_event_features_lgbm_top_down_exp80.py -v
```

Inspect exp80 metrics:

```powershell
Get-Content -Raw src\experiments_v2\80_event_features_lgbm_top_down\run_7d\metrics.json
Get-Content -Raw src\experiments_v2\80_event_features_lgbm_top_down\run_14d\metrics.json
Get-Content -Raw src\experiments_v2\80_event_features_lgbm_top_down\run_30d\metrics.json
```
