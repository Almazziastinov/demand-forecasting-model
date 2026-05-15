# Session Handoff 2026-05-15

## Scope

- Continued bakery-day forecasting work from previous handoffs.
- Extended holiday handling from simple `is_holiday / is_pre_holiday / is_post_holiday`
  toward cluster-based event features.
- Added non-recursive `exp74` 30-day comparisons and notebook helpers.
- Investigated the `2026-04-21` network spike and built a raw duplicate-audit tool.

## Main Changes

### Holiday / event modeling

- Expanded holiday calendar in
  `src/experiments_v2/bakery_day_forecast.py`.
- Added event-cluster features:
  - `current_event_cluster`
  - `prev_event_cluster`
  - `next_event_cluster`
  - `days_since_prev_event`
  - `days_to_next_event`
  - `is_near_event_window`
- Unified train and recursive future-row feature generation so holiday/event logic is
  applied consistently in both paths.
- Fixed LightGBM categorical alignment in
  `src/experiments_v2/73_weekly_total_recursive/run.py` for recursive inference.

### Analysis scripts

- Added `src/analysis/analyze_holiday_effects.py`
- Added `src/analysis/analyze_holiday_behavior.py`
- Added `src/analysis/audit_raw_sales_duplicates.py`

### Experiment 74

- Added `src/experiments_v2/74_bakery_non_recursive/run.py`
- Saved multiple `exp74` runs:
  - default 7-day non-recursive
  - `30d`
  - `30d_no_post_holiday`
  - `30d_targeted_holiday_overrides`
  - `30d_cluster_features`

### Notebook helpers

- Added `notebooks/bakery_day_backtest_latest_models.py`
- Added `notebooks/bakery_day_backtest_exp74.py`
- Updated `notebooks/bakery_day_backtest.ipynb`
  with:
  - latest `exp73` section
  - `exp74` 30-day plots
  - cross-run comparison for `30d`, `30d_targeted_holiday_overrides`,
    `30d_cluster_features`

## Key Results

### Best current 30-day non-recursive baseline

From `src/experiments_v2/74_bakery_non_recursive/30d_cluster_features/summary_by_model.csv`:

- `daily_baseline_non_recursive`
  - `avg_mae = 111.349453`
  - `avg_wmape = 10.632754`
  - `avg_bias = 1.067049`
  - `win_count = 165`

This is better than:

- plain `30d` baseline: `avg_mae = 121.780982`
- targeted holiday overrides: `avg_mae = 120.144175`

### Test status

Verified with:

- `.venv\Scripts\python.exe -m pytest tests/test_bakery_day_forecast.py tests/test_weekly_total_recursive.py -v`

Result:

- `12 passed`

## Duplicate Audit Findings

Raw audit was run against `data/raw/sales_hrs_all_clickhouse.csv`.

### 2026-04-14

- `raw_quantity_sum = 188387.273`
- `strict_duplicate_quantity_gap = 16783.473`
- `strict_gap_pct_of_qty = 8.91%`

### 2026-04-21

- `raw_quantity_sum = 233627.323`
- `strict_duplicate_quantity_gap = 23620.559`
- `strict_gap_pct_of_qty = 10.11%`

### 2026-04-28

- `raw_quantity_sum = 213001.971`
- `strict_duplicate_quantity_gap = 23099.954`
- `strict_gap_pct_of_qty = 10.84%`

### Interpretation

- Duplicate-like patterns are not unique to `2026-04-21`.
- `2026-04-21` is elevated, but the raw clickhouse layer shows a broader
  systematic duplicate-risk problem around `9-11%`.
- The next logical step is raw-layer dedup design before rebuilding
  `bakery_daily_sales.csv`.

## Recommended Next Step

1. Define a defensible raw dedup key for clickhouse check lines.
2. Build a deduped bakery-day dataset.
3. Re-check `2026-04-14 / 2026-04-21 / 2026-04-28` after dedup.
