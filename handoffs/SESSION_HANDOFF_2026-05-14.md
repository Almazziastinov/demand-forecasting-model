# Session Handoff — 2026-05-14

## Context

This session continued the `experiments_v2` bakery-driven forecasting track.

Pipeline status at end of session:

1. `bakery_day_forecast`
2. `apply_bakery_profiles`
3. `bakery_hour_profile`
4. `sku_hour_share_profile`
5. `sku_hour_share_profile_smoothed`

Core future allocation path is working end-to-end and preserves mass.

## Key Completed Work

### 1. Allocation / application layer

Implemented and stabilized:

- `src/experiments_v2/apply_bakery_profiles.py`
- `tests/test_apply_bakery_profiles.py`

Important fixes:

- switched to chunked / streaming SKU-profile processing to avoid OOM on real data
- reduced output width for large hourly files
- added fallback for missing `bakery_id x dow` hour-profile rows
- added fallback for missing `bakery_id x dow x hour` SKU-share rows
- verified future allocation mass preservation

Validated future allocation summary:

- `allocation_ratio = 1.0`

Main future outputs already exist:

- `data/processed/bakery_day_forecast.csv`
- `data/processed/sku_day_forecast_future_smoothed_bias_adj.csv`
- `data/processed/sku_hour_forecast_future_smoothed_bias_adj.csv`

### 2. Bakery-level forecasting module

Implemented:

- `src/experiments_v2/bakery_day_forecast.py`
- `tests/test_bakery_day_forecast.py`

Capabilities:

- `--mode train`
- `--mode forecast`
- holdout evaluation
- recursive future forecast
- bias reporting
- optional bias correction

Artifacts produced:

- `models/bakery_day_model.joblib`
- `models/bakery_day_meta.joblib`
- `reports/bakery_day_model_summary.json`
- `reports/bakery_day_model_holdout_predictions.csv`
- `reports/bakery_day_model_bias_by_bakery.csv`

### 3. Backtest notebook

Added:

- `notebooks/bakery_day_backtest.ipynb`

Purpose:

- inspect bakery-level holdout predictions
- compare base forecast vs bias-adjusted forecast
- plot network/day and bakery/day trajectories

### 4. Experiment 72 — regime shift alternatives

Implemented:

- `src/experiments_v2/bakery_regime_shift_common.py`
- `src/experiments_v2/72_bakery_regime_shift/run.py`
- `src/experiments_v2/72_bakery_regime_shift/README.md`
- `tests/test_bakery_regime_shift_common.py`

Compared models:

- `baseline_global_lgbm`
- `normalized_target_lgbm`
- `fast_seasonal_lgbm`
- `weekly_total_daily_share`
- `global_local_hybrid`

Result:

- baseline remained strongest overall
- `fast_seasonal_lgbm` came closest but did not qualitatively fix amplitude collapse

### 5. Experiment 73 — strict recursive weekly / daily comparison

Implemented:

- `src/experiments_v2/73_weekly_total_recursive/run.py`
- `src/experiments_v2/73_weekly_total_recursive/README.md`
- `tests/test_weekly_total_recursive.py`

This experiment now includes:

- `seasonal_naive_lag7_recursive`
- `repeat_last_week_recursive`
- `recursive_daily_baseline`
- `heuristic_blend_recursive`
- `weekly_total_daily_share_recursive`

Important work done inside `exp73`:

- fixed recursive weekly logic around partial first/last holdout weeks
- restricted weekly training history to complete weeks only
- added adaptive weekday-share allocation
- added naive baselines for honest lower-bound comparison
- added `heuristic_blend_recursive` that softly mixes global ML forecast with lag-based heuristics

## Latest `exp73` Result

Command used:

```powershell
.venv\Scripts\python.exe src\experiments_v2\73_weekly_total_recursive\run.py --dataset-path data\processed\bakery_daily_sales.csv --test-days 14 --min-train-rows 90 --recent-weeks 4
```

Current summary:

- `seasonal_naive_lag7_recursive`: `avg_mae 173.205512`
- `repeat_last_week_recursive`: `avg_mae 172.776308`
- `recursive_daily_baseline`: `avg_mae 147.425447`
- `heuristic_blend_recursive`: `avg_mae 145.333313`
- `weekly_total_daily_share_recursive`: `avg_mae 176.157009`

Main takeaway:

- weekly decomposition is not the main promising direction right now
- the best current direction is the **soft fallback blend**
- `heuristic_blend_recursive` is now better than pure global baseline on the 14-day strict recursive backtest

## Bakery 245 Finding

Important bakery investigated in detail:

- `bakery_id = 245`
- `Яблоневая, 1д Габишево`

Observed issue:

- baseline ML forecast smooths amplitude too aggressively
- `lag7` tracks this bakery much better on short horizon

Latest metrics on bakery `245`:

- `baseline mae = 296.679`
- `heuristic_blend mae = 215.945`
- `lag7 naive mae = 175.674`

Interpretation:

- blend already improves materially over baseline
- but still does not reach pure `lag7` quality for this bakery
- this supports the idea of a **soft lag trust mechanism**, not hard switching

## Recommended Next Step

Focus only on short horizon bakery forecasting.

Recommended direction for next session:

1. continue tuning `heuristic_blend_recursive`
2. keep global ML forecast as main anchor
3. selectively increase trust in `lag7` only under narrow conditions:
   - local uptrend
   - stable recent week
   - clear ML underprediction vs lag
4. do **not** spend more time on weekly decomposition unless a strong new hypothesis appears

Practical target:

- improve bakery `245` and similar cases further
- keep global `avg_mae` below pure baseline

## Relevant Files

- `src/experiments_v2/apply_bakery_profiles.py`
- `src/experiments_v2/bakery_day_forecast.py`
- `src/experiments_v2/bakery_regime_shift_common.py`
- `src/experiments_v2/72_bakery_regime_shift/run.py`
- `src/experiments_v2/73_weekly_total_recursive/run.py`
- `tests/test_apply_bakery_profiles.py`
- `tests/test_bakery_day_forecast.py`
- `tests/test_bakery_regime_shift_common.py`
- `tests/test_weekly_total_recursive.py`
- `notebooks/bakery_day_backtest.ipynb`
