# Experiment 72: Bakery Regime Shift

Bakery-level backtest focused on one failure mode from experiment 71:

- the forecast recognizes weekly seasonality,
- but collapses amplitude and pulls the series toward the mean,
- especially after a bakery-level upward or downward level shift.

This experiment compares:

1. `baseline_global_lgbm`
2. `normalized_target_lgbm`
3. `fast_seasonal_lgbm`
4. `weekly_total_daily_share`
5. `global_local_hybrid`

Input:

- `data/processed/bakery_daily_sales.csv`

Default evaluation:

- holdout = last `30` days
- outputs saved into this folder
