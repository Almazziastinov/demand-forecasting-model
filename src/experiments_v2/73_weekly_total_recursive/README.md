# Experiment 73: Weekly Total Recursive

Strict recursive bakery-level backtest for the decomposition approach:

1. forecast weekly bakery total
2. allocate weekly total to days using recent weekday shares

Compared against:

- `recursive_daily_baseline` — current bakery daily LightGBM used recursively day by day
- `weekly_total_daily_share_recursive` — weekly total model + weekday-share allocation

Input:

- `data/processed/bakery_daily_sales.csv`

Default evaluation:

- holdout = last `28` days
- recursive prediction over the whole holdout window
