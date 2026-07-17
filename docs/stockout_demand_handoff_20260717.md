# Handoff: demand-based profiles and stockout reconstruction

Date: 2026-07-17

## Safety and scope

- This work is an offline experiment only.
- Production, production ClickHouse tables, VM services, timers, active runs, and deployed profiles were not changed.
- The experiment uses 10 pilot bakeries: `20, 21, 22, 28, 80, 89, 107, 221, 222, 257`.
- Bakery `16`, which was later added to the production pilot, is intentionally outside this comparison.

## Goal

Estimate censored demand on stockout days before building the hourly SKU profile. Compare:

1. a reference based on good/non-stockout days;
2. a reference based on the SKU's mean share of total bakery sales for the same weekday and hour.

The second method is the main candidate. It anchors missing SKU demand to the bakery's actually realized hourly volume instead of assuming that the bakery has historical days when it fully realized demand.

## Implemented experiment

Core module:

- `src/experiments_v2/stockout_demand_preprocessing.py`

Experiment and analysis scripts:

- `scripts/experiment_stockout_demand_profiles.py`
- `scripts/analyze_stockout_inventory_balance.py`
- `scripts/analyze_inventory_stockout_hourly.py`
- `scripts/backtest_pseudo_stockout_reconstruction.py`
- `scripts/build_demand_adjusted_profile_experiment.py`
- `scripts/build_long_demand_adjusted_profile_experiment.py`

Tests:

- `tests/test_stockout_demand_preprocessing.py`

## Stock balance finding

For the local prepared March dataset, `Выпуск` already includes opening stock. The balance that reproduces closing inventory is therefore:

```text
available = Выпуск + incoming_moves - outgoing_moves
expected_closing = available - Продано
```

Do not add `stock_lag1` again for this particular source. With this interpretation, expected closing inventory is within one unit of the source closing inventory for 97.08% of rows.

Across 9,889 SKU-days:

- simplified `sold / produced >= 0.90`: 6,267 stockout flags;
- inventory-based flags: 5,152;
- both: 5,032;
- simplified-only false positives: 1,235;
- inventory-only: 120, mostly explained by outgoing transfers;
- movements changed the flag for 120 of 378 rows with movements.

## Reliable stockout subset

Temporal evidence is added to the inventory flag: the SKU disappears before the bakery stops actively selling other products.

- inventory-stockout days: 5,152;
- SKU ended at least two hours early: 1,281;
- bakery had at least 50 later sales after SKU disappearance: 3,726;
- strong temporal stockouts: 1,230;
- reliable strong stockouts after agreement between hourly and daily sources: 986.

Estimated missing demand on the reliable strong subset using bakery share: 2,691.9 units.

## Reconstruction policy

The bakery-share reference is:

```text
mean_share(bakery, SKU, weekday, hour)
    = mean(SKU hourly sales / all bakery hourly sales)
```

The denominator must include every product sold by the bakery. Do not normalize shares inside the selected 40-SKU experiment subset; that earlier mistake produced a false large overforecast signal and has been fixed.

Only reliable strong stockouts are corrected. For products with normal daily sales `<= 10`, cap added demand at:

```text
max(4 units, 0.5 * normal_daily_sales)
```

Higher-volume products use uncapped bakery-share reconstruction.

## Pseudo-stockout backtest

Train reference: 2026-03-01 through 2026-03-21. Holdout: 2026-03-22 through 2026-03-29. Known tail sales on reliable non-stockout days were hidden for two, three, or four hours.

Bakery share consistently beat the good-day reference. Uncapped bakery-share results:

| Hidden tail | Recovery | Bias | WAPE |
| --- | ---: | ---: | ---: |
| 2 hours | 76.6% | -23.4% | 64.7% |
| 3 hours | 78.7% | -21.3% | 52.5% |
| 4 hours | 81.0% | -19.0% | 47.1% |

The selected low-volume cap improved WAPE with a small recovery loss:

| Hidden tail | Recovery | WAPE |
| --- | ---: | ---: |
| 2 hours | 74.0% | 62.36% |
| 3 hours | 76.3% | 50.48% |
| 4 hours | 78.6% | 45.40% |

The method remains conservative: it underpredicts roughly two thirds of pseudo-stockout cases and overpredicts roughly one third.

## Profile comparison

A three-week profile is too sparse: only about three observations per weekday cell, with a maximum share change of 0.346. It should not be used for a conclusion.

For a long profile from 2025-06-01 through 2026-03-21 (293 days), only March 2026 could be inventory-corrected with the current local source. The correction added 1,187.59 units across 1,340 hours and 491 strong training stockouts.

The long demand profile is stable:

- changed profile rows: 1,181;
- mean absolute share delta: 0.0000223;
- p99 absolute delta: 0.000665;
- maximum absolute delta: 0.01786.

Normal holdout weighted share MAE is essentially unchanged:

- sales profile: 0.01167475;
- demand profile: 0.01168134.

Pseudo-stockout recovery improves slightly while WAPE becomes only marginally worse. For a three-hour hidden tail over the full long profile:

- recovery: 84.33% -> 84.58%;
- WAPE: 48.55% -> 48.59%.

Interpretation: the idea works directionally and does not cause runaway overforecast. The measured effect is tiny because only one corrected month is diluted across 293 profile days.

## Recommended next step

Extend inventory-based labels to at least three to six months, then rerun the same long-profile comparison. First check whether the existing local sources can safely extend coverage from 2026-01-31 through 2026-03-29:

1. calculate the true minimum/maximum dates in `moves_clickhouse_2025-01-15_2026-05-12.csv` (do not infer coverage from its first row);
2. verify whether January/February `Выпуск` has the same opening-stock semantics as March;
3. verify hourly-vs-daily sales agreement and keep the `<= 1` unit reliability filter;
4. rebuild corrections and compare 28/42/56/84/120/long history windows;
5. only after a meaningful offline effect is established, run a model-training comparison on the same 10 bakeries.

Do not iterate forecasts against their own outputs at this stage. Iterative correction becomes informative only after bakeries begin baking from the forecast and realized supply responds to it.

## Local-only data and reports

These inputs are not committed to Git and must be copied separately or regenerated on another device:

- `data/raw/sales_stg_2025_2026.csv` (about 14 GB; full hourly source);
- `data/raw/moves_clickhouse_2025-01-15_2026-05-12.csv`;
- `data/processed/preprocessed_data_merged.csv`.

Reports are ignored by `.gitignore`. Relevant local directories:

- `reports/stockout_inventory_balance_10/`
- `reports/inventory_stockout_hourly_10/`
- `reports/pseudo_stockout_backtest_10/`
- `reports/demand_adjusted_profile_10/`
- `reports/long_demand_adjusted_profile_10/`

The handoff contains the decision-critical results, so copying reports is optional. Copy them if row-level case inspection must continue on the second device.

## Verification

Run from the repository root:

```powershell
python -m pytest tests/test_stockout_demand_preprocessing.py -q
ruff check src/experiments_v2/stockout_demand_preprocessing.py tests/test_stockout_demand_preprocessing.py scripts/analyze_inventory_stockout_hourly.py scripts/analyze_stockout_inventory_balance.py scripts/backtest_pseudo_stockout_reconstruction.py scripts/build_demand_adjusted_profile_experiment.py scripts/build_long_demand_adjusted_profile_experiment.py scripts/experiment_stockout_demand_profiles.py --select=E,F,W
```

## Resume on another device

```powershell
git clone https://github.com/Almazziastinov/demand-forecasting-model.git
cd demand-forecasting-model
git pull origin master
```

Then restore/regenerate the local data above, read this document and `docs/ops/CURRENT_STATE.md`, and continue with the recommended next step. Production must remain untouched unless separately and explicitly authorized.
