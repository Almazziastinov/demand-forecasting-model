# Session handoff - pilot SKU corrections

## Outcome

Two category-neutral SKU correction layers were implemented and deployed to
the daily Bitrix24 pilot baking-plan publisher for the 10 pilot bakeries.

1. Forecast cold start for products 11573 and 11574 uses an own-sales EWMA
   floor with alpha 0.90 and no missed-demand input.
2. Mature bakery/SKU pairs use persistent-bias detection and adaptive
   geometrically smoothed multipliers.

Both mechanisms use only data strictly before the forecast date. Cold-start
pairs leave that mechanism after 13 positive-forecast days and can enter the
mature registry starting with day 14.

## Important finding

The initial unbounded mature correction exposed a false positive for product
4944 at bakery 257. It had 49 sales days but only one positive forecast day.
Historical zero forecasts produced a false -97.5% bias and coefficient 3.03.

The mature registry now requires at least 14 positive-forecast days. After this
guard, the mature-only WAPE result is 25.1106% to 24.8957%.

## Combined evidence

Rolling 28-day walk-forward through 2026-07-28:

- total WAPE: 25.7551% to 25.0720% (-0.6831 pp);
- underforecast quantity: -572.04 units;
- overforecast quantity: -572.04 units;
- new-SKU WAPE: 95.0597% to 57.4101%;
- exact bakery/category forecast totals preserved.

## Production state

- Server: `82bb03a8-c356-4225-97a4-a1540cdc29e6`.
- Publisher: `/opt/scripts/publish_pilot_forecast.py`.
- Added modules:
  `/opt/src/experiments_v2/sku_cold_start.py` and
  `/opt/src/experiments_v2/sku_systematic_correction.py`.
- Timer: enabled and active, `03:00 UTC` / `06:00 MSK`.
- Next scheduled run after deployment: 2026-07-30 03:00 UTC.
- Remote 2026-07-30 dry-run: 18 cold-start floors, 426 changed rows,
  535 output rows, 10 bakeries, valid 28,739-byte workbook, no chat send.
- Rollback:
  `/opt/scripts/publish_pilot_forecast.py.backup_20260729_sku_corrections`.

The correction affects only the published baking-plan workbook. It does not
rewrite active production forecast snapshots.

## Key files

- `src/experiments_v2/sku_cold_start.py`
- `src/experiments_v2/sku_systematic_correction.py`
- `scripts/backtest_sku_systematic_correction.py`
- `scripts/backtest_combined_sku_corrections.py`
- `scripts/publish_pilot_forecast.py`
- `tests/test_sku_cold_start.py`
- `tests/test_sku_systematic_correction.py`
- `tests/test_publish_pilot_forecast.py`
- `docs/pilot_sku_corrections_20260729.md`
- `docs/ops/CURRENT_STATE.md`

## Next checks

1. Confirm the 2026-07-30 timer run completed and posted one workbook.
2. Compare the first production plan with the dry-run counts and spot-check
   products 11573/11574.
3. Monitor daily registry size, maximum multiplier, and WAPE after facts close.
4. Develop the trend/seasonality/regime-aware SKU model in shadow; do not
   silently replace these temporary correction layers.
