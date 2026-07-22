# Assortment coverage guard historical backtest — 2026-07-22

## Question

Would the local pre-publication guard have detected the 47 known clear-stockout
SKU-days that had no forecast, without treating one-day/new products as hard
failures?

## Method

- Replayed 43 historical bakery/date/run contexts represented by the 47 cases.
- Used the local raw pilot sales export from 2026-05-01 through 2026-07-19.
- For every context, aggregated positive sales in the prior seven days.
- Selected only the latest city assortment batch whose `valid_from` applied to
  the forecast date and whose `loaded_at` was no later than the historical run.
- Applied the production-candidate threshold: at least two selling days and at
  least two units.
- Used sales on the forecast date and the following seven days as evidence that
  an omitted SKU was still active. Absence of future sales is left ambiguous;
  it is not labelled a false positive without independent assortment intent.

ClickHouse access was read-only. No production state was changed.

## Results

- Known cases caught: **47/47 (100%)**.
- Every known case had at least five prior selling days and at least nine prior
  units; thresholds `1/1`, `2/2`, and `3/3` all retain 47/47 recall.
- Historical contexts: 43; contexts with no applicable historical assortment
  batch: 17. These are mainly dates before the first versioned batch on
  2026-06-18 and should fail closed in a historical replay.
- Total blocking context rows: 4,174 across 978 distinct bakery/SKU pairs.
  - 2,855 rows occurred with no applicable historical batch.
  - 1,319 rows occurred despite an applicable batch.
- 3,198 blocking rows sold on the forecast date.
- 4,012 blocking rows (96.1%) sold on the forecast date or in the next seven
  days.
- Of 3,913 rows with a complete seven-day future observation window, only 108
  had no same-day or subsequent sale. These remain ambiguous rather than proven
  false positives.
- 553 missing rows were classified as one-day/low-volume diagnostics and were
  not blocked. Of them, 111 sold on the forecast date and 352 sold on the
  forecast date or in the next seven days. This is the explicit cold-start
  trade-off of the conservative threshold.

## Decision

Keep the current `2 selling days / 2 units` blocking threshold. It catches all
known failures, remains conservative for one-day introductions, and the large
majority of historical blockers have direct subsequent-sales evidence. Do not
interpret the 4,174 repeated context rows as 4,174 independent incidents.

The historical evidence supports the guard as a publication safety check, but
the code remains local and is not deployed to the production writer.

## Artifacts

- Runner: `scripts/backtest_assortment_coverage_guard.py`
- Unit tests: `tests/test_backtest_assortment_coverage_guard.py`
- Local detailed output: `reports/assortment_coverage_guard_backtest/`

