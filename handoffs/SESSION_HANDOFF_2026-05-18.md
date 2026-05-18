# Session Handoff 2026-05-18

## Scope

- Added strict raw sales deduplication for bakery daily dataset rebuild.
- Rebuilt `data/processed/bakery_daily_sales.csv` from raw clickhouse snapshot.
- Rechecked the April 2026 spike dates after dedup.

## Main Changes

### Raw strict dedup

Added reusable helper:

- `src/experiments_v2/raw_sales_dedup.py`

Strict dedup key:

- `check_datetime`
- `bakery_id`
- `product_id`
- `quantity`
- `price`
- `line_amount`
- `cash_event_type`

Important:

- This is strict line-level dedup only.
- Rows are removed only when the full business key above matches.
- Softer duplicate heuristics were not used for deletion.

### Bakery daily dataset rebuild

Updated:

- `src/experiments_v2/build_bakery_daily_dataset.py`

What changed:

- raw snapshot normalization and sales filtering were moved through the new dedup helper
- chunk processing now deduplicates raw sales lines before bakery-day aggregation
- rebuild summary now includes `raw_sales_dedup` metrics
- legacy Russian raw column aliases are still supported

### Tests

Updated / added coverage:

- `tests/test_build_bakery_daily_dataset.py`

Verified:

- strict duplicates are removed
- distinct sales with same timestamp are preserved when business fields differ
- legacy Russian snapshot columns still work

## Full Rebuild Result

Rebuilt from:

- `data/raw/sales_hrs_all_clickhouse.csv`

Output:

- `data/processed/bakery_daily_sales.csv`
- `data/processed/bakery_daily_sales_summary.json`

From `bakery_daily_sales_summary.json`:

- `raw_rows = 64,170,320`
- `deduped_rows = 58,172,714`
- `removed_rows = 5,997,606`
- `raw_quantity_sum = 86,305,456.494`
- `deduped_quantity_sum = 78,567,673.339`
- `removed_quantity_sum = 7,737,783.155`
- `raw_line_amount_sum = 8,234,677,066.85`
- `deduped_line_amount_sum = 7,383,087,777.23`
- `removed_line_amount_sum = 851,589,289.62`

## April Spike Recheck

Saved under:

- `reports/dedup_recheck_2026_04_spikes/`

Key files:

- `dedup_effect_joined.csv`
- `raw_vs_dedup_by_date.csv`
- `summary.json`

### 2026-04-14

- `raw_quantity_sum = 188,387.273`
- `deduped_quantity_sum = 171,604.800`
- `removed_quantity_sum = 16,782.473`
- `removed_qty_pct = 8.9085%`

### 2026-04-21

- `raw_quantity_sum = 233,627.323`
- `deduped_quantity_sum = 210,006.764`
- `removed_quantity_sum = 23,620.559`
- `removed_qty_pct = 10.1104%`

### 2026-04-28

- `raw_quantity_sum = 213,001.971`
- `deduped_quantity_sum = 189,902.017`
- `removed_quantity_sum = 23,099.954`
- `removed_qty_pct = 10.8449%`

## Interpretation

- The `2026-04-21` spike is not unique.
- Duplicate-like inflation exists more broadly in the raw layer.
- On the three investigated dates, quantity inflation was consistently around `9-11%`.

## SQL Equivalent

Equivalent strict dedup in ClickHouse can be reproduced with:

```sql
SELECT count()
FROM
(
    SELECT
        check_datetime,
        bakery_id,
        product_id,
        quantity,
        price,
        line_amount,
        cash_event_type
    FROM fct_check_lines_new
    WHERE cash_event_type = 'Продажа'
      AND quantity >= 0
    GROUP BY
        check_datetime,
        bakery_id,
        product_id,
        quantity,
        price,
        line_amount,
        cash_event_type
)
```

## Recommended Next Step

Run bakery forecasting experiments on the rebuilt deduped dataset:

1. rerun `exp73`
2. rerun `exp74`
3. compare whether the April spike behavior and global bakery metrics improve on deduped facts
