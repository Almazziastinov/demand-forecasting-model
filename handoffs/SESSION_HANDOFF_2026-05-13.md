# Session Handoff - 2026-05-13

## Goal
Move to a bakery-driven forecasting pipeline built from longer ClickHouse history:

1. `bakery_day_forecast`
2. `bakery_hour_profile`
3. `sku_hour_share_profile`
4. allocation:
   `bakery_day -> bakery_hour -> sku_hour -> sku_day`

## Agreed modeling direction

- We are **not** using the hourly share heuristic as just an auxiliary feature.
- We discussed a stronger business interpretation:
  for weak/bad SKU, current sales are poor because execution is poor, not because demand is absent.
- We then simplified further and agreed to start with a **bakery-driven baseline for all SKU**:
  - forecast bakery daily sales
  - distribute daily bakery sales by hourly bakery profile
  - distribute bakery hourly sales across SKU using SKU hour share profile
- Separate strong SKU models may be added later as overrides, but **not now**.

## Important conceptual decisions

### 1. Bakery-driven baseline chosen
Chosen option:

`sku_hour_forecast = bakery_day_forecast * bakery_hour_share * sku_share_in_hour`

Where:

- `bakery_hour_share = bakery_sales(hour) / bakery_sales(day)`
- `sku_share_in_hour = sku_sales(hour) / bakery_sales(hour)`

### 2. Snapshot-first, not live-training from ClickHouse

We explicitly did **not** want to train directly against live ClickHouse queries.

Current intended flow:

1. ClickHouse is source of truth
2. export a local snapshot
3. build aggregated datasets/profiles from snapshot
4. run experiments on snapshot

## ClickHouse schema / selected fields

From `notebooks/year_plus_data_from_sql.ipynb`, the available columns were:

- `check_id`
- `line_id`
- `check_number`
- `document_number`
- `check_datetime`
- `check_date`
- `operation_type`
- `cash_event_type`
- `product_id`
- `freshness`
- `quantity`
- `price`
- `discount_amount`
- `line_amount`
- `kkt_id`
- `bakery_id`
- `payment_type`
- `_updated_at`
- `bakery_id` (duplicate from joined dim)
- `bakery_name`
- `city`
- `price_region`
- `product_id` (duplicate from joined dim)
- `product_name`
- `category_name`

We decided to use **English column names** and the following minimal contract:

- `check_datetime`
- `check_date`
- `cash_event_type`
- `quantity`
- `price`
- `line_amount`
- `freshness`
- `bakery_id`
- `bakery_name`
- `city`
- `product_id`
- `product_name`
- `category_name`

## SQL contract agreed

```sql
SELECT
    fcln.check_datetime AS check_datetime,
    fcln.check_date AS check_date,
    fcln.cash_event_type AS cash_event_type,
    fcln.quantity AS quantity,
    fcln.price AS price,
    fcln.line_amount AS line_amount,
    fcln.freshness AS freshness,
    fcln.bakery_id AS bakery_id,
    db.bakery_name AS bakery_name,
    db.city AS city,
    fcln.product_id AS product_id,
    dp.product_name AS product_name,
    dp.category_name AS category_name
FROM Svezhar.fct_check_lines_new AS fcln FINAL
JOIN Svezhar.dim_bakeries AS db
    ON db.bakery_id = fcln.bakery_id
JOIN Svezhar.dim_products AS dp
    ON dp.product_id = fcln.product_id
WHERE fcln.cash_event_type = 'Продажа'
  AND fcln.check_date BETWEEN toDate('2025-01-01') AND toDate('2026-05-13')
```

## Files created / changed this session

### New / changed export layer

- `scripts/export_clickhouse_checks.py`
  - exporter for ClickHouse snapshot
  - changed to expect **English normalized columns**
- `scripts/clickhouse_export_template.sql`
  - now uses the agreed English-column SQL contract
- `requirements.txt`
  - `clickhouse-connect` was added

### New bakery-driven builders

- `src/experiments_v2/build_bakery_daily_dataset.py`
  - chunked processing from raw snapshot CSV
  - builds bakery daily sales dataset
  - adds:
    - daily bakery sales
    - weighted avg price
    - calendar features
    - bakery-level lag / rolling features

- `src/experiments_v2/build_bakery_hour_profile.py`
  - chunked processing from raw snapshot CSV
  - builds bakery-level hourly profile
  - outputs normalized hourly shares by `bakery x dow x hour`

### Tests added

- `tests/test_build_bakery_daily_dataset.py`
- `tests/test_build_bakery_hour_profile.py`

### Earlier exploratory weak-SKU files also added
These are now secondary and probably not the main path, but they exist:

- `src/experiments_v2/weak_sku_hourly_share_target.py`
- `tests/test_weak_sku_hourly_share_target.py`

## Current blocker

### Practical blocker: snapshot export

We tried to use `clickhouse_connect` from shell.

Observed behavior:

- initial confusion came from using the wrong environment
- later the correct `.venv` was confirmed and `clickhouse-connect 0.15.1` is installed there
- however, `python -c "import clickhouse_connect; ..."` from shell appeared to hang badly
- `Ctrl+C` often did not respond
- `python.exe` itself works fine
- `pip` itself works fine
- the issue seems specific to importing / using `clickhouse_connect` in the shell workflow

Because of that, we discussed a fallback:

- use the notebook that already connects successfully to ClickHouse
- export the snapshot from notebook in monthly chunks

## Most likely next step after reboot

### Preferred next step
Do snapshot export from notebook, not from shell, because notebook connection is known to have worked before.

#### Notebook-based monthly export sketch

Use `notebooks/year_plus_data_from_sql.ipynb` and export month-by-month to:

`data/raw/sales_hrs_all.csv`

Then run:

```bat
python src\experiments_v2\build_bakery_daily_dataset.py --source-path data\raw\sales_hrs_all.csv
python src\experiments_v2\build_bakery_hour_profile.py --source-path data\raw\sales_hrs_all.csv
```

### If notebook export works
Next implementation task should be:

- add `src/experiments_v2/build_sku_hour_share_profile.py`

That will complete the baseline chain:

- bakery daily dataset
- bakery hourly profile
- SKU hourly share profile

## Exact intended next implementation order

1. Export snapshot from ClickHouse for `2025-01-01` to `2026-05-13`
2. Build `bakery_daily_sales.csv`
3. Build `bakery_hour_profile.csv`
4. Build `sku_hour_share_profile.py`
5. Build allocation / application layer

## Notes on chunking / memory

We explicitly agreed:

- all large raw processing should be chunked
- no full raw year+ dataset should be loaded fully into memory
- the builders created this session already follow:
  - read chunk
  - aggregate chunk
  - merge partial aggregates

## Sanity result from code side

The new builder scripts were added successfully.

One quick local sanity check was run:

- `build_bakery_hour_profile.build_hour_profile(...)`
  returned normalized shares as expected (`0.25`, `0.75`) on synthetic input.

`pytest` was not consistently run in the broken shell state, so after reboot:

```bat
python -m pytest tests\test_build_bakery_daily_dataset.py -v
python -m pytest tests\test_build_bakery_hour_profile.py -v
```

## Short resume after reboot

If resuming quickly, the assistant should start from:

"We already agreed on the bakery-driven baseline and created the first two chunked builders. The immediate task is to rebuild the ClickHouse snapshot, preferably from notebook in monthly batches, then run the two builders on the resulting CSV."
