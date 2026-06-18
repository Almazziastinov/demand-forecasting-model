# Session Handoff - 2026-06-16 - Git Actualization

## Scope

This handoff refreshes the project state from git after reading the existing
`handoffs/` directory. The previous latest operational handoffs ended around
2026-06-11, but `origin/master` contains additional commits after that.

Current local repository was fetched from `origin` and local `master` matches
`origin/master`.

## Current Git State

Current HEAD:

```text
ceeb4e9 fix: show loading overlay for embedded actions
```

Branch:

```text
master == origin/master == origin/HEAD
```

Important untracked local files:

```text
.codex/prod_vm.env
notebooks/bakery_fact_vs_forecast_review.ipynb
```

Do not print or commit `.codex/prod_vm.env`; it likely contains VM/runtime
environment values or secrets.

The notebook is a local analysis artifact and may be useful for rollout review.
It is not currently committed.

`git status` also warns about permission-denied temp directories:

```text
.pytest_tmp_codex/
codex_tmp/
tests/_tmp_pytest/
```

These appear to be local temp/runtime directories, not project source.

## Commits After Snapshot Handoff

The latest committed handoff documents:

```text
711973e feat: read embedded history from lead snapshots
```

Additional commits now present on `master`:

```text
9026d33 docs: add embedded snapshot history handoff
8f031f2 fix: read embedded actuals from raw sales
3554a9a fix: retry embedded clickhouse client init
97b2723 fix: dedupe embedded raw actuals
a46f0a1 fix: dedupe bakery daily clickhouse export
f2f0fc1 fix: support deduped raw recent sku correction
1f576e3 fix: alias deduped raw recent sales columns
674a7f3 fix: label embedded run selector as forecasts
0ad6cea fix: account for baking batch sizes
ceeb4e9 fix: show loading overlay for embedded actions
```

## Updated Production/Serving Interpretation

### Forecast pipeline remains VM-only

The core invariant still holds:

- production forecast generation runs on the SSH VM;
- VibeCode/Bitrix app is only the embedded frontend/API;
- do not deploy the forecast pipeline to VibeCode.

Known production paths from previous handoffs:

```text
VM path: /opt/demand-forecasting-model
VibeCode server id: 82bb03a8-c356-4225-97a4-a1540cdc29e6
VibeCode app URL: https://app-8613ac40f10d.vibecode.bitrix24.tech
```

### Active run context from last handoff

Latest handoff-known active production run after dataset refresh:

```text
prod_uplifted_bakery_norm_uplift_sku_20260611_h14
horizon: 2026-06-11 .. 2026-06-24
```

Snapshot tables exist and are used by embedded history:

```text
bakery_forecast_day_snapshots
sku_forecast_day_snapshots
sku_forecast_hour_snapshots
```

Embedded API combines:

- active-run tables for current/future dates;
- snapshot tables with `lead_days = 1` for historical forecast display.

## Important New Behavior Since 2026-06-11

### 1. Embedded actuals now use deduped raw check lines

`apps/forecast_embedded/app/services/bakery.py` now reads actual sales from:

```text
Svezhar.fct_check_lines
```

through a deduped source:

```sql
SELECT DISTINCT
    fcl.check_datetime,
    fcl.check_date,
    fcl.bakery_id,
    fcl.product_id,
    fcl.quantity,
    fcl.price,
    fcl.line_amount,
    fcl.cash_event_type
FROM Svezhar.fct_check_lines AS fcl
WHERE hex(fcl.cash_event_type) = %(sales_event_hex)s
```

The sale event hex constant is:

```text
D09FD180D0BED0B4D0B0D0B6D0B0
```

This replaced previous reliance on `mart_sales_60d` for embedded factuals in
the affected service queries. Treat this as the current source of truth for UI
actuals unless later changed.

### 2. ClickHouse bakery-day export dedupes raw checks

`scripts/clickhouse_bakery_daily_template.sql` now also dedupes raw check lines
with `SELECT DISTINCT` before aggregating bakery-day sales.

This matters for production dataset refresh:

- source remains raw ClickHouse checks;
- aggregation avoids duplicated raw lines;
- output feeds the compact bakery daily refresh path.

### 3. Recent SKU correction supports deduped raw sales

`src/experiments_v2/apply_bakery_profiles_clickhouse.py` now supports recent
correction using the deduped raw sales source.

Affected mode:

```text
runner_city_prior_soft_weekpart
```

The logic still:

- uses recent daily / weekpart shares;
- guards runners with city prior;
- caps/reduces eclair over-allocation;
- excludes/de-emphasizes service categories.

But the recent stats source has been adjusted so duplicated raw lines do not
distort recent SKU shares.

### 4. Embedded ClickHouse client has retry

`apps/forecast_embedded/app/db.py` now retries ClickHouse client creation:

```text
attempts: 3
sleep: 2 seconds
connect_timeout: 25
send_receive_timeout: 300
```

The important earlier invariant still applies: do not restore a process-global
cached ClickHouse client, because that previously caused concurrent-session
errors in FastAPI.

### 5. Run selector display was renamed

`apps/forecast_embedded/app/services/runs.py` now creates user-facing display
names like:

```text
Прогноз 15.06-28.06
```

The embedded selector should be understood as a forecast selector, not a model
selector.

### 6. Baking plan now accounts for batch sizes

`apps/forecast_embedded/app/services/baking_plan.py` was substantially updated.

Current behavior:

- reads each SKU row's pre-filled C:L cells as that SKU's baking schedule;
- derives batch size / rounding step from the GCD of template quantities;
- rounds forecast production quantities up to batch size;
- carries surplus from an earlier bake window into later windows;
- for one early bake with later demand, can add a midday split window;
- defrost cells are detected from cell value, not SKU name;
- night defrost uses next-day early-window forecast when available;
- defrost keeps the original annotation text while replacing the number.

Tests in `tests/test_baking_plan.py` cover:

- coverage-hour tiling;
- single-window full-day coverage;
- scheduled-column-only writes;
- batch GCD rounding;
- surplus carry;
- midday split;
- defrost handling;
- aliases and revenue bucket thresholds.

### 7. Embedded UI loading overlay

`apps/forecast_embedded/app/static/app.js`,
`apps/forecast_embedded/app/static/app.css`, and `layout.html` now implement a
loading overlay for:

- form submits;
- select/input changes;
- normal navigation links;
- baking-plan Excel downloads.

For `baking-plan.xlsx`, the overlay auto-hides after a timeout/focus return so
the page is not stuck after browser download handling.

## Untracked Notebook: Bakery Fact Vs Forecast Review

Local file:

```text
notebooks/bakery_fact_vs_forecast_review.ipynb
```

Purpose:

- interactive bakery/SKU review;
- compares fact vs forecast at bakery and SKU levels;
- supports holdout variants and fresh ClickHouse live mode.

Important live-mode defaults currently visible in the notebook:

```python
LIVE_BAKERY_IDS = [16, 28, 23]
LIVE_DATE_FROM = "2026-06-11"
LIVE_DATE_TO = "2026-06-24"
LIVE_LEAD_DAYS = 1
```

Live mode:

- reads fact from deduped `Svezhar.fct_check_lines`;
- reads forecast from `sku_forecast_day_snapshots`;
- uses `lead_days = 1`;
- returns the same schema as holdout review:
  `date`, `bakery_id`, `bakery_name`, `city`, `product_id`,
  `product_name`, `category_name`, `fact_qty`, `forecast_qty`.

This notebook is useful for rollout selection and post-snapshot quality review,
but it is not currently tracked.

## Rollout Selection Context Still Relevant

The earlier rollout guidance remains valid:

- do not select stores only by bakery-day WMAPE;
- SKU allocation quality is the main risk;
- first rollout should be decision support, not automatic production control;
- runner SKU and eclair rows need separate inspection.

Previously recommended cleaner candidate shortlist:

```text
16   Кулагина 4 Казань
28   Гудованцева 27 Казань
14   Ямашева 19А Казань
89   Парина 6 Казань
25   Фучика 30 Казань
23   Зорге 101 Казань
62   Шамиля Усманова 16А Казань
29   Айдарова 8А корп 1 Казань
```

Previously excluded from first rollout:

```text
30   Баумана 29/11 Казань
105  Фучика 96 Казань
60   Мусина 68 Казань
```

The new notebook is likely the best current artifact for reviewing this cohort
against fresh June snapshot forecasts.

## Current High-Value Next Steps

1. Verify actual production state on the VM:
   - latest deployed commit;
   - active run id and horizon;
   - timer health;
   - snapshot counts for latest run;
   - whether commits through `ceeb4e9` have been deployed to VibeCode and/or VM
     where relevant.

2. Use `notebooks/bakery_fact_vs_forecast_review.ipynb` for candidate rollout
   review on fresh snapshot dates:
   - bakery-day total;
   - top runner SKU;
   - eclair rows;
   - forecast-only/dead SKU rows;
   - category-level bias.

3. Decide whether to commit the notebook:
   - if yes, clean outputs/secrets first;
   - make sure it does not embed ClickHouse credentials or prod env values.

4. If deploying embedded app again, ensure VibeCode has:
   - current frontend/API code through `ceeb4e9`;
   - exactly one uvicorn service on port `3000`;
   - access control still enabled;
   - no cached global ClickHouse client.

5. If deploying VM pipeline again, ensure VM has:
   - current code for deduped raw refresh/export and recent SKU correction;
   - `.env` still has `FORECAST_REFRESH_DATASETS=1`;
   - recent correction mode remains `runner_city_prior_soft_weekpart`;
   - enough swap for the production inference job.

## Verification Already Performed In This Actualization

Local git fetch:

```text
git fetch --prune origin
```

completed successfully after elevated permission.

Confirmed:

```text
HEAD -> master
origin/master -> ceeb4e9
origin/HEAD -> origin/master
```

No tests were run during this handoff creation. This was a repository-state and
context refresh only.

