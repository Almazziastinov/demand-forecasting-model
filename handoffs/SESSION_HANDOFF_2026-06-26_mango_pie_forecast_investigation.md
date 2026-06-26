# Session Handoff — 2026-06-26 — Mango Pie Forecast Investigation

## Scope

Investigation of why Пирог с Манго (product_id=11465) at Кулагина 4 Казань
(bakery_id=16) produces near-zero forecasts on Mondays and Saturdays despite
consistent actual sales of ~4-8 items/day across all days of the week.

No code was changed. This is a research-only session.

## Symptom

Baking plan forecast for this SKU:

| DOW (CH) | Date example | Forecast | Actual |
|---|---|---|---|
| Mon | 2026-06-29 | **0.17** | ~6 |
| Tue | 2026-06-23 | 1.86 | ~4 |
| Wed | 2026-07-01 | 5.90 | ~7 |
| Thu | 2026-06-25 | 1.06 | ~5 |
| Fri | 2026-06-26 | 1.06 | ~6 |
| **Sat** | **2026-06-27** | **0.10** | **~4** |
| Sun | 2026-06-28 | 3.75 | ~4 |

The pattern repeats with zero variance week-over-week (same DOW = same %
share of bakery total, identical to 5+ significant figures). This means the
forecast is derived entirely from the static profile — the recent correction
is not changing anything.

## Root Cause (fully traced)

### Step 1 — Profile is too sparse for tier1

`MIN_TIER1_N_DAYS = 8` in `src/experiments_v2/apply_bakery_profiles.py:61`.

`stream_profile_chunks` (line 640 in `apply_bakery_profiles_clickhouse.py`)
filters `where n_days >= 8`. SKU 11465 at bakery 16 has **only one row** that
passes: `(DOW=1/Tue, hour=7, n_days=8)`. All other (DOW, hour) combinations
have n_days ≤ 7 because the product only started selling at this bakery on
approximately 2026-04-24 (~9 weeks of history at session time).

### Step 2 — Monday gets anomalous hour-22 allocation

For Monday (DOW=0): no tier1 rows for this SKU → zero allocation from tier1.

The only allocation comes from the **fallback path** (`bakery_hour_fallback`):
this applies to (bakery, DOW, hour) triples not covered by tier1 of ANY SKU.
For bakery 16 on Monday, the only such triple is **(DOW=0, hour=22)** — all
hours 6–21 are covered by other SKUs' tier1 data.

The fallback averages `mean_sku_share_in_hour_norm` across ALL DOWs per
(bakery, hour). For hour 22, SKU 11465 has exactly one profile row:
`(DOW=4/Fri, hour=22, n_days=1, mean_sku_share_in_hour_norm=0.333333)`.
This is an anomalous single-observation entry where the SKU was the **only
product sold at hour 22 on that specific Friday** (share_in_hour = 1.0 →
after smoothing → norm = 0.333333).

Bakery 16 has a tiny late-night operation on Mondays:
`bakery_hour_22_Monday ≈ 1.35 items`. After fallback normalization,
SKU 11465 gets ~12.6% of that hour:

```
base_hour_22 = 1.35 × (0.333 / Σ_all_skus_h22) ≈ 0.17
base_daily_Monday = 0.17
```

### Step 3 — Pie cap locks the forecast at 0.17

Category "Пироги сладкие" matches `DEFAULT_RECENT_UPWARD_CAP_CATEGORY_PATTERN`
→ cap is applied.

```
cap = min(base × max_multiplier, recent_dow_avg_qty × recent_absolute_cap_multiplier)
    = min(0.17 × 1.0, ~5.5 × 1.0)
    = 0.17
```

The cap lands at `base`, not at `recent_dow_avg`. The correction model
produces `corrected ≈ 3.9` (from weekpart blending), but the cap cuts it
to 0.17.

`daily_multiplier = corrected / base = 0.17 / 0.17 = 1.0` → hourly stays as-is.
All 0.17 remain at hour 22, hours 6–21 remain at 0.

### Why Wednesday = 5.9 (not 0.17)?

Wednesday also has no tier1 data for this SKU AND bakery 16 has no hour 22
forecast on Wednesdays. So `base_daily_Wed = 0`.

When base = 0, the cap fix (added in the June 2026 VM migration session)
kicks in:
```python
cap = recent_avg_cap  # = recent_dow_avg × 1.0 ≈ 5.9
```
Correction model gives `corrected ≈ 5.9` → passes cap → forecast = 5.9.

**Irony**: having zero profile data on Wednesday is better than having 0.17
of bad data from the hour-22 anomaly on Monday. The anomalous base prevents
the `base=0` protective path from firing.

### Saturday (0.09) — same mechanism

Saturday (DOW=5 Python) has no tier1 for this SKU but its hour 22 has a
tiny bakery forecast (~0.7 items). The fallback gives `base_daily_Sat ≈ 0.09`.
Same cap logic locks it there.

## Key Evidence Queries

All run via `_tmp_query.py` using `.venv/Scripts/python.exe`.

```sql
-- Only tier1 row for SKU 11465 at bakery 16
select dow, hour, n_days, mean_sku_share_in_hour_norm
from sku_hour_share_profile_smoothed_embedded
where bakery_id = 16 and product_id = 11465 and n_days >= 8;
-- Result: (dow=1, hour=7, n_days=8, share=0.003717) — ONE row only

-- Hourly breakdown proving all Monday budget is at hour 22
select forecast_date, hour, sum(forecast_qty) bakery_total,
       sumIf(forecast_qty, product_id = 11465) sku_qty
from sku_forecast_hour_embedded
where run_id = 'prod_weatherfix_uplifted_bakery_norm_uplift_sku_20260623_h14'
  and bakery_id = 16 and forecast_date = '2026-06-29'
group by forecast_date, hour order by hour;
-- Result: hours 6-21 → sku_qty=0.000000, hour 22 → sku_qty=0.170404
```

## Fix Options

**Option A (recommended short-term): Remove the anomalous profile row**

Delete or null-out the row `(bakery_id=16, product_id=11465, dow=4, hour=22)`
from `sku_hour_share_profile_smoothed_embedded` (and `_dev` mirror).

```sql
-- Check first
select * from sku_hour_share_profile_smoothed_embedded
where bakery_id = 16 and product_id = 11465 and dow = 4 and hour = 22;
-- n_days=1, mean_sku_share_in_hour=1.000000
```

After deletion: bakery 16 would have no hour-22 forecast on Mon/Sat for this
SKU → `base_daily` falls to 0 → `base=0` path fires → recent correction gives
~5.5 items for Mon and ~5 for Sat.

ClickHouse MergeTree tables don't support DELETE directly — use
`ALTER TABLE ... DELETE WHERE ...` (mutation) or insert a replacement with
share=0 and n_days=0 and use a different deduplication strategy.

**Option B (medium-term): Rebuild profile from current data**

After another 3–5 weeks, most (DOW, hour) combos will reach n_days=8 → tier1
allocation starts working normally. Profile rebuild script lives at
`src/experiments_v2/apply_bakery_profiles_clickhouse.py` (see profile upload
pipeline in `pipelines/forecast_publish/`).

**Option C: Lower MIN_TIER1_N_DAYS**

Change `MIN_TIER1_N_DAYS = 8` to 3 or 4. More aggressive — affects all SKUs.
Requires testing that thin-data SKUs don't regress.

## Files Referenced

| File | Note |
|---|---|
| `src/experiments_v2/apply_bakery_profiles.py:61` | `MIN_TIER1_N_DAYS = 8` |
| `src/experiments_v2/apply_bakery_profiles_clickhouse.py:640` | `stream_profile_chunks` — `where n_days >= 8` |
| `src/experiments_v2/apply_bakery_profiles_clickhouse.py:1304-1318` | `daily_multiplier` scaling; `base=0` diverges to `new_rows` |
| `src/experiments_v2/apply_bakery_profiles_clickhouse.py:877-960` | `_apply_category_upward_cap` — pie cap logic |
| `src/experiments_v2/apply_bakery_profiles_clickhouse.py:66-72` | `DEFAULT_RECENT_UPWARD_CAP_CATEGORY_PATTERN`, `max_multiplier=1.0` |

## Active Run at Session End

```
run_id: prod_weatherfix_uplifted_bakery_norm_uplift_sku_20260623_h14
horizon: 2026-06-23..2026-07-06
```

No forecast runs, no deploys, no table changes were made in this session.
