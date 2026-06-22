# Session Handoff - 2026-06-22 - Full SKU Profile and Lead-1 Assortment

## Scope

Fix the pilot SKU forecast explosion caused by applying assortment filtering
before profile normalization and then renormalizing the remaining SKUs back to
the full bakery total.

Production was not changed. All profile loads, forecast runs, snapshots, and
activation in this session used `_dev` ClickHouse tables.

## Root Cause

The first assortment-aware profile was built with raw SKU rows filtered before
share normalization. The surviving 75-76 city-assortment SKUs therefore
absorbed the mass of the full catalog inside the profile itself.

Allocation then filtered again and restored the surviving SKU-hour rows to the
full bakery-hour forecast. This second renormalization amplified pilot runners
roughly `1.4x..2.4x` versus the earlier lead-1 forecast.

Examples from the bad run:

- Parina 6, `Кыстыбый П`: `2466 -> 5114`;
- Saliha Batyeva 15, `Треугольник курица безд`: `777 -> 1857`;
- Tukaya 62A, `Треугольник курица безд`: `606 -> 1430`;
- Sibirsky Trakt 25, `Кыстыбый П`: `1183 -> 2497`.

## Corrected Contract

The accepted dev construction is now:

1. build a full chunk-safe SKU profile without an assortment filter;
2. apply recent SKU correction on the full profile;
3. filter the final SKU output to the current city assortment;
4. do not renormalize removed catalog/service mass back into active SKU.

The resulting `allocation_ratio` is not expected to equal `1.0`, because the
bakery model predicts the full catalog while the final SKU output intentionally
contains only the planning assortment. For v3 the ratio is `0.559497`.

## Full Profile Rebuild

VM:

```text
host: 201.51.7.24
root: /root/demand-forecasting-model
log: /root/demand-forecasting-model/logs/rebuild_sku_profiles_20260622_114051.log
```

The existing 6 GB raw ClickHouse export was reused. The profile build ran
without `--assortment-path`.

Loaded dev profile:

```text
table: sku_hour_share_profile_smoothed_embedded_dev
profile rows: 2,620,257
bakeries: 200
products: 883
mean_norm_share_sum: 1.0
```

Loaded uplift multipliers:

```text
table: sku_hour_uplift_multiplier_embedded_dev
profile_version: dev_full_chunk_safe_20260622
rows: 25,456
```

No production profile table was modified.

## Lead-1 v3

Active dev run:

```text
dev_lead1_history_20260601_20260614_full_profile_v3
```

Period and row counts:

```text
dates: 2026-06-01..2026-06-14
bakery rows: 3,025
sku day rows: 215,348
sku hour rows: 1,965,876
lead-1 bakery snapshots: 3,025
lead-1 SKU-day snapshots: 215,348
lead-1 SKU-hour snapshots: 1,965,876
```

Allocation summary:

```text
assortment filter: enabled
renormalization: disabled
allowed city-product pairs: 531
bakery forecast total: 3,247,599.08
planning-assortment SKU total: 1,817,021.99
allocation ratio: 0.559497
```

Summary artifacts:

- `reports/dev_lead1_full_profile_v3_summary.json`
- `data/processed/apply_bakery_profiles_summary_dev_lead1_assortment.json`

## Pilot Audit

The active OCR assortment has:

- 76 products per Tatarstan city;
- 75 products for Cheboksary.

Therefore pilot metrics must be computed against actual sales restricted to
the same active assortment. Comparing the planning forecast to all bakery
sales is not valid.

Assortment-scoped results:

| bakery_id | bakery | bias_pct | sku_wmape_pct | allocation_wmape_pct | pie_bias_pct |
|---:|---|---:|---:|---:|---:|
| 20 | Mira 45 | 14.78 | 44.14 | 36.54 | -9.34 |
| 21 | Parkovaya 7 | 3.23 | 26.49 | 25.11 | 6.47 |
| 22 | Sibirsky Trakt 25 | -11.02 | 27.82 | 23.59 | -12.84 |
| 28 | Gudovantseva 27 | 5.73 | 23.73 | 20.98 | 2.26 |
| 80 | Kalinina 63 | 3.28 | 36.17 | 34.83 | -24.52 |
| 89 | Parina 6 | -10.80 | 27.65 | 25.60 | -3.75 |
| 107 | Chetaeva 46A | -2.77 | 27.76 | 26.84 | -8.59 |
| 221 | Saliha Batyeva 15 | -7.87 | 40.39 | 41.72 | -35.65 |
| 222 | Tukaya 62A | -11.45 | 34.59 | 30.17 | -25.07 |
| 257 | Yarmarochnaya 12 | -0.52 | 37.30 | 36.04 | -0.45 |

Report directory:

```text
reports/dev_pilot_lead1_full_profile_v3_assortment_scope_audit/
```

Top SKU values no longer exhibit the previous mass explosion. Examples:

- Chetaeva 46A, `Треугольник курица безд`: forecast `1482`, fact `1486`;
- Tukaya 62A, `Треугольник курица безд`: forecast `776`, fact `846`;
- Yarmarochnaya 12, `Сосиска в тесте`: forecast `543`, fact `538`.

Pre-06 / 22:00 spike check remained clean for the main pilots. Kalinina has a
small 05/22 combined forecast share of `1.85%`; investigate separately if the
hourly chart still looks operationally wrong.

## Code Changes Not Yet Committed

- `src/experiments_v2/apply_bakery_profiles_clickhouse.py`
  - city-scoped gap fallbacks;
  - own-recent and network cold-start fallbacks;
  - optional assortment renormalization disable switch;
  - pandas empty-frame concat warning fix.
- `pipelines/forecast_publish/run_production_inference.py`
  - forwards `--disable-assortment-renormalization`.
- `scripts/remote_rebuild_sku_profiles.ps1`
  - new `-DisableAssortmentFilter` full-profile build mode.
- `scripts/rebuild_dev_lead1_history.py`
  - reproducible lead-1 history rebuild from bakery lead-1 snapshots.
- `scripts/audit_dev_assortment_run.py`
  - pilot run audit with optional assortment-scoped actuals.
- `tests/test_apply_bakery_profiles_clickhouse_recent.py`
  - recent and network fallback tests.

Current git base:

```text
636c20e feat: assortment renorm, hour-gap fill, individual baking templates
```

## Verification

```text
pytest selected regression suite: 44 passed
ruff E,F,W on changed Python files: passed
git diff --check: passed
dev UI HTTP smoke: 200 at http://127.0.0.1:3001
```

Non-blocking warnings:

- Python 3.16 `u` type-code deprecation from an imported dependency;
- local pytest cache cannot be written due an existing filesystem/ACL issue.

## Current Runtime State

- local dev API is running at `http://127.0.0.1:3001`;
- active dev run is v3 above;
- VibeCode production frontend was not redeployed;
- production VM forecast run and production ClickHouse tables were not changed.

## Recommended Next Steps

1. Review v3 visually in the local dev UI across all ten pilot bakeries.
2. Inspect remaining weak cases:
   - Mira 45 total `+14.78%`;
   - Kalinina pie total `-24.52%` and edge-hour share `1.85%`;
   - Saliha Batyeva allocation WMAPE `41.72%` and pie bias `-35.65%`;
   - Tukaya pie bias `-25.07%`.
3. Confirm that the OCR list of 76/75 products is the intended planning
   assortment, not the full sellable catalog.
4. Do not roll this logic to production until partner review of the pilot
   workbooks/UI is complete.
5. Commit the code and this handoff after reviewing the working-tree diff.

## Resume Commands

```powershell
# rebuild full dev profile on VM
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File scripts\remote_rebuild_sku_profiles.ps1 `
  -SshKeyPath tmp_remote_keys\new_prod_vm_ed25519 `
  -SkipExport -SkipInstall -DisableAssortmentFilter `
  -ProfileVersion dev_full_chunk_safe_20260622 -Background

# rebuild v3 lead-1 history
.venv\Scripts\python.exe scripts\rebuild_dev_lead1_history.py `
  --run-id dev_lead1_history_20260601_20260614_full_profile_v3 `
  --uplift-profile-version dev_full_chunk_safe_20260622 `
  --disable-assortment-renormalization `
  --summary-path reports\dev_lead1_full_profile_v3_summary.json

# assortment-scoped pilot audit
.venv\Scripts\python.exe scripts\audit_dev_assortment_run.py `
  --run-id dev_lead1_history_20260601_20260614_full_profile_v3 `
  --bakery-ids 20 21 22 28 80 89 107 221 222 257 `
  --assortment-path reports\required_assortment\assortment_city_products.csv `
  --output-dir reports\dev_pilot_lead1_full_profile_v3_assortment_scope_audit
```
