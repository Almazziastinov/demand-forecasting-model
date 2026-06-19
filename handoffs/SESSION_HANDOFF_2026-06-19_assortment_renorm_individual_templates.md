# Session Handoff - 2026-06-19 - Assortment Renorm, Hour-Gap Fill, Individual Baking Templates

## Scope

Git actualization + commit of the working-tree changes that accumulated on top
of `6c9a5ff feat: add assortment controls and baking audit`. This session reads
the existing `handoffs/`, documents the uncommitted work, verifies it (tests +
lint), and pushes to `origin/master`.

## Starting Git State

- Branch: `master == origin/master` (was up to date before this session's push).
- HEAD before commit: `6c9a5ff feat: add assortment controls and baking audit`.
- Working tree had 12 modified files + 5 untracked paths (see below).

## What Changed Since 6c9a5ff (this session's commit)

### 1. Assortment-aware allocation: city-scoped filtering + renormalization

`src/experiments_v2/apply_bakery_profiles_clickhouse.py` (+228 lines):

- **City-scoped filter fix.** `filter_by_active_assortment` previously dropped
  any SKU not in the active assortment. Now it only filters cities that are
  actually *scoped* in the assortment table:
  ```python
  scoped_cities = set(allowed[CITY_COL].dropna().astype(str))
  city_is_scoped = work[CITY_COL].astype(str).isin(scoped_cities)
  keep_mask = (~city_is_scoped) | work["_in_active_assortment"] == 1
  ```
  Cities with no assortment coverage pass through unfiltered instead of being
  emptied.
- **New `renormalize_hourly_to_bakery_forecast(...)`.** After assortment
  filtering removes SKUs, the remaining SKU-hour forecasts are rescaled per
  `(date, bakery, hour)` so they sum back to the bakery-hour forecast total.
  Prevents under-forecasting when SKUs are dropped. Returns
  `{groups_scaled, groups_without_sku}`.
- **New `fill_missing_bakery_hours(...)`.** When a bakery-hour has a positive
  bakery forecast but zero SKU allocation (everything filtered out), it
  back-fills SKU rows using, in order:
  1. same-day product day-shares for that bakery (`assortment_hour_gap_fallback`);
  2. city-level hour×product shares via `bakery_city_lookup` for bakeries that
     still have no same-day weights.
  Returns `{groups_filled, groups_unfilled}`.
- Same city-scoped guard was mirrored into
  `build_sku_hour_share_profile.py::filter_hourly_by_assortment`.

### 2. Baking plan: per-bakery individual templates + assortment-driven rows

`apps/forecast_embedded/app/services/baking_plan.py` (+263 lines):

- **Individual templates** for 3 Kazan bakeries, selected by bakery id:
  ```python
  INDIVIDUAL_TEMPLATE_PATHS = {
      20: assets/baking_plan_individual/20_mira_45.xlsx,
      21: assets/baking_plan_individual/21_parkovaya_7.xlsx,
      22: assets/baking_plan_individual/22_sibirskiy_trakt_25.xlsx,
  }
  ```
  New `template_path_for_bakery(bakery_id)` returns the individual template if
  present, else `DEFAULT_TEMPLATE_PATH`.
- **Assortment-driven workbook.** `build_baking_plan_workbook(...)` now accepts
  `assortment_rows` and `template_path`. It builds an assortment lookup
  (`build_assortment_lookup`, `resolve_assortment_product`,
  `_assortment_match_priority`) so plan rows are matched to the city's actual
  active assortment, and appends assortment products that the forecast did not
  surface. Added row snapshot/restore helpers (`_snapshot_row`, `_restore_row`,
  `_numeric_plan_total`) for safe in-place template editing.

`apps/forecast_embedded/app/routers/ui.py`: download endpoint now passes
`assortment_rows=get_city_assortment(city)` and
`template_path=template_path_for_bakery(bakery_id)`.

`apps/forecast_embedded/app/services/bakery.py`: new `get_city_assortment(city)`
reads active rows from `assortment_city_products` UNION unmatched rows from
`assortment_source_audit` (match_status='not_found', keyed as
`unmatched:<hex>`), so the plan can still surface raw assortment names that
never matched `dim_products`.

### 3. Assortment table build: Tatarstan now from OCR, not director file

`scripts/build_city_assortment_table.py` (+52/-43):

- Tatarstan source switched from `read_director_tatarstan` to a generalized
  `read_ocr_scope(...)` → `read_ocr_tatarstan` / `read_ocr_cheboksary`. Both
  scopes now fan a scope's product list across all `SCOPE_TO_CITIES[scope]`.
- **Removed** the special-case that excluded `капуста + курица` from active
  assortment (the Cheboksary carve-out is gone); now only `вишнев*` and the
  explicit `Вишневый` key are manually excluded everywhere.

### 4. Baking-plan template audit: service rows + more aliases

`scripts/audit_baking_plan_templates_assortment.py` (+23):

- New `is_service_row(...)` + `"Служебная строка"` classification with
  recommendation `"Оставить без изменения"` (note/header rows with no role and
  zero qty no longer flagged as missing products).
- Expanded `ALIASES` (конвертик курица, капустный, капуста и мясо, горбуша саго,
  жар киш курица, киш курица, треугольник курица безд, элеш с курицей,
  трехслойник новый, клубника и банан новый, кыстыбый п). NOTE: alias for
  `пирожок капуста курица` changed to `пирожок капуста и курица`.

### 5. Remote rebuild script hardening

`scripts/remote_rebuild_sku_profiles.ps1` (addresses the 2026-06-18 re-export
bug):

- Export step now guarded by `[ ! -s data/raw/sales_hrs_all_clickhouse.csv ]`,
  so a rerun **reuses** the existing CSV instead of re-exporting 30M rows.
- Staging copy uses `-ErrorAction SilentlyContinue` to tolerate transient files.
- Background launch switched from `nohup ... &` to
  `setsid -f bash ... </dev/null` for a cleaner detached process.

### 6. Dev inference passes assortment table

`scripts/dev_run_inference.ps1`: adds
`--assortment-table assortment_city_products$FORECAST_TABLE_SUFFIX`.

## New / Untracked Files Committed

- `apps/forecast_embedded/app/assets/baking_plan_individual/` — 3 Kazan
  templates (`20_mira_45.xlsx`, `21_parkovaya_7.xlsx`,
  `22_sibirskiy_trakt_25.xlsx`).
- `config/baking_plan_template_overrides.csv` — manual template→bakery_id
  mapping with `use`/`hold` decisions. Two **night** scenarios marked `hold`
  (Чуйкова, Дек 8) — do not connect/modify until partner agreement.
- `tests/test_audit_baking_plan_templates_assortment.py`,
  `tests/test_build_city_assortment_table.py` — new tests for the above.

## Not Committed (intentionally)

- `codex_tmp/` — added to `.gitignore` this session. Contains deploy/temp
  artifacts (`forecast_embedded.tar.gz`, deploy bodies, access-grant scripts,
  schema dumps). Local only.

## Verification Performed This Session

- Tests:
  ```
  pytest tests/test_baking_plan.py tests/test_build_sku_hour_share_profile.py \
    tests/test_apply_bakery_profiles_clickhouse_recent.py \
    tests/test_audit_baking_plan_templates_assortment.py \
    tests/test_build_city_assortment_table.py -q
  => 38 passed (4 pandas FutureWarnings, non-blocking)
  ```
- Lint (ruff E,F,W) on all changed `.py` files: **All checks passed**.
- The two pandas `FutureWarning`s come from `pd.concat` over empty/all-NA frames
  in `fill_missing_bakery_hours` (lines ~301, ~336). Harmless now; will need an
  explicit empty-frame exclusion before a future pandas major. TODO.

## Production Invariants (unchanged)

- Forecast pipeline runs on the SSH VM only; VibeCode/Bitrix is embedded
  frontend/API only — do not deploy the pipeline to VibeCode.
- Recent SKU correction mode: `runner_city_prior_soft_weekpart` (in prod since
  2026-06-10).
- Embedded actuals read deduped raw `Svezhar.fct_check_lines`
  (sale event hex `D09FD180D0BED0B4D0B0D0B6D0B0`).
- Do not restore a process-global cached ClickHouse client in the embedded app.

## Next Steps

1. **VM SKU profile rebuild** (carried from 2026-06-18): confirm
   `rebuild_sku_profiles_*.log` finished; verify dev tables
   `sku_hour_share_profile_smoothed_embedded_dev` /
   `sku_hour_uplift_multiplier_embedded_dev` row counts. The re-export bug is
   now fixed in `remote_rebuild_sku_profiles.ps1`.
2. **Run dev inference** with assortment filter + renorm + hour-gap fill; check
   the previously-bad cases (pie overforecast, 05:00/22:00 spikes, 3 Kazan
   bakeries). Do not activate prod until dev comparison is reviewed.
3. **Deploy embedded app** once baking-plan/assortment UI changes are validated:
   ensure individual templates ship and `get_city_assortment` works against the
   prod-suffix tables (`assortment_city_products`, `assortment_source_audit`).
4. Use `template_assortment_audit.xlsx` (2026-06-18) + the new
   `baking_plan_template_overrides.csv` to finish baking-plan template cleanup.
5. Address the `pd.concat` FutureWarning in `fill_missing_bakery_hours`.

## Useful Commands

```powershell
# tests
.venv\Scripts\python.exe -m pytest tests/test_baking_plan.py `
  tests/test_apply_bakery_profiles_clickhouse_recent.py `
  tests/test_audit_baking_plan_templates_assortment.py `
  tests/test_build_city_assortment_table.py -q

# lint
.venv\Scripts\python.exe -m ruff check src/ scripts/ apps/ tests/ --select=E,F,W

# rebuild assortment table
.venv\Scripts\python.exe scripts\build_city_assortment_table.py

# check VM rebuild log
C:\Windows\System32\OpenSSH\ssh.exe -i tmp_remote_keys\new_prod_vm_ed25519 root@201.51.7.24 `
  "tail -n 120 /root/demand-forecasting-model/logs/rebuild_sku_profiles_*.log"
```
