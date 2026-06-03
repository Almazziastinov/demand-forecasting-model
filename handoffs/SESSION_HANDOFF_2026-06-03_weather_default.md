# Session Handoff - 2026-06-03 - Weather as Default Bakery Model Feature

## Context

The user accepted the product/model decision that weather is a default
production bakery-level feature and should always be considered by the bakery
forecast pipeline.

This continues the `2026-06-02` handoff item "Add weather to the production
bakery model".

## Code Changes

Tracked working tree changes:

- `src/experiments_v2/bakery_day_forecast.py`
  - Added `WEATHER_FEATURES` and safe `WEATHER_DEFAULTS`.
  - Added weather normalization/loading/merge helpers.
  - Training and forecasting now load `data/processed/bakery_weather_features.csv`
    by default.
  - `run_train_mode()` and `run_forecast_mode()` also use this default when
    called with a manually-built `argparse.Namespace` that has no `weather_path`.
- `pipelines/forecast_publish/run_production_inference.py`
  - Added default weather path and passes it into bakery forecast mode.
- `pipelines/forecast_publish/nightly_refresh.py`
  - Added weather path argument.
  - Refreshes weather features before train/forecast unless
    `--skip-weather-refresh` is set and the weather file already exists.
  - Passes weather path into train/forecast namespaces.
- `src/experiments_v2/build_bakery_weather_features.py`
  - New production weather feature builder over Open-Meteo.
  - Infers city/date range from `bakery_daily_sales.csv` and optional uplifted
    dataset.
  - Outputs the production contract:
    `date`, `city`, and the weather feature columns used by
    `bakery_day_forecast.py`.
- `tests/test_bakery_day_forecast.py`
  - Added tests for weather normalization, city/date merge, defaults, and
    recursive future weather usage.
- `.gitignore`
  - Added ignore patterns for dated large ClickHouse raw exports and
    `tests/_tmp_smooth_profiles/`.

## Generated Local Artifacts

Generated weather CSV:

```text
data/processed/bakery_weather_features.csv
rows: 5,160
known weather cities: 10
dataset city values: 11 including "unknown"
date range: 2025-01-15..2026-06-14
missing ratio vs known city/date grid: 0.0000
```

The first run failed inside sandbox with a proxy/network error. It succeeded
after escalated network permission for:

```powershell
.venv\Scripts\python.exe -m src.experiments_v2.build_bakery_weather_features
```

## Retrained Local Model Artifacts

Both production bakery models were retrained locally with weather features:

```text
models/bakery_day_model.joblib
models/bakery_day_meta.joblib
models/bakery_day_bias.json

models/bakery_day_model_uplifted.joblib
models/bakery_day_meta_uplifted.joblib
models/bakery_day_bias_uplifted.json
```

Note: the `bakery_day_bias*.json` files are actually CSV-formatted bias tables
with `.json` extension. This matches the existing production code path and was
not changed.

Model metadata now has:

```text
feature_count = 69
weather features present = 13
```

Weather features present in both model metas:

```text
temp_mean
temp_range
precipitation
rain
snowfall
windspeed_max
is_rainy
is_snowy
is_cold
is_warm
is_windy
is_bad_weather
weather_cat_code
```

## Holdout Metrics After Weather Retrain

Base bakery model:

```text
MAE:   97.184164
WMAPE: 10.290592
bias:  21.270282
```

Uplifted bakery model:

```text
MAE:   137.348743
WMAPE: 12.034104
bias:  26.787731
```

Important quality note:

- The previous handoff cited base exp80 smoke metrics around
  `MAE=96.736796`, `WMAPE=9.913078`.
- The weather-enabled base model is slightly worse on this holdout.
- Therefore do not activate a new weather-enabled production run without an
  explicit comparison/acceptance decision.

## Same-Window Check Against Old Exp80 Model

The first aggregate comparison was not fully fair because old exp80 holdout
covered `2026-04-13..2026-05-12`, while the weather retrain holdout covered
`2026-05-02..2026-05-31`.

To compare on one current holdout window, the old exp80 model was evaluated
against current data for `2026-05-02..2026-05-31` and compared with the new
weather model on the same rows.

Report artifacts:

```text
reports/weather_default_same_holdout_comparison_20260603.json
reports/weather_default_same_holdout_city_compare_20260603.csv
```

Same-window aggregate result:

```text
old exp80 model on current holdout:
  MAE:   103.219930
  WMAPE: 10.929704
  bias:  -20.348414

new weather model on current holdout:
  MAE:   97.184164
  WMAPE: 10.290592
  bias:  -21.270282

delta new - old:
  MAE:   -6.035766
  WMAPE: -0.639112 percentage points
  bias:  -0.921868
```

Weather-segment result on the same holdout:

```text
bad-weather rows:
  old WMAPE: 12.053998
  new WMAPE: 12.339593
  delta: +0.285596 percentage points

normal-weather rows:
  old WMAPE: 10.609324
  new WMAPE: 9.706706
  delta: -0.902618 percentage points
```

Interpretation:

- On the current same-window holdout, the weather-enabled model is better on
  aggregate.
- It is not yet better specifically on bad-weather rows.
- Therefore weather can be accepted as a default model feature, but weather
  explanations should not overclaim that bad-weather-day accuracy improved
  until factor-level explanations and bad-weather validation are stronger.

## Smoke Forecasts

Both 14-day bakery forecasts completed successfully:

```text
data/processed/bakery_day_forecast_weather_smoke.csv
rows: 3,038
dates: 2026-06-01..2026-06-14
bakeries: 217
forecast total: 2,657,033.259

data/processed/bakery_day_forecast_uplifted_weather_smoke.csv
rows: 3,038
dates: 2026-06-01..2026-06-14
bakeries: 217
forecast total: 3,219,707.549
```

## Verification

Passed:

```powershell
.venv\Scripts\python.exe -m py_compile pipelines\forecast_publish\nightly_refresh.py src\experiments_v2\build_bakery_weather_features.py src\experiments_v2\bakery_day_forecast.py pipelines\forecast_publish\run_production_inference.py

.venv\Scripts\python.exe -m pytest tests/test_bakery_day_forecast.py tests/test_nightly_refresh.py -v

.venv\Scripts\python.exe -m ruff check src\experiments_v2\build_bakery_weather_features.py src\experiments_v2\bakery_day_forecast.py pipelines\forecast_publish\run_production_inference.py tests\test_bakery_day_forecast.py --select=E,F,W
```

Parser defaults verified for:

```text
src.experiments_v2.bakery_day_forecast
pipelines.forecast_publish.run_production_inference
pipelines.forecast_publish.nightly_refresh
```

## Current Git/Workspace Notes

Tracked files modified:

```text
.gitignore
pipelines/forecast_publish/nightly_refresh.py
pipelines/forecast_publish/run_production_inference.py
src/experiments_v2/bakery_day_forecast.py
tests/test_bakery_day_forecast.py
```

New tracked candidate:

```text
src/experiments_v2/build_bakery_weather_features.py
handoffs/SESSION_HANDOFF_2026-06-03_weather_default.md
```

Ignored local artifacts include:

```text
data/processed/bakery_weather_features.csv
data/processed/bakery_day_forecast_weather_smoke.csv
data/processed/bakery_day_forecast_uplifted_weather_smoke.csv
models/bakery_day_model*.joblib
models/bakery_day_meta*.joblib
models/bakery_day_bias*.json
reports/bakery_day_model*_summary.json
reports/bakery_day_model*_holdout_predictions.csv
```

The huge raw CSV exports are now ignored:

```text
data/raw/sales_hrs_all_clickhouse_2026-05-31.csv
data/raw/sales_hrs_increment_2026-05-13_2026-05-31.csv
```

## Recommended Next Step

Before publishing/activating a weather-enabled production run:

1. Compare old production runs vs new weather-enabled local forecasts by city,
   bakery, and bad-weather-day segments.
2. Decide whether the slight aggregate WMAPE regression is acceptable for the
   product benefit of weather-aware explanations.
3. If accepted, run production inference and publish a non-active draft run
   first, then compare in ClickHouse before activation.

## UI Scope Update: Weather, Events, Holidays Only

The user decided the first UI/context layer should stop at:

- weather as a factual feature;
- holidays;
- event windows.

No model-influence percentages should be shown for weather/events yet.

Additional code changes made after this decision:

- Added ClickHouse serving table:
  `forecast_day_context_embedded`.
- Added context builder:
  `pipelines/forecast_publish/forecast_context.py`.
- `load_forecast_run.py`, `run_production_inference.py`, and
  `nightly_refresh.py` now publish context rows with each forecast run.
- Embedded API summary now returns `context`.
- Embedded index/detail templates show compact context badges:
  temperature, precipitation, rain, snow, wind, bad-weather flag,
  holiday name, pre/post holiday, and event window.

Local context smoke result for the 14-day weather forecast:

```text
rows: 154
dates: 2026-06-01..2026-06-14
cities: 11
bad_weather_rows: 40
```

Important deployment note:

- Existing active ClickHouse runs do not yet have rows in
  `forecast_day_context_embedded`.
- The UI will only show context badges for newly published runs after the
  updated schema and publish code are deployed.

## Backend Access Control Update: Partners

The user decided not to roll anything into production yet. No production deploy,
publish, or activation was performed after this point.

Backend partner access control was added for the embedded forecast app:

- `apps/forecast_embedded/app/auth.py`
  - Added `AuthContext`.
  - Reads Vibe/Bitrix transparent auth headers:
    `X-Vibe-User-Id`, `X-Vibe-Portal-Id`, `X-Vibe-User-Role`,
    `X-Vibe-User-Email`, `X-Vibe-User-Name`,
    `X-Vibe-User-Name-Encoded`, `X-Vibe-Authorization`.
  - Admin roles remain unrestricted:
    `admin`, `administrator`, `portal_admin`, `owner`.
  - Partner/member users require both user id and portal id when access control
    is enabled.
- `apps/forecast_embedded/app/settings.py`
  - Added `ACCESS_CONTROL_ENABLED`.
  - Default is inherited from `BITRIX_EMBED_MODE`.
  - Local/dev mode remains unrestricted unless explicitly enabled.
- `apps/forecast_embedded/sql/schema.sql`
  - Added `bitrix_user_bakery_access_embedded`.
  - Access rows are tenant-bound by `bitrix_portal_id`.
  - The table also stores audit fields:
    `bitrix_work_position`, `partner_name`, `bakery_name`, and
    `match_method`.
- `apps/forecast_embedded/app/services/bakery.py`
  - All bakery/day/SKU/hour queries now accept `AuthContext`.
  - Partner users are filtered by allowed `bakery_id` from
    `bitrix_user_bakery_access_embedded`.
  - Admin/unrestricted users bypass the access table.
- `apps/forecast_embedded/app/routers/api_bakeries.py`
  - API bakery list, summary, SKU-day, and SKU-hour endpoints now enforce
    backend access.
- `apps/forecast_embedded/app/routers/ui.py`
  - Index and bakery detail views now enforce backend access.
- `apps/forecast_embedded/app/routers/api_exports.py` and
  `apps/forecast_embedded/app/services/exports.py`
  - Bakery CSV export now uses the same partner filter.
- `tests/test_forecast_embedded_access.py`
  - Added tests for partner filtering, admin bypass, and required portal id.
- `pipelines/forecast_publish/sync_bitrix_partner_access.py`
  - Added a dry-run/apply sync script.
  - Loads Bitrix/VibeCode users from `/v1/users`.
  - Loads partner-to-bakery assignment from ClickHouse `dim_management`.
  - Matches `dim_management.partner` to Bitrix `lastName + name`.
  - Uses only active Bitrix users by default.
  - Does not require `workPosition` containing `партн` by default; the user
    confirmed that access should follow assigned points even when the Bitrix
    position says manager/admin/other.
  - Excludes closed bakeries by `coalesce(dim_management.status, '') != 'Закрыта'`.
  - Supports manual partner-name overrides from
    `config/bitrix_partner_access_overrides.csv`.
  - Writes only with `--apply`; default mode is read-only dry-run.
- `tests/test_sync_bitrix_partner_access.py`
  - Added tests for name normalization, parenthetical surnames, partner
    position detection, overrides, and optional strict partner-position mode.
- `apps/forecast_embedded/app/services/bakery.py`
  - All bakery-serving queries now also exclude bakeries marked as
    `Закрыта` in `dim_management`, including admin/unrestricted users.

Access table contract:

```sql
bitrix_portal_id String
bitrix_user_id String
bitrix_email Nullable(String)
bitrix_user_name Nullable(String)
bitrix_work_position Nullable(String)
partner_name Nullable(String)
bakery_id Int64
bakery_name Nullable(String)
access_role LowCardinality(String)
match_method LowCardinality(String)
source LowCardinality(String)
updated_at DateTime64(3)
```

Initial read-only dry-run against VibeCode and ClickHouse on 2026-06-03:

```text
portal_id: franshizasvezhar.bitrix24.ru
bitrix_users_loaded: 814
management_rows: 238
management_partners: 77
access_rows: 176
matched_partners: 62
matched_users: 62
apply: false
```

Partners requiring manual review or extra matching rules:

```text
Арсентьева Лилия
Ахатова Айгуль
Ванидовская Анна
Васильева Анастасия
Гарипова Наиля
Гибадуллина Гулия
Имагилов Салават
Кажемякина Екатерина
Карташов Александр
Кислянская Надежда
Лунев Анатолий
Матвеева Владлена
Никифорова Альбина
Сохина Светлана
Хайруллин Рустам
```

Some of these have active Bitrix name matches but do not have a work position
containing `партн`; the default sync intentionally excludes them until the
business rule is confirmed.

Updated business decisions from the user:

- Manual name overrides:

```text
Арсентьева Лилия -> Арсентьева Юлия
Ахатова Айгуль -> Ахатов Ильяс
Гарипова Наиля -> Гарипова Неля
Имагилов Салават -> Исмагилов Салават
Карташов Александр -> Карташова Евгения
```

- For users whose Bitrix position is not partner-like, still provide access
  according to the bakeries assigned to them in `dim_management`.
- Inactive users are dropped unless there is an explicit active replacement.
- Closed bakeries must not be shown in the app.

Updated read-only dry-run after these decisions:

```text
portal_id: franshizasvezhar.bitrix24.ru
bitrix_users_loaded: 814
management_rows: 216
management_partners: 74
access_rows: 194
matched_partners: 74
matched_users: 74
unmatched_partners: []
apply: false
```

Verification passed:

```powershell
.venv\Scripts\python.exe -m ruff check apps\forecast_embedded\app\auth.py apps\forecast_embedded\app\settings.py apps\forecast_embedded\app\services\bakery.py apps\forecast_embedded\app\services\exports.py apps\forecast_embedded\app\routers\api_bakeries.py apps\forecast_embedded\app\routers\api_exports.py apps\forecast_embedded\app\routers\ui.py tests\test_forecast_embedded_access.py --select=E,F,W

.venv\Scripts\python.exe -m py_compile apps\forecast_embedded\app\auth.py apps\forecast_embedded\app\settings.py apps\forecast_embedded\app\services\bakery.py apps\forecast_embedded\app\services\exports.py apps\forecast_embedded\app\routers\api_bakeries.py apps\forecast_embedded\app\routers\api_exports.py apps\forecast_embedded\app\routers\ui.py

.venv\Scripts\python.exe -m pytest tests\test_forecast_embedded_access.py tests\test_forecast_publish_load_run.py tests\test_bakery_day_forecast.py tests\test_nightly_refresh.py -v

.venv\Scripts\python.exe -m ruff check pipelines\forecast_publish\sync_bitrix_partner_access.py tests\test_sync_bitrix_partner_access.py apps\forecast_embedded\app\services\bakery.py tests\test_forecast_embedded_access.py --select=E,F,W

.venv\Scripts\python.exe -m pytest tests\test_sync_bitrix_partner_access.py tests\test_forecast_embedded_access.py -v

.venv\Scripts\python.exe -m pytest tests\test_sync_bitrix_partner_access.py tests\test_forecast_embedded_access.py tests\test_forecast_publish_load_run.py tests\test_bakery_day_forecast.py tests\test_nightly_refresh.py -v
```

## Fallback

If the next step needs to pause or rollback without losing the current state:

- keep `ACCESS_CONTROL_ENABLED=0` in non-embedded local runs;
- do not run `sync_bitrix_partner_access.py --apply`;
- leave `bitrix_user_bakery_access_embedded` unmodified;
- keep closed-bakery filtering in place because it only removes rows that
  should not be shown anyway;
- resume from the current dry-run output and the overrides file in
  `config/bitrix_partner_access_overrides.csv`.

Current remaining access-control work before production:

1. Define/populate `bitrix_user_bakery_access_embedded` from the real partner
   ownership source.
2. Decide whether email-based matching should remain as a fallback or whether
   production should rely only on explicit Bitrix user ids.
3. Apply the schema on the ClickHouse target environment.
4. Smoke-test with real Vibe/Bitrix headers before publishing/activating any
   production run.
