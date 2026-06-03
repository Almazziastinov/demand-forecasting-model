# Session Handoff - 2026-06-03 - Weather Production Rollout

## Context

Continuation of the same-day handoff
`SESSION_HANDOFF_2026-06-03_weather_default.md`.

The user accepted weather as a default bakery model feature and approved
rolling it out to production. This session performed the actual deployment
on the production VM and ClickHouse, including partner access control and
forecast context. After this session the production active forecast run is
the weather-enabled model.

## Decision Recap

Weather was accepted as a default model feature despite a small bad-weather
segment regression:

- aggregate on `2026-05-02..2026-05-31`: MAE -6.04, WMAPE -0.64 p.p.
- bad-weather rows: WMAPE +0.29 p.p. (worse)
- normal-weather rows: WMAPE -0.90 p.p. (better)

UI rule confirmed:

- show weather/holiday/event as factual context badges only
- do not claim "weather: -X%" influence on forecast until factor-level
  explanations and bad-weather validation are stronger

## Latest Relevant Commits

Pushed to `origin/master`:

```text
7cd5d69 fix: drop nullable from context table sort key
21555c9 feat: add embedded partner access controls
ddc8175 docs: update weather access ui handoff
```

## Schema Fix

`apps/forecast_embedded/sql/schema.sql` had a `Nullable(String)` in the
MergeTree `ORDER BY` of `forecast_day_context_embedded`. ClickHouse Yandex
Cloud rejects this without `allow_nullable_key`:

```text
DB::Exception: Sorting key contains nullable columns,
but merge tree setting `allow_nullable_key` is disabled.
```

Fix in `7cd5d69`: changed `city Nullable(String) -> city String`. Safe
because `forecast_context.py` always `fillna("unknown")` before insert, and
the embedded reader does not unpack a nullable.

## VM Schema Apply

VM at `89.108.76.196`, user `forecast`, path `/opt/demand-forecasting-model`.

```bash
sudo -u forecast git pull --ff-only        # 7cd5d69
sudo -u forecast .venv/bin/python ...      # load_schema()
```

Result:

```text
schema applied OK
forecast_day_context_embedded: 0 rows
bitrix_user_bakery_access_embedded: 0 rows
```

Both new tables created via `if not exists`; existing tables untouched.

## Partner Access Sync

`pipelines/forecast_publish/sync_bitrix_partner_access.py --apply`.

The script reads `VIBECODE_API_KEY` only via `os.getenv` (not from `.env`).
Workaround used on VM:

```bash
sudo -u forecast bash -c 'set -a; source .env; set +a; \
  .venv/bin/python -m pipelines.forecast_publish.sync_bitrix_partner_access --apply'
```

(Future improvement: extend the script to read from `.env` like the
ClickHouse settings do.)

Dry-run and apply produced identical numbers, matching the previous handoff:

```json
{
  "management_rows": 216,
  "management_partners": 74,
  "access_rows": 194,
  "matched_partners": 74,
  "matched_users": 74,
  "unmatched_partners": [],
  "non_partner_position_matches": [],
  "portal_id": "franshizasvezhar.bitrix24.ru",
  "bitrix_users_loaded": 814,
  "apply": true
}
```

ClickHouse verification after apply:

```text
total_rows = 194
users      = 74
partners   = 74
bakeries   = 194

match_method:
  partner_name_exact     188
  partner_name_override    6   (5 partners, Карташов has 2 bakeries)
```

Overrides from `config/bitrix_partner_access_overrides.csv`:

```text
Арсентьева Лилия     -> Арсентьева Юлия
Ахатова Айгуль       -> Ахатов Ильяс
Гарипова Наиля       -> Гарипова Неля
Имагилов Салават     -> Исмагилов Салават
Карташов Александр   -> Карташова Евгения
```

## VM Embedded API Access Control Enabled

`.env` was extended on VM:

```text
BITRIX_EMBED_MODE=1
ACCESS_CONTROL_ENABLED=1
```

The `forecast-embedded-api.service` was restarted; environment confirmed via
`/proc/<pid>/environ`.

## VM Smoke Tests Against Embedded API

Endpoint: `http://127.0.0.1:3000`. Real test partner from access table:

```text
bitrix_user_id = 1007 (Лычагин Алексей)
bakery_id = 114 (Энергетиков 3 Казань)
other bakery = 198 (Проспект Московский 145а Наб Челны)
```

Results:

```text
T1 GET /health                                                      200 OK
T2 GET /api/v1/bakeries?date=...     no headers                     401  "Missing X-Vibe-User-Id"
T3 GET /api/v1/bakeries?date=...     partner 1007 + portal          1 bakery: [114]
T4 GET /api/v1/bakeries/114/summary?date=...    partner 1007        200 OK with forecast
T5 GET /api/v1/bakeries/198/summary?date=...    partner 1007        404 "Bakery forecast not found"
T6 GET /api/v1/bakeries?date=...     admin role                     184 bakeries (217 - закрытые)
T7 GET /api/v1/bakeries?date=...     no X-Vibe-Portal-Id            401 "Missing X-Vibe-Portal-Id"
```

Confirmed:

- access control enforced at DB-filter level, not just UI
- partner gets 404 on someone else's bakery
- closed bakeries (`dim_management.status='Закрыта'`) are hidden even from
  admins
- both `X-Vibe-User-Id` and `X-Vibe-Portal-Id` are required

## Weather Artifacts Transferred To VM

Files transferred from local Windows to VM `/tmp/weather_upload` via scp,
then installed with `chown forecast:forecast`:

```text
data/processed/bakery_weather_features.csv          (445 KB)
models/bakery_day_model.joblib                       (21 MB)
models/bakery_day_meta.joblib                         (3 KB)
models/bakery_day_bias.json                          (40 KB)
models/bakery_day_model_uplifted.joblib              (21 MB)
models/bakery_day_meta_uplifted.joblib                (3 KB)
models/bakery_day_bias_uplifted.json                 (42 KB)
```

Pre-weather VM models were backed up to:

```text
/opt/demand-forecasting-model/models/archive_pre_weather/<timestamp>/
```

Meta verification on VM:

```text
bakery_day_meta.joblib
  feature_count = 69
  weather features present = 13 / 13
  date_max = 2026-05-31
  bias = 21.270282

bakery_day_meta_uplifted.joblib
  feature_count = 69
  weather features present = 13 / 13
  date_max = 2026-05-31
  bias = 26.787731
```

## Draft Weather Inference

Run with `--activate-run none` and `--run-prefix weather_draft` to avoid
auto-activation. Timer was stopped beforehand. Command:

```bash
sudo -u forecast bash -c 'set -a; source .env; set +a; \
  .venv/bin/python -m pipelines.forecast_publish.run_production_inference \
    --env-file /opt/demand-forecasting-model/.env \
    --scenario "$FORECAST_SCENARIO" \
    --horizon-days "$FORECAST_HORIZON_DAYS" \
    --uplift-profile-version "$FORECAST_UPLIFT_PROFILE_VERSION" \
    --activate-run none \
    --run-prefix "weather_draft" \
    --notes "weather model draft, no activation"'
```

Result:

```text
PRODUCTION INFERENCE COMPLETE
base_raw_uplift: weather_draft_base_bakery_raw_uplift_sku_20260601_h14
uplifted_norm:   weather_draft_uplifted_bakery_norm_uplift_sku_20260601_h14
```

## ClickHouse Sanity Compare: Active vs Draft

Comparison runs:

```text
ACTIVE: prod_uplifted_bakery_norm_uplift_sku_20260601_h14
DRAFT:  weather_draft_uplifted_bakery_norm_uplift_sku_20260601_h14
```

Row counts (identical):

```text
bakery_forecast_day_embedded     3,038
sku_forecast_day_embedded      425,604
sku_forecast_hour_embedded   3,096,400
forecast_day_context_embedded      154   (draft only; active was pre-context)
```

Bakery day totals:

```text
active:  3,225,853.46
draft:   3,291,938.34    (+2.05%)
avg per bakery-day: 1061.83 -> 1083.59
```

Per-bakery delta distribution (n=217):

```text
up:   189   (87%)
down:  28   (13%)
median delta:  +2.31%
p5 / p95:      -3.66% / +5.64%
```

Outlier discussion:

- bakery_id 188 (Октябрьская 13А Песчаные Ковали): +30.7% on small base
  (~4,315 active -> ~5,641 draft); volatility on small absolute volume, not
  a model bug

## Production Activation

```bash
sudo -u forecast .venv/bin/python -m pipelines.forecast_publish.activate_run \
  --run-id weather_draft_uplifted_bakery_norm_uplift_sku_20260601_h14 \
  --env-file /opt/demand-forecasting-model/.env
```

Result:

```text
RUN ACTIVATED
run_id: weather_draft_uplifted_bakery_norm_uplift_sku_20260601_h14
```

ClickHouse status after activation:

```text
prod_uplifted_*                  archived (1)  draft (1)
weather_draft_uplifted_*         active  (1)  draft (1)
```

Embedded API verification (same partner+bakery as smoke):

```text
GET /api/v1/bakeries/114/summary?date=2026-06-01
  run_id  = weather_draft_uplifted_bakery_norm_uplift_sku_20260601_h14
  status  = active
  day.forecast_final = 889.40        (was 819.68 pre-weather)
  context = {
    city: Казань
    temp_mean: 15.60
    precipitation: 0.0
    windspeed_max: 12.36
    is_bad_weather: false
    event_window_type: post_event_4_7
    days_since_prev_event: 5
    days_to_next_event: 11
  }
```

The key shift: `context` is no longer `null` in API responses.

## Production Timer Re-enabled

```bash
sudo systemctl start forecast-production.timer
```

```text
Active: active (waiting)
Trigger: Thu 2026-06-04 03:30:00 UTC; 12h left
```

That is 06:30 Moscow time. `.env` still has
`FORECAST_ACTIVATE_RUN=uplifted_norm`, so the next scheduled run will
generate `prod_uplifted_bakery_norm_uplift_sku_<date>_h14` from fresh data
and replace the currently-active `weather_draft_*` automatically.

## Current Production State

```text
active forecast run: weather_draft_uplifted_bakery_norm_uplift_sku_20260601_h14
embedded API on VM:  access control ON, context delivered
ClickHouse:          context table + access table populated
partner access:      74 partners, 194 (user, bakery) rows
closed bakeries:     hidden in all queries (including admin)
next scheduled run:  Thu 2026-06-04 03:30 UTC
```

## Not Yet Done

1. **VibeCode embedded redeploy.** The Blackhole app at
   `https://app-8613ac40f10d.vibecode.bitrix24.tech` was last deployed
   before `7cd5d69` and before access control was wired. A fresh deploy is
   needed to pick up the new code; otherwise the app served via Bitrix24 is
   still the pre-access-control version.

2. **VibeCode env: `ACCESS_CONTROL_ENABLED=1`.** When redeploying, the
   VibeCode service environment must include `BITRIX_EMBED_MODE=1` or
   `ACCESS_CONTROL_ENABLED=1`. Without it the VibeCode app stays
   unrestricted.

3. **Smoke through real Bitrix24 entry.** Verify that real
   `X-Vibe-User-Id`/`X-Vibe-Portal-Id` headers from Bitrix24 reach the app
   and filter correctly, ideally with a real partner login and an admin
   login.

4. **`sync_bitrix_partner_access.py` `.env` support.** The script reads
   `VIBECODE_API_KEY` only from `os.getenv`. For systemd usage or for
   simpler manual runs, extend it to also read `VIBECODE_API_KEY` from the
   `.env` file via `load_env_file`, the same way ClickHouse settings work.

5. **Factor-level forecast explanations.** Still required before any UI
   text claiming weather/event/holiday influence on the forecast. Until
   then keep UI to context badges only.

6. **Tomorrow morning verification.** After the next scheduled timer at
   03:30 UTC, verify in ClickHouse that `prod_uplifted_*` was generated and
   activated automatically, and that `weather_draft_*` is no longer
   `active`.

## Useful Commands For Resuming

VM timer state:

```bash
systemctl status forecast-production.timer --no-pager
systemctl status forecast-production.service --no-pager
systemctl status forecast-embedded-api.service --no-pager
```

ClickHouse active run check:

```bash
cd /opt/demand-forecasting-model
sudo -u forecast .venv/bin/python - <<'PY'
from pipelines.forecast_publish.load_forecast_run import create_client
client = create_client(".env")
print(client.query_df("""
  select run_id, status, generated_at, notes
  from forecast_runs_embedded
  where status = 'active'
  order by generated_at desc
"""))
PY
```

Partner access count:

```bash
sudo -u forecast .venv/bin/python - <<'PY'
from pipelines.forecast_publish.load_forecast_run import create_client
client = create_client(".env")
print(client.query_df("""
  select count() as rows, uniqExact(bitrix_user_id) as users,
         uniqExact(bakery_id) as bakeries, uniqExact(partner_name) as partners
  from bitrix_user_bakery_access_embedded final
  where bitrix_portal_id = 'franshizasvezhar.bitrix24.ru'
"""))
PY
```

Rollback (only if needed):

- restore models from
  `/opt/demand-forecasting-model/models/archive_pre_weather/<ts>/`
- re-run inference with the old non-weather artifacts and activate the
  resulting `prod_*` run
- alternatively, find and re-activate the previously-active run via
  `activate_run.py --run-id <archived run_id>`
