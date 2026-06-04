# Session Handoff - 2026-06-04 - VibeCode Redeploy and Smoke

## Context

Continuation of:

- `SESSION_HANDOFF_2026-06-03_weather_prod_rollout.md`

The goal was to verify the first scheduled production run after weather rollout
and redeploy the VibeCode embedded app so it uses the post-access-control code
and environment.

## Production Run Verification

Checked ClickHouse locally through `.env`.

Current active run:

```text
prod_uplifted_bakery_norm_uplift_sku_20260601_h14
```

Status:

```text
status: active
horizon: 2026-06-01..2026-06-14
generated_at: 2026-06-04 06:31:58.279 MSK
notes: uplifted bakery forecast + normalized uplift SKU allocation
```

This confirms the scheduled VM timer replaced the temporary
`weather_draft_*` active run.

Recent run state:

```text
prod_uplifted_bakery_norm_uplift_sku_20260601_h14     draft + active
prod_base_bakery_raw_uplift_sku_20260601_h14          draft
weather_draft_uplifted_bakery_norm_uplift_sku_...     draft + archived
```

Context rows exist for the new production runs:

```text
prod_uplifted_bakery_norm_uplift_sku_20260601_h14     154 rows
prod_base_bakery_raw_uplift_sku_20260601_h14          154 rows
date range: 2026-06-01..2026-06-14
```

## VibeCode Server

Server:

```text
id: 82bb03a8-c356-4225-97a4-a1540cdc29e6
name: bakery-forecast-embedded
appUrl: https://app-8613ac40f10d.vibecode.bitrix24.tech
accessPolicy: PORTAL
```

Initial state:

```text
status: sleeping
blackholeStatus: DISCONNECTED
```

`wake` and `deploy` initially failed because the provider could not start the
VM:

```text
PROVIDER_ERROR: Failed to start VM
```

`refresh` reported only these available actions:

```text
repair
delete
```

`repair` was started. It eventually reported `failed`, but the server recovered
to:

```text
status: running
blackholeStatus: CONNECTED
```

## VibeCode Redeploy

Source:

```text
https://github.com/Almazziastinov/demand-forecasting-model/archive/refs/heads/master.tar.gz
```

App directory:

```text
/opt/app/demand-forecasting-model-master/apps/forecast_embedded
```

Runtime approach:

- no VibeCode runtime id
- install system Python packages in `preStart`
- create local `.venv`
- run `uvicorn app.main:app --host 0.0.0.0 --port 3000`

Production env passed during deploy:

```text
APP_ENV=prod
APP_TITLE=Bakery Forecast Embedded
PORT=3000
BITRIX_EMBED_MODE=1
ACCESS_CONTROL_ENABLED=1
CLICKHOUSE_HOST=<from local .env>
CLICKHOUSE_PORT=<from local .env>
CLICKHOUSE_USER=<from local .env>
CLICKHOUSE_PASSWORD=<from local .env>
CLICKHOUSE_DATABASE=<from local .env>
CLICKHOUSE_SECURE=true
CLICKHOUSE_VERIFY=false
```

Important deploy detail:

- first redeploy installed and started the service, but VibeCode marked it
  failed because healthcheck hit a protected route and received `401`;
- second redeploy added `healthPath=/health`;
- final deploy succeeded.

Final deploy result:

```text
stop_existing: ok
clean: ok
download: ok
install: ok
env: ok
pre_start: ok
systemd: ok
start: ok
healthcheck: ok
tunnel_routing: ok
status: running
appUrl: https://app-8613ac40f10d.vibecode.bitrix24.tech
```

## Smoke Tests

Smoke tests were executed inside the Blackhole VM via Deploy API `exec`, against:

```text
http://127.0.0.1:3000
```

Results:

```text
T1 GET /health
  200 {"ok": true}

T2 GET /api/v1/bakeries?date=2026-06-01
  no Vibe headers
  401 {"detail": "Missing X-Vibe-User-Id"}

T3 GET /api/v1/bakeries?date=2026-06-01
  X-Vibe-User-Id: 1007
  X-Vibe-Portal-Id: franshizasvezhar.bitrix24.ru
  200 count=1 ids=[114]

T4 GET /api/v1/bakeries/114/summary?date=2026-06-01
  partner 1007
  200
  run_id: prod_uplifted_bakery_norm_uplift_sku_20260601_h14
  forecast_final: 889.396466661137
  context_is_null: false

T5 GET /api/v1/bakeries/198/summary?date=2026-06-01
  partner 1007
  404 {"detail": "Bakery forecast not found"}

T6 GET /api/v1/bakeries?date=2026-06-01
  admin role
  200 count=184
```

Confirmed:

- VibeCode app is redeployed with current code.
- Access control is enabled in the VibeCode runtime.
- Missing transparent auth headers are rejected.
- Partner backend filtering works.
- Partner cannot read another bakery.
- Admin bypass works.
- Closed bakeries remain hidden from admin lists.
- Context is present for the active production run.

A temporary VibeCode `api-bearer` access token was created during validation and
revoked after the smoke test.

## Remaining Work

1. Smoke through a real Bitrix24 entry point with a real partner login and an
   admin login, to verify gateway-injected `X-Vibe-*` headers in the portal UI.
2. Extend `pipelines/forecast_publish/sync_bitrix_partner_access.py` so it can
   load `VIBECODE_API_KEY` from `.env`, not only from `os.getenv`.
3. Keep UI weather/event/holiday display as factual context badges only until
   factor-level explanations are implemented and validated.

## Same-Day Follow-Up: OAuth App Replacement and Portal Access Fixed

The original OAuth app was still problematic for non-admin users from the
Bitrix24 left menu. The user created a fresh app authorization key. The app was
authorized through the portal OAuth flow, then published as the left-menu app.

New published app:

```text
app id: 4ad75c84-c899-4dc6-a4b7-87e1264e55ce
Bitrix client: local.6a2182c5a5fee6.33725295
title: ИИ прогноз плана
appUrl: https://app-8613ac40f10d.vibecode.bitrix24.tech
placements: LEFT_MENU
```

The previous app was unpublished:

```text
app id: 40e205b0-a0bf-4f5a-bfee-80c109b5948a
Bitrix client: local.6a0b1a3ee359e3.69188261
placements: []
```

Real portal log checks after the fix:

```text
user 27979, Алмаз Зиастинов: bakeries=184
user 25215, Алмаз Юсупов: bakeries=184
user 17455, Александр Вышкварко: bakeries=184
user 17455 opened /bakery/89 with 200
```

No new runtime errors were present after the replacement app was published.

## Same-Day Follow-Up: Partner Frontend Rework

The embedded frontend was reworked around the partner workflow:

- left sidebar with bakeries available to the current user;
- main weekly view for the selected bakery;
- week start date selector, defaulting to yesterday in Moscow time when it is
  inside the active run horizon;
- day cards with fact sales, forecast, revenue, weather and event context;
- day drill-down with hourly bakery profile and SKU list;
- SKU group/category selector;
- SKU drill-down with hourly SKU profile;
- model/run selector hidden from partners and left visible only for admins.

Fact data sources used:

```text
mart_zero_sales_60d  -> daily actual SKU/bakery quantities and revenue
mart_sales_60d       -> hourly actual quantities
```

Deployment target stayed unchanged:

```text
server id: 82bb03a8-c356-4225-97a4-a1540cdc29e6
appUrl: https://app-8613ac40f10d.vibecode.bitrix24.tech
```

Validation:

```text
python -m compileall apps/forecast_embedded/app
.venv\Scripts\ruff.exe check apps/forecast_embedded/app --select=E,F,W

remote smoke:
GET /?week_start=2026-06-03                         200
GET /bakery/89?date=2026-06-03&week_start=2026-06-03 200
GET /bakery/89/sku/<first_sku>?date=2026-06-03...     200
UTF-8 content checks for Russian page blocks           ok
partner user 25215: weekly page 200, model selector absent
```

### Frontend Bugfix Pass

Follow-up fixes after user review:

- daily fact on the week cards now uses `mart_sales_60d`, the same source as
  the hourly profile, instead of `mart_zero_sales_60d`;
- explicit ClickHouse aliases were added for joined fact/context fields so the
  week cards receive `actual_qty`, `actual_revenue`, `temp_mean`, etc.;
- main week cards now always show compact context badges for temperature,
  precipitation and event/day type;
- raw event codes such as `post_event_4_7` are hidden from the main cards;
- SKU hourly profile now uses a full outer join of forecast and actual hours;
- hourly charts now include point markers, selected value labels and a 24-hour
  fact/forecast value grid under the chart;
- week cards were widened and made horizontally scrollable to avoid text
  squeezing on narrow screens.

Validation after deploy:

```text
GET /?week_start=2026-06-03&bakery_id=89              200
GET /bakery/89?date=2026-06-03&week_start=2026-06-03  200
GET /bakery/89/sku/<first_sku>?date=2026-06-03...     200
main fact/revenue/context checks                       ok
bakery hourly value labels/grid                        ok
SKU hourly points/value grid                           ok
```

## Same-Day Follow-Up: LEFT_MENU Re-Publish and User Access

The user reported that the app still did not work from the Bitrix24 side menu
and that user access was missing.

### Key Finding

`GET /v1/me` for the provided key reports:

```text
type: personal
portal: franshizasvezhar.bitrix24.ru
owner.userId: 27979
owner.name: Алмаз Зиастинов
```

VibeCode docs explicitly say a personal `vibe_api_*` key cannot manage
placements directly. `GET /v1/placements` with this key returned:

```text
OAUTH_APP_REQUIRED: Placement management is only available for OAuth app keys
```

However, app publish/unpublish is available for the existing OAuth app metadata.

### Existing App

```text
app id: 40e205b0-a0bf-4f5a-bfee-80c109b5948a
title: Bakery Forecast Embedded
handlerUrl: https://vibecode.bitrix24.tech/v1/bitrix-handler
appUrl: https://app-8613ac40f10d.vibecode.bitrix24.tech
portalId: 390d6913-26b6-4516-9da0-d8d575031afa
scopes: placement
```

Before the fix, `POST /publish` returned:

```text
ALREADY_PUBLISHED
```

The app was then re-published via the supported lifecycle:

```text
POST /v1/apps/<app_id>/unpublish
POST /v1/apps/<app_id>/publish
```

Publish body used:

```json
{
  "catalogTitle": "Прогноз пекарен",
  "catalogDescription": "Прогноз спроса по пекарням",
  "menuTitle": "Прогноз пекарен",
  "appUrl": "https://app-8613ac40f10d.vibecode.bitrix24.tech",
  "placements": ["LEFT_MENU"]
}
```

Final app metadata:

```text
placements: [LEFT_MENU]
updatedAt: 2026-06-04T11:33:19.874Z
```

### User Access Granted

The current Bitrix24 user was fetched through `/v1/users/me`:

```text
bitrix_user_id: 27979
name: Алмаз Зиастинов
email: a.ziastinov@svezhar.ru
workPosition: Специалист по искусственному интеллекту
```

VibeCode runtime docs say real Gateway `X-Vibe-Portal-Id` is a portal UUID,
while the existing partner access table had rows keyed by portal domain:

```text
domain: franshizasvezhar.bitrix24.ru
uuid:   390d6913-26b6-4516-9da0-d8d575031afa
```

To support both manual smoke headers and real Gateway headers:

1. Existing partner rows from source `dim_management_bitrix_name_match` were
   duplicated from the domain portal id to the UUID portal id under source:

```text
dim_management_bitrix_name_match_uuid_portal
```

2. User `27979` was granted manual access to all currently open bakeries under
   both portal ids with source:

```text
manual_admin_access_20260604
```

Final access summary for user `27979`:

```text
franshizasvezhar.bitrix24.ru                220 bakeries
390d6913-26b6-4516-9da0-d8d575031afa       220 bakeries
```

Note: the active forecast run currently serves 184 open bakeries, so the app
list returns 184 rows even though access was granted for 220 open management
rows.

### Access Smoke

Executed inside the VibeCode VM against `127.0.0.1:3000`.

For user `27979` with portal domain:

```text
GET /api/v1/bakeries?date=2026-06-01     200 count=184
GET /api/v1/bakeries/114/summary?...     200 context_is_null=false
```

For user `27979` with portal UUID:

```text
GET /api/v1/bakeries?date=2026-06-01     200 count=184
GET /api/v1/bakeries/114/summary?...     200 context_is_null=false
```

### Remaining Follow-Up After This Fix

Ask the user to reopen the app from Bitrix24 left menu after the re-publish.
If it still does not appear or still fails, the next required artifact is the
OAuth app key (`vibe_app_*`) from the "Ключи авторизации" section, because
direct placement inspection/bind endpoints reject personal `vibe_api_*` keys.

## Same-Day Follow-Up 2: Side Menu Still Did Not Open

The user then reported:

- side-menu item appeared;
- user access worked;
- but the app still did not open from the side menu.

### Infrastructure Issue Found

VibeCode logs showed conflicting systemd services after prior redeploys:

```text
app.service                            restarting, port already in use
bakery-forecast-embedded.service       restarting, ./.venv/bin/uvicorn not found
python pid 766                         still listening on 0.0.0.0:3000
```

The old services were disabled and removed:

```text
app.service
bakery-forecast-embedded.service
```

The stale process on port 3000 was killed. The app was redeployed as a single
stable service:

```text
serviceName: forecast-embedded
```

Final service state:

```text
forecast-embedded.service active (running)
port 3000 listener: one python process
app.service: not found
bakery-forecast-embedded.service: not found
```

### Gateway Probe Fix

Logs also showed that the platform periodically probes:

```text
HEAD /
```

Before the fix FastAPI returned:

```text
405 Method Not Allowed
```

Added `HEAD / -> 200` in:

```text
apps/forecast_embedded/app/routers/ui.py
```

The app was redeployed from a local archive because GitHub `master` did not yet
contain this local patch.

Verification after deploy:

```text
HEAD /                                      200
GET /health                                200 {"ok": true}
GET / with user 27979 + portal UUID        200 HTML
GET / without Vibe headers                 401 Missing X-Vibe-User-Id
GET /api/v1/bakeries user 27979            200 count=184
```

External Gateway smoke with temporary `api-bearer` token also passed:

```text
GET https://app-8613ac40f10d.vibecode.bitrix24.tech/health   200
HEAD https://app-8613ac40f10d.vibecode.bitrix24.tech/         200
GET https://app-8613ac40f10d.vibecode.bitrix24.tech/          200 HTML
```

The temporary token was revoked after the smoke test.

### Current Interpretation

The app server, Blackhole tunnel, direct Gateway path, and backend access for
user `27979` now work.

If the Bitrix24 left-menu item still does not open after this point, the
remaining failure is before the request reaches the app server: likely the
VibeCode `/v1/bitrix-handler` / OAuth placement bootstrap path. Deeper
inspection or manual `placement.bind` requires the OAuth app key
(`vibe_app_*`), because personal `vibe_api_*` keys return:

```text
OAUTH_APP_REQUIRED
```

## Same-Day Follow-Up 3: Colleague Access Screenshot

The user shared a screenshot from a colleague. Bitrix24 showed:

```text
У вас недостаточно прав для доступа к данному приложению.
Обратитесь к администратору вашего Битрикс24.
```

The screenshot URL was on the Bitrix24 side:

```text
franshizasvezhar.bitrix24.ru/devops/placement/109/
```

Logs showed that one Bitrix placement request reached the app as:

```text
POST /?DOMAIN=franshizasvezhar.bitrix24.ru&PROTOCOL=1&LANG=ru&APP_SID=...
405 Method Not Allowed
```

The app previously supported `GET /` and `HEAD /`, but not `POST /`.

Fix added:

```text
apps/forecast_embedded/app/routers/ui.py
  @router.post("/", response_class=HTMLResponse)
```

The app was redeployed from a local archive again.

Verification on the VibeCode VM:

```text
POST / with user 27979 + portal UUID      200 HTML
POST / without Vibe headers               401 Missing X-Vibe-User-Id
HEAD /                                    200
forecast-embedded.service                 active
```

If the colleague still sees the Bitrix24 permission screen after this POST fix,
then the issue is likely Bitrix24 app-level permission before iframe delivery.
At that point we need either:

- the colleague's Bitrix user id/name/email to grant backend forecast access if
  the iframe opens but data is empty/forbidden; or
- the OAuth app key (`vibe_app_*`) to inspect/fix placement-level permissions,
  because personal `vibe_api_*` cannot call placement management endpoints.

## Same-Day Follow-Up 4: Concurrent ClickHouse Session Fix

After the user confirmed the `vibe_app_*` key already existed in VibeCode UI,
the latest logs showed that colleague/portal requests were now reaching the app.

Important request trace:

```text
GET /?member_id=...&placement=LEFT_MENU&placement_options=...
```

One request failed with:

```text
500 Internal Server Error
clickhouse_connect.driver.exceptions.ProgrammingError:
Attempt to execute concurrent queries within the same session.
Please use a separate client instance per thread/process.
```

Root cause:

```text
apps/forecast_embedded/app/db.py
```

`get_client()` used `@lru_cache(maxsize=1)`, so every FastAPI request shared
one ClickHouse client/session. Under concurrent iframe/user requests,
ClickHouse Connect rejected overlapping queries in the same session.

Fix:

```text
removed @lru_cache(maxsize=1) from get_client()
```

This makes each service call create its own ClickHouse client instead of
sharing one process-global session.

Validation:

```text
py_compile apps/forecast_embedded/app/db.py apps/forecast_embedded/app/routers/ui.py
ruff check apps/forecast_embedded/app/db.py apps/forecast_embedded/app/routers/ui.py --select=E,F,W
```

VibeCode app was redeployed from a local archive.

Parallel smoke inside the VibeCode VM:

```text
6 concurrent GET/POST / requests with user 27979 + portal UUID
all returned 200
response size: 102958 bytes each
GET /health returned 200
journal grep for 500/Internal/concurrent/ProgrammingError returned no rows
```

Current interpretation:

- left-menu requests now reach the app;
- `POST /` and `HEAD /` are supported;
- process-global ClickHouse concurrency bug is fixed;
- if a colleague still fails, inspect that user's Bitrix id and backend access
  rows next, rather than app-level infrastructure first.

## Same-Day Follow-Up 5: Explicit Colleague Access

The user asked to explicitly add:

```text
Юсупов Алмаз
Вышкварко Александр
```

Bitrix users were found by paginating `/v1/users`:

```text
25215  Алмаз Юсупов          a.yusupov@svezhar.ru       Аналитик
17455  Александр Вышкварко   a.vyshkvarko@svezhar.ru    Аналитик
```

Both users were added to `bitrix_user_bakery_access_embedded` with manual
all-open-bakery access under both possible portal ids:

```text
franshizasvezhar.bitrix24.ru
390d6913-26b6-4516-9da0-d8d575031afa
```

ClickHouse verification:

```text
17455  a.vyshkvarko@svezhar.ru  220 bakeries under each portal id
25215  a.yusupov@svezhar.ru     220 bakeries under each portal id
source: manual_admin_access_20260604
```

## Same-Day Follow-Up 6: Partner Retry Diagnostics

After another failed retry from partner accounts, fresh VibeCode logs showed a
left-menu request reaching the FastAPI app:

```text
GET /?member_id=...&placement=LEFT_MENU&placement_options={"URI":"/online/"} 200 OK
```

This means at least one recent attempt passed the Bitrix/VibeCode iframe
bootstrap and reached the embedded app. The next likely failure mode is user
data access: a Bitrix user without rows in `bitrix_user_bakery_access_embedded`
gets an empty forecast list.

Added diagnostics and empty-state UX:

```text
apps/forecast_embedded/app/routers/ui.py
  log request_id, user_id, email, portal_id, role, selected date, bakery count

apps/forecast_embedded/app/templates/index.html
  show "Нет доступных пекарен..." with the current Bitrix user id when the
  forecast list is empty

apps/forecast_embedded/app/templates/layout.html
  fix visible title/nav text to UTF-8 Russian:
  "ИИ прогноз плана", "Главная"
```

The diagnostic log is emitted at warning level so it appears in VibeCode
`/logs`.

The app was redeployed again from a local archive using merge deploy into
`/opt/app` with existing `/opt/app/.env` preserved.

Validation after deploy:

```text
py_compile apps/forecast_embedded/app/routers/ui.py apps/forecast_embedded/app/db.py
ruff check apps/forecast_embedded/app/routers/ui.py apps/forecast_embedded/app/db.py --select=E,F,W
```

Internal smoke on the VibeCode VM as normal `member` users:

```text
UID=27979 ROWS=184
UID=25215 ROWS=184
UID=17455 ROWS=184
UID=999999 ROWS=1   # empty-state row; diagnostic log reports bakeries=0
```

VibeCode logs now include:

```text
embedded index request_id=smoke_27979 user_id=27979 ... role=member is_admin=False date=2026-06-01 bakeries=184
embedded index request_id=smoke_25215 user_id=25215 ... role=member is_admin=False date=2026-06-01 bakeries=184
embedded index request_id=smoke_17455 user_id=17455 ... role=member is_admin=False date=2026-06-01 bakeries=184
embedded index request_id=smoke_999999 user_id=999999 ... role=member is_admin=False date=2026-06-01 bakeries=0
```

Next debugging step if a real user still reports "does not work":

1. Ask them to reopen the app from the left menu.
2. Immediately inspect VibeCode logs for `embedded index`.
3. If their `bakeries=0`, grant that exact `user_id`/email access in
   `bitrix_user_bakery_access_embedded`.
4. If there is no `embedded index` line at all, the failure is still before the
   iframe reaches the app, likely Bitrix/VibeCode placement permission rather
   than forecast backend ACL.

Note: PowerShell mangled the Cyrillic audit-name fields on the first insert, but
the access filter uses `bitrix_user_id` and email, so access behavior is not
affected. Emails and ids are correct.

Also checked VibeCode Blackhole server access list. Both users were already
present there:

```text
25215  Алмаз Юсупов
17455  Александр Вышкварко
```

Retrying Deploy API `exec` for an internal smoke returned `TUNNEL_NOT_FOUND`
despite `/infra/servers/:id` reporting:

```text
status: running
blackholeStatus: CONNECTED
accessPolicy: PORTAL
```

So the final direct app smoke for these two users was not completed in this
follow-up, but ClickHouse access rows and VibeCode server access rows are in
place.

## Same-Day Follow-Up: Embedded UI and SKU Hour Profile Notes

The partner UI was redesigned and deployed to VibeCode server
`82bb03a8-c356-4225-97a4-a1540cdc29e6`.

Implemented:

- left sidebar with bakeries available to the current Bitrix user;
- weekly view for the selected bakery with a week-start date picker;
- day cards with forecast, actual sales, revenue, temperature, precipitation
  and day/event type;
- drill-down from week card to bakery day profile;
- bakery day page with hourly fact/forecast chart and 24-hour value grid;
- SKU group filter;
- drill-down from SKU list to SKU hourly profile;
- partner users use the active/default run only; run/model selector is visible
  only for admins.

Important bugfixes after user review:

- week-card actual sales now use `mart_sales_60d`, matching the hourly fact
  source, instead of `mart_zero_sales_60d`;
- explicit ClickHouse aliases were added for joined fields, so week cards
  receive `actual_qty`, `actual_revenue`, `temp_mean`, etc.;
- raw event codes like `post_event_4_7` are hidden and replaced with Russian
  labels;
- SKU hourly profile uses full outer join of forecast and actual hours so
  actual-only hours are not dropped;
- chart markers, selected value labels and a 24-hour fact/forecast value grid
  were added;
- week cards are wider and horizontally scrollable to avoid squeezed text.

Validation after final deploy:

```text
python -m compileall apps/forecast_embedded/app
.venv\Scripts\ruff.exe check apps/forecast_embedded/app --select=E,F,W

GET /?week_start=2026-06-03&bakery_id=89               200
GET /bakery/89?date=2026-06-03&week_start=2026-06-03   200
GET /bakery/89/sku/<first_sku>?date=2026-06-03...      200
main fact/revenue/context checks                        ok
bakery hourly labels/grid                               ok
SKU hourly points/value grid                            ok
```

SKU hourly forecast formula:

```text
bakery_hour_forecast = bakery_day_forecast * mean_hour_share_norm

sku_hour_forecast =
    bakery_hour_forecast * mean_sku_share_in_hour_norm

full form:
sku_hour_forecast =
    bakery_day_forecast
    * bakery_hour_share_norm
    * sku_share_in_hour_norm
```

Profile construction:

- `mean_hour_share_norm` is built from historical bakery hourly share by
  `bakery_id + dow + hour` and normalized within `bakery_id + dow`;
- `mean_sku_share_in_hour_norm` is built from historical SKU share inside the
  bakery-hour by `bakery_id + product_id + dow + hour`;
- long and recent SKU shares are blended as `0.6 * long_share + 0.4 *
  recent_share` when recent history exists;
- exact SKU profile rows require `n_days >= 8`;
- below that gate the allocation falls back to `bakery_id + hour + product_id`
  shares, tagged as `bakery_hour_fallback_thin` or `bakery_hour_fallback`.
