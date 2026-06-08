# Session Handoff - 2026-06-08 - VibeCode Baking Plan Redeploy

## Scope

The latest baking-plan changes existed in git but were not fully active on the
VibeCode Blackhole runtime. The goal was to deliver the current baking-plan
implementation to the production embedded app.

Target server:

```text
server id: 82bb03a8-c356-4225-97a4-a1540cdc29e6
name: bakery-forecast-embedded
appUrl: https://app-8613ac40f10d.vibecode.bitrix24.tech
accessPolicy: PORTAL
```

Relevant latest code:

```text
70bbf68 fix: restore per-SKU baking schedule in baking plan export
b882ddd feat: add baking plan excel export
```

## What Was Deployed

The VibeCode app under `/opt/app` was updated from GitHub `master`, specifically
from:

```text
demand-forecasting-model/apps/forecast_embedded
```

Updated runtime files include:

```text
/opt/app/app/services/baking_plan.py
/opt/app/app/routers/ui.py
/opt/app/app/assets/baking_plan_template.xlsx
/opt/app/requirements.txt
```

The deployed `baking_plan.py` was verified to contain the per-SKU schedule fix:

```text
has_per_sku_schedule_fix = True
has_template = True
```

Dependencies were installed with:

```bash
/opt/app/.venv/bin/python -m pip install -r /opt/app/requirements.txt
```

This includes `openpyxl`, required for `.xlsx` generation.

## Runtime Issue Found

Before the final restart, the server had two competing service definitions:

```text
app.service
forecast-embedded.service
```

`app.service` was the actual working service on port `3000`.

`forecast-embedded.service` was in a restart loop because it tried to start a
second uvicorn process on the same port:

```text
ERROR: [Errno 98] error while attempting to bind on address ('0.0.0.0', 3000): address already in use
```

The conflict was resolved by disabling the duplicate service:

```bash
systemctl disable --now forecast-embedded.service
```

The working service was then restarted:

```bash
systemctl restart app.service
```

Final service state:

```text
app.service: active
forecast-embedded.service: inactive
port 3000: one python uvicorn process
```

Current listener after restart:

```text
0.0.0.0:3000 -> python pid 14018
```

## Smoke Tests

### Internal Blackhole Smoke

Headers used for access-controlled checks:

```text
X-Vibe-User-Id: 27979
X-Vibe-Portal-Id: 390d6913-26b6-4516-9da0-d8d575031afa
X-Vibe-User-Role: member
```

Results:

```text
GET /health                                                     200
GET /bakery/29?date=2026-06-09&week_start=2026-06-09            200
page contains baking-plan.xlsx link                             true
GET /bakery/29/baking-plan.xlsx?date=2026-06-09                 200
generated file is valid XLSX zip                                true
```

The control Excel response was valid after the app restart:

```text
xlsx_size = 11616 bytes
xlsx_zip_ok = True
```

### External Gateway Smoke

A temporary VibeCode `api-bearer` token was created only for the smoke test and
revoked afterward.

Results through the public app URL:

```text
GET  https://app-8613ac40f10d.vibecode.bitrix24.tech/health   200
HEAD https://app-8613ac40f10d.vibecode.bitrix24.tech/          200
```

After the smoke test:

```text
active temporary access tokens = 0
```

## Current Production State

The baking-plan export is now active on the VibeCode app.

Relevant endpoint:

```text
GET /bakery/{bakery_id}/baking-plan.xlsx?date=YYYY-MM-DD
```

The UI button on the bakery detail page now points to this endpoint:

```text
/bakery/{{ bakery.bakery_id }}/baking-plan.xlsx?date={{ selected_date }}{{ run_query }}
```

The deployed runtime uses the current per-SKU baking schedule behavior:

- each SKU row uses its own pre-filled C:L baking schedule from the template;
- empty schedule cells remain empty;
- forecast quantities are allocated only into that SKU's scheduled baking
  windows;
- defrost rows can use next-day early-window demand when available.

## Important Notes

Do not re-enable `forecast-embedded.service` unless the deployment strategy is
changed. The active VibeCode runtime is currently managed by `app.service`.

If a future deploy endpoint recreates `forecast-embedded.service`, check for
port conflicts before assuming the app is down. The correct invariant is:

```text
exactly one uvicorn process listening on 0.0.0.0:3000
```

No ClickHouse schema changes or forecast run changes were needed for this
redeploy.

