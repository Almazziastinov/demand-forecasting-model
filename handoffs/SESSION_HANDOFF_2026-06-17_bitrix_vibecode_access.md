# Session Handoff - 2026-06-17 - Bitrix/VibeCode Access and Placement

## Scope

The user reported that some partners could not open the embedded forecast app
from Bitrix24. The concrete partner mentioned was Milyausha Burganova. The
visible failure was not an in-app "no bakeries" state, but a higher-level
VibeCode/Bitrix access/bootstrap problem: the app screen asked the user to
authorize on the VibeCode platform or stayed on the Bitrix app-loading screen.

This handoff records the current access findings, placement configuration, and
the live placement republish performed on 2026-06-17.

Do not add API keys, OAuth app keys, bearer tokens, or `.env` contents to this
handoff or to git.

## Current App And Server

VibeCode Blackhole server:

```text
server id: 82bb03a8-c356-4225-97a4-a1540cdc29e6
name: bakery-forecast-embedded
appUrl: https://app-8613ac40f10d.vibecode.bitrix24.tech
mode: BLACKHOLE
status: running
blackholeStatus: CONNECTED
accessPolicy: PORTAL
localPort: 3000
```

Current published OAuth app:

```text
app id: 4ad75c84-c899-4dc6-a4b7-87e1264e55ce
title: ИИ прогноз плана
description: Прогноз плана по пекарням
bitrixClientId: local.6a2182c5a5fee6.33725295
portalId: 390d6913-26b6-4516-9da0-d8d575031afa
portal domain: franshizasvezhar.bitrix24.ru
handlerUrl: https://vibecode.bitrix24.tech/v1/bitrix-handler
appUrl: https://app-8613ac40f10d.vibecode.bitrix24.tech
placements: LEFT_MENU
```

Old app still present but not bound to placements:

```text
app id: 40e205b0-a0bf-4f5a-bfee-80c109b5948a
title: ИИ прогноз плана
bitrixClientId: local.6a0b1a3ee359e3.69188261
handlerUrl: https://vibecode.bitrix24.tech/v1/bitrix-handler
appUrl: https://app-8613ac40f10d.vibecode.bitrix24.tech
placements: []
```

## Placement Handler Finding

The accessible VibeCode app metadata shows the correct handler:

```text
https://vibecode.bitrix24.tech/v1/bitrix-handler
```

This is important. Directly binding the Bitrix placement to the app subdomain
would bypass the VibeCode gateway/session bootstrap and can produce a VibeCode
login screen. The current metadata is correct: `LEFT_MENU` points to the
platform handler, not directly to `app-8613ac40f10d`.

Direct placement inspection through `GET /v1/placements` was not possible with
the available personal `vibe_api_*` key:

```text
OAUTH_APP_REQUIRED: Placement management is only available for OAuth app keys
```

Therefore, the actual Bitrix placement binding could not be inspected through
the placement API without the OAuth app key (`vibe_app_*`). App publish/unpublish
was still available through the app lifecycle API.

## Milyausha Burganova Access Findings

Bitrix user:

```text
bitrix_user_id: 819
name: Миляуша Бурганова
email: mburganova@svezhar.ru
active: true
userType: employee
departmentId: 331
workPosition: франшизный партнер
lastLogin: 2026-06-17T11:57:47.000Z
```

ClickHouse access table:

```text
table: bitrix_user_bakery_access_embedded
bitrix_user_id: 819
email: mburganova@svezhar.ru
rows found under portal domain and portal UUID
source:
  dim_management_bitrix_name_match
  dim_management_bitrix_name_match_uuid_portal
known bakery count from grouped check: 2
min_bakery_id from grouped check: 28
```

The exact full bakery list query later hit ClickHouse connection/read timeout,
so do not rely on this handoff for the second bakery id. The important access
conclusion is that user `819` is active in Bitrix and has backend bakery access
rows. Her reported failure was likely before the request reached the app's
normal authorization/filtering layer.

VibeCode server access list:

```text
userId: 819
userName: Бурганова Миляуша
networkUserId: null
createdAt: 2026-06-08T08:08:49.900Z
```

`networkUserId = null` alone was not treated as proof of failure because other
known users in the server access list also had null network user ids.

## Live Republish Performed

The user asked to try refreshing the placement binding.

First attempt:

```text
POST /v1/apps/4ad75c84-c899-4dc6-a4b7-87e1264e55ce/publish
result: ALREADY_PUBLISHED
```

Then the app was unpublished:

```text
POST /v1/apps/4ad75c84-c899-4dc6-a4b7-87e1264e55ce/unpublish
result: success
placements after unpublish: []
```

Initial republish failed because the current VibeCode API requires a recent
source snapshot before app publish:

```text
SNAPSHOT_REQUIRED
requiredAction: POST /v1/apps/:id/sources
freshnessWindowMinutes: 10
```

A source snapshot archive was created from:

```text
apps/forecast_embedded
```

Excluded:

```text
.env
__pycache__
*.pyc
```

Snapshot upload:

```text
POST /v1/apps/4ad75c84-c899-4dc6-a4b7-87e1264e55ce/sources
contentType: application/zip
tags: manual,published
note: Republish LEFT_MENU placement after auth issue on 2026-06-17
result: success
snapshot id: cmqi3etf10qv6sg0znvbd7l2d
versionId: v1
sha256: 2bc7b58a8007e3b26f3cf08462e645f3a5ad796e948e3f00dbd23f6d99285185
size: 65816
```

The app was then republished with explicit `LEFT_MENU` placement:

```text
POST /v1/apps/4ad75c84-c899-4dc6-a4b7-87e1264e55ce/publish
placements: ["LEFT_MENU"]
appUrl: https://app-8613ac40f10d.vibecode.bitrix24.tech
result: success
updatedAt: 2026-06-17T13:14:43.741Z
```

## Encoding Issue And Fix

After the first republish, Bitrix showed a loading text with broken Cyrillic:

```text
Идет загрузка приложения ?? ?????? ?????
```

Root cause: PowerShell sent the JSON body in a way that corrupted Cyrillic for
the Bitrix placement title/menu title, even though VibeCode app metadata still
looked correct.

Fix applied:

1. `PATCH /v1/apps/:id` with UTF-8 bytes and content type
   `application/json; charset=utf-8`.
2. Short `unpublish -> publish` again, with the publish JSON body explicitly
   encoded as UTF-8 bytes.

Final metadata after the UTF-8 republish:

```text
title: ИИ прогноз плана
description: Прогноз плана по пекарням
handlerUrl: https://vibecode.bitrix24.tech/v1/bitrix-handler
appUrl: https://app-8613ac40f10d.vibecode.bitrix24.tech
placements: ["LEFT_MENU"]
updatedAt: 2026-06-17T13:19:58.858Z
```

Operational note: when sending Cyrillic JSON to VibeCode from PowerShell, do
not pass a plain string body. Convert JSON to UTF-8 bytes:

```powershell
$json = $bodyObj | ConvertTo-Json -Depth 5 -Compress
$bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
Invoke-RestMethod `
  -Method Post `
  -Headers $headers `
  -ContentType 'application/json; charset=utf-8' `
  -Body $bytes `
  -Uri $uri
```

## Separate Backend Issue Seen During Investigation

App runtime itself was reachable, and the server reported `running` /
`CONNECTED`, but app logs also showed a separate ClickHouse timeout while
resolving active forecast run:

```text
clickhouse_connect.driver.exceptions.OperationalError
Read timed out. (read timeout=25)
app/services/runs.py -> get_active_run()
```

This is a backend/runtime data access issue after gateway/auth. It can cause
slow responses or 500s after the user reaches the app, but it does not explain
the VibeCode login screen or broken Bitrix placement loading text.

Direct public requests to the app subdomain without a valid VibeCode gateway
session can return the Blackhole login shell / `BH_LOGIN_REQUIRED`. This is
expected for `accessPolicy=PORTAL` and should not be treated as proof that the
Bitrix placement is broken.

## Current Interpretation

As of the end of this work:

- Milyausha Burganova is an active Bitrix user.
- She has backend bakery access rows.
- She is present in the VibeCode server access list.
- The current app is bound to `LEFT_MENU`.
- The placement handler metadata points to the correct VibeCode platform
  handler.
- The old duplicate app has no placements.
- The Cyrillic menu/loading title was repaired through UTF-8 republish.

If partners still see a VibeCode login screen after reopening from the Bitrix24
left menu, the likely remaining causes are before normal app logic:

1. They are opening the direct `app-8613ac40f10d...` URL instead of the Bitrix24
   left-menu placement.
2. Browser/Bitrix cached an old placement iframe/menu entry.
3. The actual Bitrix placement binding needs deeper inspection or manual bind
   using the OAuth app key (`vibe_app_*`), not the personal `vibe_api_*` key.
4. VibeCode/Bitrix OAuth bootstrap fails for that specific user/session.

## Recommended Next Debug Flow

If the issue repeats for a partner:

1. Ask them to fully refresh Bitrix24 and reopen the app from the left menu,
   not by direct app URL.
2. Immediately inspect VibeCode/app logs for an `embedded index` request or any
   request carrying that Bitrix user id.
3. If no app log entry appears, the failure is still in Bitrix/VibeCode
   placement bootstrap.
4. If an app log entry appears, then debug backend/app access:
   - `X-Vibe-User-Id`;
   - `X-Vibe-Portal-Id`;
   - bakery count for that user;
   - ClickHouse timeout/errors.
5. If placement-level inspection or manual `placement.bind` is required, obtain
   the OAuth app key (`vibe_app_*`) for app
   `4ad75c84-c899-4dc6-a4b7-87e1264e55ce`.

## Files Changed

This handoff is documentation only. No application code was changed as part of
the Bitrix/VibeCode access investigation.
