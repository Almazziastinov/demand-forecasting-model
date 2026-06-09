# Session Handoff - 2026-06-09 - Embedded Frontend Search Redeploy

## Scope

The embedded VibeCode/Bitrix frontend was updated after the baking-plan deploy.
The requested UI changes were:

- add bakery search;
- move the baking-plan Excel export to a more visible position;
- rename the export action to `Выгрузить план выпекания`;
- include the bakery name in the downloaded XLSX filename in Russian;
- fix mojibake in the user-name pill;
- show labels for every hourly chart tick/value, not only selected hours;
- add back-navigation buttons.

After initial production feedback, the bakery search was fixed and restyled
because the first implementation did not hide results reliably and looked too
plain.

## Commits

Two commits were pushed to `origin/master`:

```text
db29d44 feat: improve embedded bakery navigation
83d0a79 fix: improve embedded bakery search
```

The latest production code after this handoff is:

```text
83d0a79 fix: improve embedded bakery search
```

## Files Changed

Main frontend/runtime files:

```text
apps/forecast_embedded/app/auth.py
apps/forecast_embedded/app/routers/ui.py
apps/forecast_embedded/app/static/app.css
apps/forecast_embedded/app/static/app.js
apps/forecast_embedded/app/templates/bakery.html
apps/forecast_embedded/app/templates/index.html
apps/forecast_embedded/app/templates/layout.html
apps/forecast_embedded/app/templates/sku.html
tests/test_forecast_embedded_access.py
```

## Implemented Behavior

### Bakery Search

The sidebar search now:

- filters while the user types;
- searches bakery name and city text from each sidebar item;
- normalizes case and `ё`/`е`;
- hides rows with CSS class `is-search-hidden`;
- shows a styled search box with a magnifier icon;
- includes a clear button;
- shows a result counter;
- shows `Пекарни не найдены` when no rows match.

Important fix:

The first version used `link.hidden = true`, but `.bakery-link { display: grid; }`
could override the browser hidden behavior. The final version uses:

```text
.bakery-link.is-search-hidden {
  display: none !important;
}
```

### Asset Cache Busting

`layout.html` now references:

```text
/static/app.css?v=20260609b
/static/app.js?v=20260609b
```

This was added so already-open Bitrix iframe tabs do not keep stale CSS/JS.

### Baking Plan Export

On the bakery day page:

- the old small `Excel` link was removed from the right side of the header;
- a larger primary button was added under the bakery city:

```text
Выгрузить план выпекания
```

The XLSX response now uses an ASCII fallback plus RFC 5987 UTF-8 filename:

```text
Content-Disposition:
attachment;
filename="baking_plan_<bakery_id>_<date>.xlsx";
filename*=UTF-8''План%20выпекания%20- ...
```

Example verified in production:

```text
План выпекания - Кулагина 4 Казань - 2026-06-09.xlsx
```

### User Name

`AuthContext.display_name` was added and used in `layout.html`.

It attempts to:

- decode `X-Vibe-User-Name-Encoded`;
- repair common UTF-8 mojibake from `X-Vibe-User-Name`;
- fall back to email.

Tests were added for:

```text
ÐÐ»Ð¼Ð°Ð· ÐÐ¸Ð°ÑÑÐ¸Ð½Ð¾Ð² -> Алмаз Биастинов
base64 encoded Алмаз Биастинов -> Алмаз Биастинов
```

### Hourly Charts

Bakery and SKU chart templates now render:

- all hour tick labels;
- all non-zero actual and forecast value labels.

### Navigation

Added:

- `Назад к неделе` on bakery day page;
- `Назад к дню` on SKU detail page.

## Local Validation

Before deploy:

```text
.venv\Scripts\python.exe -m ruff check apps\forecast_embedded\app\auth.py apps\forecast_embedded\app\routers\ui.py tests\test_forecast_embedded_access.py --select=E,F,W
```

Result:

```text
All checks passed
```

Template compilation:

```text
layout.html, index.html, bakery.html, sku.html -> templates ok
```

Targeted tests after the first frontend batch:

```text
pytest tests/test_forecast_embedded_access.py tests/test_baking_plan.py -v
```

Result:

```text
14 passed
```

## Production Redeploy

Target server:

```text
server id: 82bb03a8-c356-4225-97a4-a1540cdc29e6
name: bakery-forecast-embedded
appUrl: https://app-8613ac40f10d.vibecode.bitrix24.tech
accessPolicy: PORTAL
```

The production runtime was updated by VibeCode `/exec`, not full `/deploy`:

1. download GitHub `master` tarball;
2. copy `apps/forecast_embedded/app` to `/opt/app/app`;
3. copy `requirements.txt`, `README.md`, `RUNBOOK.md` to `/opt/app`;
4. run `py_compile` for changed Python runtime files;
5. keep `forecast-embedded.service` disabled;
6. restart `app.service`.

Important production invariant remains:

```text
app.service: active
forecast-embedded.service: inactive
exactly one uvicorn/python listener on 0.0.0.0:3000
```

Do not re-enable `forecast-embedded.service` unless the deployment strategy is
changed.

## Production Smoke

After the final search redeploy:

```text
app_service=active
dup_service=inactive
health=200
deployed_search_marker_ok
```

Internal HTML/CSS/JS smoke:

```text
GET /?week_start=2026-06-08                         200
search_box                                          yes
clear_button                                        yes
cache_buster                                        yes
GET /static/app.js?v=20260609b                      200
js_filter contains is-search-hidden                 yes
GET /static/app.css?v=20260609b                     200
css_box contains bakery-search-box                  yes
```

Earlier XLSX smoke after frontend navigation deploy:

```text
GET /bakery/16/baking-plan.xlsx?date=2026-06-09     200
xlsx_size                                           11597 bytes
Content-Disposition includes UTF-8 Russian filename yes
```

External gateway smoke:

```text
GET https://app-8613ac40f10d.vibecode.bitrix24.tech/health -> 200
```

Note: because `accessPolicy=PORTAL`, anonymous external `/health` may return
the Blackhole HTML wrapper rather than raw `{"ok": true}`. Internal `/health`
against `127.0.0.1:3000` returned the actual app response.

## User-Facing Note

If an already-open Bitrix tab still shows the old search, reopen the app from
the Bitrix left menu or hard-refresh the iframe page. The final deploy includes
asset cache-busting, but already-open browser state can still hold old DOM until
the page reloads.
