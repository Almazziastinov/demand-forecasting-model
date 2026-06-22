# Session Handoff - 2026-06-22 - Baking Plan Production Deploy

## Scope

Deploy only the embedded Excel baking-plan changes and the production
`bakeable_products` reference table. The forecast model, VM forecast pipeline,
profile tables, active run, and forecast outputs were not changed.

## Git

Implementation commit:

```text
a096be4 feat: finalize assortment-scoped baking plans
```

The commit was pushed to `origin/master` before deployment.

## Production ClickHouse

The old empty `bakeable_products` table used the obsolete key
`(product_id, valid_from)` and did not contain `city`.

It was recreated with:

```text
order by (city, product_id, valid_from)
```

Loaded production data:

```text
rows: 286
cities: 7
Tatarstan cities: 41 SKU per city
Cheboksary: 40 SKU
```

The dev table `bakeable_products_dev` has the same schema and row counts.

## Embedded App Deployment

Target:

```text
server id: 82bb03a8-c356-4225-97a4-a1540cdc29e6
app URL: https://app-8613ac40f10d.vibecode.bitrix24.tech
```

Only `apps/forecast_embedded/app` was updated from commit `a096be4` under
`/opt/app`. A server-side backup was created before replacement, and the deploy
command had automatic rollback on failed healthcheck.

Verified runtime invariant:

```text
app.service: active
forecast-embedded.service: inactive
listeners on port 3000: 1
internal health: 200
external health: 200
```

## Excel Smoke

Production endpoint smoke:

```text
GET /bakery/29/baking-plan.xlsx?date=2026-06-22 -> 200
valid XLSX: yes
last column: M / Итого
max columns: 13
obsolete columns 8:00-10:00, 10:00-12:00, 12:00-15.00: absent
```

The endpoint now reads the city/date-effective `bakeable_products` allowlist.
Missing city or allowlist data fails closed with HTTP 503 instead of adding
bought-in products back to the baking plan.

## Forecast Invariant

The active forecast remained:

```text
prod_uplifted_bakery_norm_uplift_sku_20260622_h14
horizon: 2026-06-22..2026-07-05
```

No model or forecast pipeline deployment was performed on the production VM.

## Strawberry And Banana SKU Audit

Both names are real, separate records in `dim_products`:

```text
000001743  Клубника и банан
000011301  Клубника и банан НОВЫЙ
```

The current city bakeable list includes both records.

Deduped raw sale-event facts from `fct_check_lines`:

```text
000001743 / Клубника и банан
  history: 2025-01-15..2026-06-22
  total quantity: 161,697.96
  last 30 days: 66 units, 28 sale days, only 1 bakery

000011301 / Клубника и банан НОВЫЙ
  history: 2025-12-05..2026-06-22
  total quantity: 97,844
  last 30 days: 17,811 units, 31 sale days, 174 bakeries
```

Recent daily facts show that the non-`НОВЫЙ` SKU is now effectively legacy:
roughly 1-6 units/day at one bakery. The `НОВЫЙ` SKU is the network-active
product, roughly 437-711 units/day across about 147-166 bakeries on recent full
days.

The active forecast still contains both:

```text
000001743: 871.02 units across 189 bakeries for 2026-06-22..2026-07-05
000011301: 6,993.39 units across 181 bakeries for 2026-06-22..2026-07-05
```

## Open Decision

Review whether legacy SKU `000001743 / Клубника и банан` should be removed from
the global bakeable allowlist. Current facts indicate it should not be planned
network-wide, but no removal was made in this session because the business rule
has not yet been confirmed.

