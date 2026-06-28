# Current Project State

Last updated: 2026-06-28

## Summary

The production forecast writer is the VM only. VibeCode/Blackhole is a
read-only embedded UI/API over ClickHouse and must not run forecast generation.

## Production Source Of Truth

- Production VM: `root@201.51.7.24`
- VM path: `/opt/demand-forecasting-model`
- VM hostname observed: `msk-1-vm-tpez`
- VM systemd timer: `forecast-production.timer`
- VM timer schedule: daily `03:30:00 UTC`
- VM repo state observed: behind origin by docs/handoff only; production code
  was effectively current during the 2026-06-28 audit.

## Embedded App

- VibeCode server id: `82bb03a8-c356-4225-97a4-a1540cdc29e6`
- VibeCode server name: `bakery-forecast-embedded`
- VibeCode app URL: `https://app-8613ac40f10d.vibecode.bitrix24.tech`
- Mode: `BLACKHOLE`
- Role: read-only FastAPI/UI for Bitrix24 users.
- Forecast generation on VibeCode/Blackhole is forbidden.

## Active Forecast

- Active run on 2026-06-28: `prod_uplifted_bakery_norm_uplift_sku_20260623_h14`
- Horizon: `2026-06-23` through `2026-07-06`
- Scenario: `uplifted_norm`
- Horizon days: `14`
- Recent correction mode: `runner_city_prior_soft_weekpart`
- Recent correction days: `30`
- Recent sales table: `mart_sales_60d`

Expected active snapshot rows for the 2026-06-28 run:

- `bakery_forecast_day_snapshots`: `3024`
- `sku_forecast_day_snapshots`: `480206`
- `sku_forecast_hour_snapshots`: `5138172`

## Current Timers

Must be enabled and active:

- VM `forecast-production.timer`

Must remain disabled and inactive:

- Blackhole `forecast-production.timer`
- Blackhole `forecast-production.service`
- Blackhole `bakery-forecast-nightly.timer`
- Blackhole `bakery-forecast-nightly.service`

## Important Incident Fixed On 2026-06-28

The active ClickHouse run was being overwritten after the VM job by an old
Blackhole timer. The stale writer ran from VibeCode/Blackhole host
`fhmab3h2o3lo0jqd552k`, path `/opt/forecast_job`, and loaded:

- stale run: `prod_uplifted_bakery_norm_uplift_sku_20260601_h14`
- source IP in ClickHouse query log: `84.201.174.223`

Action taken:

- Re-activated fresh run `prod_uplifted_bakery_norm_uplift_sku_20260623_h14`.
- Disabled Blackhole `forecast-production.timer`.
- Verified VM timer remains active and ClickHouse active run is consistent.

## Verification Commands

On the VM:

```bash
cd /opt/demand-forecasting-model
systemctl is-enabled forecast-production.timer
systemctl is-active forecast-production.timer
systemctl list-timers --all --no-pager | grep forecast-production
.venv/bin/python -m scripts.verify_prod_deploy --env-file .env
```

Expected final line:

```text
VERIFY OK: env, summary, and active run are consistent
```

## Do Not Do

- Do not run production forecast generation from VibeCode/Blackhole.
- Do not enable Blackhole forecast timers.
- Do not treat `handoffs/` as current truth without checking this file first.
- Do not manually change active ClickHouse runs except through the documented
  activation script and only after verifying the intended run id.
- Do not print secrets from `.env`, ClickHouse config, VibeCode API keys, or
  VM SSH keys.

## When This File Must Be Updated

Update this file after any change to:

- production writer ownership;
- VM host, path, timer, or schedule;
- VibeCode/Blackhole role;
- ClickHouse active run contract;
- forecast scenario, horizon, correction mode, or source tables;
- emergency production state changes.
