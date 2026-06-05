# Session Handoff - 2026-06-05 - Recent SKU Correction Production Rollout

## Context

Continuation of the SKU holdout/backtest work and the committed production code:

- `69ae7d2 feat: add recent sku allocation correction`
- `src/experiments_v2/apply_bakery_profiles_clickhouse.py`
- `pipelines/forecast_publish/run_production_inference.py`

The goal was to finish production rollout of the recent SKU allocation correction
after the holdout analysis showed large SKU-level error from stale/dead profile
buckets.

The selected production mode is:

```text
FORECAST_RECENT_CORRECTION_MODE=blend_recent_50
FORECAST_RECENT_CORRECTION_DAYS=30
FORECAST_RECENT_SALES_TABLE=mart_sales_60d
```

This keeps bakery-day totals unchanged and only redistributes SKU shares using
recent assortment facts.

## Production VM

VM:

```text
host: 89.108.76.196
path: /opt/demand-forecasting-model
service: forecast-production.service
timer: forecast-production.timer
user: forecast
```

The service reads `/opt/demand-forecasting-model/.env`:

```text
EnvironmentFile=/opt/demand-forecasting-model/.env
ExecStart=/opt/demand-forecasting-model/.venv/bin/python -m pipelines.forecast_publish.run_production_inference ...
```

The final `.env` production scheduling values are:

```text
FORECAST_SCENARIO=uplifted_norm
FORECAST_ACTIVATE_RUN=uplifted_norm
FORECAST_RECENT_CORRECTION_MODE=blend_recent_50
FORECAST_RECENT_CORRECTION_DAYS=30
FORECAST_RECENT_SALES_TABLE=mart_sales_60d
```

Note: `.env` currently contains a harmless duplicate
`FORECAST_RECENT_CORRECTION_MODE=blend_recent_50`. The duplicate has the same
value and does not affect behavior, but can be cleaned later.

## What Changed Operationally

### 1. Git Pull on VM

Git commands must be run as `forecast`, not `root`, because root sees the repo
as dubious ownership.

```bash
cd /opt/demand-forecasting-model
sudo -u forecast git status --short --branch
sudo -u forecast git pull origin master
sudo -u forecast git rev-parse --short HEAD
```

Expected code version:

```text
69ae7d2
```

### 2. Timer Scenario Narrowed

The previous setting was:

```text
FORECAST_SCENARIO=both
```

This was changed to:

```text
FORECAST_SCENARIO=uplifted_norm
```

Reason: production only activates `uplifted_norm`; computing `base_raw_uplift`
is unnecessary for the scheduled production path and increases memory pressure.

### 3. Swap Added

The VM has only about 1.9 GiB RAM and no swap by default:

```text
Mem:  1.9Gi
Swap: 0B
```

The first systemd run with recent correction was OOM-killed:

```text
Result: oom-kill
Mem peak: 1.5G
```

The reason is that the new recent correction does a post-allocation pass over
the generated SKU-hour forecast. The old path was mostly streaming/chunked; the
new correction uses larger in-memory merge/groupby steps.

Added 4 GiB swap:

```bash
fallocate -l 4G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

Verified:

```text
Swap: 4.0Gi
/swapfile file 4G
```

### 4. Manual Production Run

After narrowing the scenario and adding swap, this succeeded:

```bash
systemctl start forecast-production.service
```

Systemd status:

```text
Active: inactive (dead)
status=0/SUCCESS
Mem peak: 1.5G
swap peak: 1.3G
```

The run completed at about:

```text
2026-06-05 09:46:57 UTC
2026-06-05 12:46:57 MSK
```

## Active Run After Rollout

Active run:

```text
prod_uplifted_bakery_norm_uplift_sku_20260601_h14
status: active
horizon: 2026-06-01..2026-06-14
generated_at: 2026-06-05 12:46:49 MSK
notes: uplifted bakery forecast + normalized uplift SKU allocation
```

The notes are generic because the systemd service does not pass a custom
`--notes`; the authoritative confirmation is the production summary and row
counts below.

`reports/production_inference_summary.json` confirms:

```json
{
  "recent_correction_mode": "blend_recent_50",
  "recent_correction_days": 30,
  "recent_sales_table": "mart_sales_60d",
  "scenarios": [
    {
      "scenario": "uplifted_norm",
      "run_id": "prod_uplifted_bakery_norm_uplift_sku_20260601_h14",
      "loaded_rows": {
        "bakery_rows": 3038,
        "context_rows": 154,
        "sku_day_rows": 576806,
        "sku_hour_rows": 5575452
      },
      "activated": true
    }
  ]
}
```

Serving table verification for
`prod_uplifted_bakery_norm_uplift_sku_20260601_h14`:

```text
bakery_forecast_day_embedded
  rows: 3038
  bakeries: 217
  date range: 2026-06-01..2026-06-14
  forecast_sum: 3.291938e+06

sku_forecast_day_embedded
  rows: 576806
  bakeries: 217
  date range: 2026-06-01..2026-06-14
  forecast_sum: 3.291938e+06

sku_forecast_hour_embedded
  rows: 5575452
  bakeries: 217
  date range: 2026-06-01..2026-06-14
  forecast_sum: 3.291938e+06
```

This confirms bakery totals are preserved and SKU day/hour totals match.

## Timer State

Timer was restarted after the successful manual run:

```bash
systemctl start forecast-production.timer
systemctl status forecast-production.timer --no-pager
```

Final timer state:

```text
Active: active (waiting)
next trigger: Sat 2026-06-06 03:30:00 UTC
```

That is 06:30 Moscow time.

## Tomorrow Morning Check

After 2026-06-06 06:30 MSK, verify:

```bash
cd /opt/demand-forecasting-model

sudo -u forecast .venv/bin/python - <<'PY'
from pipelines.forecast_publish.load_forecast_run import create_client

client = create_client(".env")
print(client.query_df("""
select
  run_id,
  status,
  horizon_start,
  horizon_end,
  generated_at,
  notes
from forecast_runs_embedded
where status = 'active'
order by generated_at desc
""").to_string(index=False))
PY
```

Also check the service log:

```bash
systemctl status forecast-production.service --no-pager
journalctl -u forecast-production.service -n 100 --no-pager
```

Expected:

- service finishes with `status=0/SUCCESS`;
- active run has fresh `generated_at` on 2026-06-06 morning;
- production summary still shows `recent_correction_mode = blend_recent_50`;
- SKU day/hour row counts stay in the new higher range, not the old 425,604 /
  3,096,400 range.

## Operational Notes

1. Keep `FORECAST_SCENARIO=uplifted_norm` for scheduled production unless
   there is a specific need to publish the base draft run.
2. Do not remove swap unless the correction code is optimized or VM RAM is
   increased.
3. The current active production data is already corrected; the timer setup is
   now aligned so it should not revert to old allocation tomorrow.
4. The code should eventually be optimized to avoid loading the full SKU-hour
   forecast into memory during recent correction.

