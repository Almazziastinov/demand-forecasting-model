# Session Handoff - 2026-06-10 - City Prior SKU Guard Prod Deploy

## What was done

Deployed the city-prior SKU allocation guard (`runner_city_prior_soft_weekpart`)
to the production SSH VM. The code was already committed/pushed (`d33387c`); this
session brought it live on the VM and made it stick across timer runs.

## VM

```text
host: 89.108.76.196
path: /opt/demand-forecasting-model
service: forecast-production.service
timer:   forecast-production.timer
user:    forecast (run git as `sudo -u forecast`, root sees repo as dubious)
```

## Key finding: why the mode had to change in .env

- `run_production_inference.py` default mode is now `runner_city_prior_soft_weekpart`
  (line ~41), read from `os.getenv("FORECAST_RECENT_CORRECTION_MODE", ...)`.
- The systemd `ExecStart` does NOT pass `--recent-correction-mode`, so the mode
  comes entirely from `.env`.
- `.env` was pinning `FORECAST_RECENT_CORRECTION_MODE=blend_recent_50` (twice).
- Therefore: editing `.env` was the correct lever. Without it, the nightly timer
  would keep running the old `blend_recent_50` mode.

## Changes made on the VM

1. Backed up `.env` -> `.env.bak_20260610`.
2. Replaced the mode and removed the duplicate line. Final:
   ```text
   FORECAST_RECENT_CORRECTION_MODE=runner_city_prior_soft_weekpart
   ```
3. Ran the job manually: `systemctl start forecast-production.service`.

## Result (verified)

- Service: `status=0/SUCCESS`, ~9 min wall, Mem peak 1.6G (swap 1.4G).
- `reports/production_inference_summary.json`:
  ```text
  mode = runner_city_prior_soft_weekpart
  days = 30, table = mart_sales_60d
  ```
- SKU row counts changed vs old mode (correction really applied):
  ```text
  sku_day  576806 -> 555404
  sku_hour 5575452 -> 5221880
  ```
- Active run in ClickHouse:
  ```text
  prod_uplifted_bakery_norm_uplift_sku_20260601_h14
  status: active
  generated_at: 2026-06-10 16:34 MSK
  ```
- Timer: `active (waiting)`, next trigger 2026-06-11 03:30 UTC (06:30 MSK),
  will now use the new mode from `.env`.

## Notes / caveats

- SSH dropped during `systemctl start` (memory pressure killed the ssh session),
  but the oneshot kept running under systemd and finished successfully. The 4 GiB
  swap is required — do not remove it.
- The new mode is slightly heavier on memory than `blend_recent_50`
  (peak 1.6G + 1.4G swap). Keep an eye on this if VM specs change.
- Run notes in the DB say "uplifted bakery forecast + normalized uplift SKU
  allocation" (generic) because the service passes no `--notes`; the
  authoritative mode confirmation is the summary json above, not the notes.
- Active run horizon ends 2026-06-14 (~4 days left). A fresh-horizon run will be
  needed soon.

## VM deploy invariant (do not forget)

- Forecast pipeline lives ONLY on this SSH VM, NOT VibeCode.
- VibeCode = embedded frontend only.
