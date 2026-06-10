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

## Deploy automation added (commit 2794678)

To replace the manual SSH deploy flow:

- `deploy/vm/deploy.sh` — idempotent wrapper: pull master (as forecast user),
  conditional pip install, `.env` key/duplicate validation, run the production
  service, fail on non-success, then verify. Modes: default / `--verify-only`
  / `--no-run`.
- `scripts/verify_prod_deploy.py` — cross-checks `.env` mode vs summary vs the
  active ClickHouse run; flags duplicate `FORECAST_RECENT_CORRECTION_MODE`.
- `.gitattributes` — forces LF on `*.sh` so the shebang works on the VM.

First real use on the VM should be the safe read-only check:

```bash
cd /opt/demand-forecasting-model
sudo -u forecast git pull --ff-only origin master
sudo bash deploy/vm/deploy.sh --verify-only
```

## Past-days backfill: NOT needed (verified)

User asked whether past days (1-9 Jun) needed backfilling into ClickHouse so the
frontend shows new-model predictions for them. Investigation showed it is
already done by the 16:34 run:

- Frontend selects exactly ONE active run, then filters every query by that
  `run_id` + `forecast_date` (see `apps/forecast_embedded/app/services/runs.py`,
  `bakery.py`). Data must live under the active run_id to be visible.
- The active run `prod_uplifted_bakery_norm_uplift_sku_20260601_h14` already
  covers `2026-06-01..2026-06-14` (14 days, 3038 bakery-day rows).
- API confirms:
  ```text
  GET /api/v1/runs/active -> run_id ...20260601_h14, horizon 2026-06-01..06-14
  GET /api/v1/dates       -> ["2026-06-01",...,"2026-06-14"]   (all past days present)
  ```

Conclusion: no backfill required. If the frontend still shows stale/empty past
days, the cause is the Bitrix iframe cache or the week selector in the UI, not
missing data. Fix = reopen the app from the Bitrix menu / pick the right week.

## How the forecast horizon is chosen (and a known limitation)

The service runs without `--start-date`, so the forecast window is derived from
the dataset, NOT the calendar:

```python
# src/experiments_v2/bakery_day_forecast.py:831 (recursive_forecast)
next_date = pd.Timestamp(start_date) if start_date is not None \
            else history[DATE_COL].max() + pd.Timedelta(days=1)
# then horizon_days (14) steps forward
```

- `run_id` is built from the min forecast date + horizon
  (`_build_run_id`, run_production_inference.py:84).
- Dataset on the VM (`data/processed/bakery_daily_sales_uplifted.csv`) currently
  ends at **2026-05-31**, so forecast start = 2026-06-01, horizon
  2026-06-01..06-14, run_id `...20260601_h14`.

### What tomorrow's 06:30 MSK timer run will do

Because the dataset is static (max date stays 2026-05-31):

- same start (2026-06-01), same horizon (..06-14), same run_id `...20260601_h14`;
- `replace_existing=True` -> data overwritten with the same city-prior numbers
  (model + data unchanged);
- stays activated. Nothing shifts, nothing breaks. Past days 1-9 Jun remain
  visible.

### KNOWN LIMITATION (not addressed this session)

The dataset is NOT auto-refreshed with recent sales facts. Consequences:

- the horizon is "frozen" at 2026-06-01..06-14;
- days 1-9 Jun are "forecasts" of dates already in the past (actuals known);
- only 11-14 Jun is real future, and it is predicted from data ~10 days stale.
- IMPORTANT: once the dataset is ever updated past 9 Jun, the new run will start
  its horizon later and the OLD past days will drop out of the active run (the
  frontend shows only one active run_id).

Future work: add a step that refreshes `bakery_daily_sales_uplifted.csv` from
ClickHouse before the inference run, so the horizon is always "from today
forward". No such step exists in the pipeline today.

## VM deploy invariant (do not forget)

- Forecast pipeline lives ONLY on this SSH VM, NOT VibeCode.
- VibeCode = embedded frontend only.
