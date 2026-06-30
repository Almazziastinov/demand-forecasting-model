# Runbook

Last updated: 2026-06-28

## Verify Production State

Run on the production VM:

```bash
cd /opt/demand-forecasting-model
systemctl is-enabled forecast-production.timer
systemctl is-active forecast-production.timer
systemctl list-timers --all --no-pager | grep forecast-production
.venv/bin/python -m scripts.verify_prod_deploy --env-file .env
```

Expected:

- timer is `enabled`;
- timer is `active`;
- verify command ends with `VERIFY OK`.

## Activate A Known Good Run

Use this only when the intended run id is known and verified:

```bash
cd /opt/demand-forecasting-model
.venv/bin/python -m pipelines.forecast_publish.activate_run \
  --env-file .env \
  --run-id prod_uplifted_bakery_norm_uplift_sku_YYYYMMDD_h14
```

After activation, run:

```bash
.venv/bin/python -m scripts.verify_prod_deploy --env-file .env
```

## Check VM Production Timer

```bash
systemctl status forecast-production.timer --no-pager -l
systemctl status forecast-production.service --no-pager -l
journalctl -u forecast-production.service -n 120 --no-pager -o short-iso
```

## Check Blackhole Timers

Use VibeCode `/v1/infra/servers/:id/exec?stream=false` with server id
`82bb03a8-c356-4225-97a4-a1540cdc29e6`.

Command to run on Blackhole:

```bash
systemctl is-enabled forecast-production.timer 2>&1 || true
systemctl is-active forecast-production.timer 2>&1 || true
systemctl is-active forecast-production.service 2>&1 || true
systemctl is-enabled bakery-forecast-nightly.timer 2>&1 || true
systemctl is-active bakery-forecast-nightly.timer 2>&1 || true
systemctl list-timers --all --no-pager | grep -E 'bakery|forecast' || true
```

Expected:

- `forecast-production.timer`: `disabled`, `inactive`
- `forecast-production.service`: `inactive`
- `bakery-forecast-nightly.timer`: `disabled`, `inactive`

## Disable Blackhole Forecast Writers

Use only if a Blackhole forecast timer is active or enabled:

```bash
systemctl disable --now forecast-production.timer
systemctl stop forecast-production.service 2>/dev/null || true
systemctl disable --now bakery-forecast-nightly.timer
systemctl stop bakery-forecast-nightly.service 2>/dev/null || true
systemctl reset-failed forecast-production.service forecast-production.timer \
  bakery-forecast-nightly.service bakery-forecast-nightly.timer 2>/dev/null || true
```

Then verify ClickHouse active run from the VM.

## Investigate Unexpected Active Run Changes

1. Verify the current active run from the VM.
2. Check VM service logs for the expected run id.
3. Check ClickHouse query log for external writer IPs around the time the active
   run changed.
4. Check Blackhole timers and logs.
5. Re-activate the known good run only after identifying the writer.

Known stale writer from the 2026-06-28 incident:

- Blackhole host: `fhmab3h2o3lo0jqd552k`
- ClickHouse client IP: `84.201.174.223`
- stale run id: `prod_uplifted_bakery_norm_uplift_sku_20260601_h14`

## Build Missing Lead-1 Backfill

Use this when facts exist for historical dates, but `lead_days = 1` snapshots
are missing for fact-vs-forecast comparison.

Run on the production VM only:

```bash
cd /opt/demand-forecasting-model
.venv/bin/python scripts/build_prod_lead1_model_backfill.py \
  --env-file .env \
  --date-from YYYY-MM-DD \
  --date-to YYYY-MM-DD \
  --uplift-profile-version prod_allowlist_22_222_old_else_20260617 \
  --replace-existing
```

The script creates draft runs named:

```text
backfill_uplifted_bakery_norm_uplift_sku_YYYYMMDD_h1
```

Do not activate these runs. The active production forecast remains the current
`prod_uplifted_bakery_norm_uplift_sku_YYYYMMDD_h14` run.

Verification query:

```bash
.venv/bin/python - <<'PY'
from pipelines.forecast_publish.load_forecast_run import create_client
c = create_client(".env")
for table in [
    "bakery_forecast_day_snapshots",
    "sku_forecast_day_snapshots",
    "sku_forecast_hour_snapshots",
]:
    df = c.query_df(f"""
    select forecast_date, count() rows, uniqExact(source_run_id) runs
    from {table}
    where lead_days = 1
      and forecast_date between 'YYYY-MM-DD' and 'YYYY-MM-DD'
    group by forecast_date
    order by forecast_date
    """)
    print("\\n" + table)
    print(df.to_string(index=False))
PY
```

## Update Ops Docs After Incidents

After any production incident:

1. Update `CURRENT_STATE.md`.
2. Add a durable decision to `DECISIONS.md` if ownership or architecture changed.
3. Add reusable commands to this runbook.
4. Leave handoffs as session history, not as the primary state record.
