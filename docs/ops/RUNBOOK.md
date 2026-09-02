# Runbook

Last updated: 2026-09-01

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
- exactly one active run has pattern `prod_direct_alpha_025_*` and its notes
  identify the inactive `prod_base_bakery_norm_recent_*` source run.

Do not identify `base_norm_recent` as the current model merely because it is
shown in the refresh summary. It is the inactive bakery-volume/source stage;
the active Direct run is the served forecast.

## Check Pilot Management Statistics Refresh

Run on the production VM:

```bash
systemctl status pilot-management-report.timer --no-pager -l
systemctl status pilot-management-report.service --no-pager -l
systemctl list-timers --all --no-pager | grep pilot-management-report
journalctl -u pilot-management-report.service -n 120 --no-pager -o short-iso
```

Expected: the timer is enabled/active with the next trigger at `05:00 UTC`.
The service is normally `inactive (dead)` after a successful oneshot and its
last result is `status=0/SUCCESS`.

Manual refresh:

```bash
systemctl start pilot-management-report.service
journalctl -u pilot-management-report.service -n 80 --no-pager
```

The builder validates every date from `2026-07-23` through Moscow yesterday
before upload. Blackhole repeats date validation before atomically replacing
`/opt/reports/pilot_management_summary`; if validation or upload fails, the
currently served report remains unchanged. A successful swap keeps a backup
named `/opt/backups/pilot_management_summary_before_YYYYMMDD_HHMMSS`.

To stop only statistics refresh without changing forecast generation:

```bash
systemctl disable --now pilot-management-report.timer
```

## Activate A Known Good Run

Use this only when the intended run id is known and verified:

```bash
cd /opt/demand-forecasting-model
.venv/bin/python -m pipelines.forecast_publish.activate_run \
  --env-file .env \
  --run-id prod_direct_alpha_025_YYYYMMDD_h14
```

After activation, run:

```bash
.venv/bin/python -m scripts.verify_prod_deploy --env-file .env
```

Manual activation is recovery-only. Normal nightly publication is performed by
the Direct `ExecStartPost`; do not activate the intermediate
`prod_base_bakery_norm_recent_*` run during an ordinary successful cycle.

## Check VM Production Timer

```bash
systemctl status forecast-production.timer --no-pager -l
systemctl status forecast-production.service --no-pager -l
journalctl -u forecast-production.service -n 120 --no-pager -o short-iso
```

For the active Direct alpha=.25 architecture, the inactive legacy SKU source
stage must not block the nightly run on hourly-profile age. Verify the
production environment contains:

```bash
grep '^FORECAST_PROFILE_MAX_AGE_DAYS=' .env
```

Expected: `FORECAST_PROFILE_MAX_AGE_DAYS=-1`. This setting is valid only while
the served model is Direct and the legacy SKU allocation remains an inactive
source-stage artifact. The assortment freshness and coverage guards remain
enabled.

If a pilot workbook shows `нет данных по остатку`, check the preceding day's
production-event completeness before treating the value as zero. A bakery
with positive sales and no recorded production has no reconstructable opening
inventory in the current event model; the publisher intentionally avoids
subtracting an invented stock value.

The pilot publisher's observable closing-stock formula is:

```text
max(produced + received - sent - sold - written_off, 0)
```

The inputs come directly from deduplicated `fct_production_release`,
`fct_moves`, `fct_check_lines`, and `fct_write_offs` for the preceding date.
This is not an authoritative opening-balance ledger: stock carried into that
preceding date is unavailable in the current source contract.

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

The separate read-only `pilot-forecast-publish.timer` is expected to be
enabled and active. When changing its schedule after the current day's old
slot has passed, do not restart a timer with `Persistent=true` directly: that
causes an immediate catch-up publication. Stop the timer first and ensure its
next calculated trigger is in the future before starting it again.

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
`prod_direct_alpha_025_YYYYMMDD_h14` run.

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
