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

## Update Ops Docs After Incidents

After any production incident:

1. Update `CURRENT_STATE.md`.
2. Add a durable decision to `DECISIONS.md` if ownership or architecture changed.
3. Add reusable commands to this runbook.
4. Leave handoffs as session history, not as the primary state record.
