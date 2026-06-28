# Services

Last updated: 2026-06-28

## Service Ownership Matrix

| Service | Environment | Role | May write forecast runs? | Status |
| --- | --- | --- | --- | --- |
| Production forecast VM | `201.51.7.24` | Generates and publishes forecasts | Yes | Active |
| ClickHouse | External database | Serving tables and snapshots | N/A | Active |
| VibeCode/Blackhole app | `bakery-forecast-embedded` | Embedded read-only API/UI | No | Active |
| Legacy Flask app | `web/app.py` | Local/demo legacy app | No prod role | Legacy |

## Production Forecast VM

- SSH target: `root@201.51.7.24`
- Repo path: `/opt/demand-forecasting-model`
- Python env: `/opt/demand-forecasting-model/.venv`
- Env file: `/opt/demand-forecasting-model/.env`
- Primary command:

```bash
.venv/bin/python -m pipelines.forecast_publish.run_production_inference \
  --env-file .env
```

The systemd unit expands the production settings from `.env` and command-line
flags. See `CURRENT_STATE.md` for the current scenario and verification command.

## VibeCode / Blackhole

- Server id: `82bb03a8-c356-4225-97a4-a1540cdc29e6`
- Server name: `bakery-forecast-embedded`
- App URL: `https://app-8613ac40f10d.vibecode.bitrix24.tech`
- Runtime role: serve the embedded FastAPI/UI from `/opt/app`.
- Historical forecast job path: `/opt/forecast_job`

The historical `/opt/forecast_job` tree may still exist. It must not be treated
as the production writer. Forecast timers there must stay disabled.

## ClickHouse

ClickHouse is the production serving store. The forecast writer publishes run
metadata and forecast snapshots there. The embedded app reads from those tables.

Known serving/snapshot tables include:

- `forecast_runs_embedded`
- `bakery_forecast_day_embedded`
- `forecast_day_context_embedded`
- `sku_forecast_day_embedded`
- `sku_forecast_hour_embedded`
- `bakery_forecast_day_snapshots`
- `sku_forecast_day_snapshots`
- `sku_forecast_hour_snapshots`

## Local Development

Use the repo root on the workstation for code changes. Do not infer production
state from local generated files without checking VM and ClickHouse.

Useful commands:

```bash
ruff check src/ web/ tests/ --select=E,F,W
pytest tests/ -v
```
