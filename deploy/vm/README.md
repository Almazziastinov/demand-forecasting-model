# VM production setup

This deployment path runs the lightweight production inference/publish job on a
small VM. Heavy profile rebuilds and model training stay off-box.

Expected project path:

```bash
/opt/demand-forecasting-model
```

Required local artifacts on the VM:

```text
data/processed/bakery_daily_sales.csv
data/processed/bakery_daily_sales_uplifted.csv
data/processed/bakery_hour_profile.csv
models/bakery_day_model.joblib
models/bakery_day_meta.joblib
models/bakery_day_model_uplifted.joblib
models/bakery_day_meta_uplifted.joblib
models/bakery_day_bias.json
models/bakery_day_bias_uplifted.json
```

The large SKU hour profile CSV is not required on the VM. SKU profiles and
product/category lookup are read from ClickHouse.

## Bootstrap

```bash
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv git
sudo useradd --system --create-home --shell /usr/sbin/nologin forecast
sudo mkdir -p /opt/demand-forecasting-model
sudo chown -R forecast:forecast /opt/demand-forecasting-model
```

Clone or copy the repository into `/opt/demand-forecasting-model`, then:

```bash
cd /opt/demand-forecasting-model
sudo -u forecast python3.12 -m venv .venv
sudo -u forecast .venv/bin/python -m pip install --upgrade pip
sudo -u forecast .venv/bin/python -m pip install -r requirements-prod.txt
```

Create `.env` from `deploy/vm/forecast.env.example` and fill ClickHouse
credentials.

## Smoke checks

```bash
cd /opt/demand-forecasting-model
sudo -u forecast .venv/bin/python -m pipelines.forecast_publish.run_production_inference --help
sudo -u forecast .venv/bin/python -m pipelines.forecast_publish.run_production_inference --env-file .env --scenario uplifted_norm --horizon-days 14 --uplift-profile-version sku_uplift_20260601
```

## systemd

```bash
sudo cp deploy/vm/forecast-production.service /etc/systemd/system/
sudo cp deploy/vm/forecast-production.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start forecast-production.service
sudo journalctl -u forecast-production.service -n 100 --no-pager
```

Enable daily scheduled publishing only after the manual run is clean:

```bash
sudo systemctl enable --now forecast-production.timer
systemctl list-timers forecast-production.timer
```

## Embedded API

Install/update runtime dependencies after pulling app changes:

```bash
cd /opt/demand-forecasting-model
sudo -u forecast .venv/bin/python -m pip install -r requirements-prod.txt
```

Start the read-only API/UI:

```bash
sudo cp deploy/vm/forecast-embedded-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now forecast-embedded-api.service
sudo journalctl -u forecast-embedded-api.service -n 100 --no-pager
```

Smoke endpoints:

```bash
curl http://127.0.0.1:3000/health
curl http://127.0.0.1:3000/api/v1/runs/active
curl http://127.0.0.1:3000/api/v1/runs
```
