#!/usr/bin/env bash
# Deploy the daily pilot forecast publisher as a systemd timer on Blackhole VM.
# Run once on the VM: bash scripts/deploy_pilot_forecast_timer.sh
# Prereq: BITRIX24_WEBHOOK and BITRIX24_CHAT_ID must be set in /opt/app/.env
set -euo pipefail

SCRIPT_SRC="/opt/scripts/publish_pilot_forecast.py"
ENV_FILE="/opt/app/.env"
PYTHON="/opt/app/.venv/bin/python"
LOG_DIR="/var/log/pilot_forecast"

mkdir -p "$(dirname "$SCRIPT_SRC")"
mkdir -p "$LOG_DIR"

# Ensure VIBECODE_API_KEY is in .env (must be added manually before running this script)
if ! grep -q "VIBECODE_API_KEY" "$ENV_FILE"; then
    echo "WARNING: VIBECODE_API_KEY not found in $ENV_FILE — Bitrix24 publish will be skipped."
    echo "Add it manually: echo 'VIBECODE_API_KEY=<key>' >> $ENV_FILE"
fi

# Copy the script from repo
cp /opt/demand-forecasting-model/scripts/publish_pilot_forecast.py "$SCRIPT_SRC" 2>/dev/null || true

# --- systemd service unit ---
cat > /etc/systemd/system/pilot-forecast-publish.service << 'EOF'
[Unit]
Description=Publish daily pilot baking forecast to Bitrix24
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/opt
Environment="PYTHONPATH=/opt:/opt/app"
Environment="ENV_FILE=/opt/app/.env"
ExecStart=/opt/app/.venv/bin/python /opt/scripts/publish_pilot_forecast.py --env-file /opt/app/.env
StandardOutput=append:/var/log/pilot_forecast/publish.log
StandardError=append:/var/log/pilot_forecast/publish.log
EOF

# --- systemd timer unit (06:00 Moscow time = 03:00 UTC) ---
cat > /etc/systemd/system/pilot-forecast-publish.timer << 'EOF'
[Unit]
Description=Daily pilot baking forecast publish at 06:00 MSK

[Timer]
OnCalendar=*-*-* 03:00:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
systemctl enable --now pilot-forecast-publish.timer
systemctl status pilot-forecast-publish.timer --no-pager

echo "=== Timer deployed. Next run: ==="
systemctl list-timers pilot-forecast-publish.timer --no-pager
