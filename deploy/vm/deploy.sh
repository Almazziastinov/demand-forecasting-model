#!/usr/bin/env bash
#
# Production deploy for the forecast pipeline on the SSH VM.
#
# Runs as root (uses `sudo -u forecast` for repo/git operations, since root
# sees the repo as a dubious-ownership directory). Idempotent: safe to re-run.
#
# Steps:
#   1. git pull master (as the forecast user), report old->new commit
#   2. pip install if requirements-prod.txt changed
#   3. validate .env keys against the example
#   4. run the production inference service (systemctl start)
#   5. verify the resulting prod state (scripts/verify_prod_deploy.py)
#
# Usage:
#   sudo bash deploy/vm/deploy.sh                # pull + install + run + verify
#   sudo bash deploy/vm/deploy.sh --verify-only  # just verify current state
#   sudo bash deploy/vm/deploy.sh --no-run       # update code/.env, skip the run
#
set -euo pipefail

APP_DIR="/opt/demand-forecasting-model"
APP_USER="forecast"
PY="${APP_DIR}/.venv/bin/python"
SERVICE="forecast-production.service"
ENV_FILE="${APP_DIR}/.env"
ENV_EXAMPLE="${APP_DIR}/deploy/vm/forecast.env.example"

VERIFY_ONLY=0
RUN_SERVICE=1
for arg in "$@"; do
  case "$arg" in
    --verify-only) VERIFY_ONLY=1 ;;
    --no-run)      RUN_SERVICE=0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

run_as_app() { sudo -u "$APP_USER" "$@"; }

echo "=== forecast prod deploy ==="
cd "$APP_DIR"

if [[ "$VERIFY_ONLY" -eq 1 ]]; then
  echo "--- verify-only ---"
  run_as_app "$PY" -m scripts.verify_prod_deploy --env-file "$ENV_FILE"
  exit $?
fi

# --- 1. git pull ------------------------------------------------------------
OLD_REV="$(run_as_app git rev-parse --short HEAD)"
echo "--- git pull (current: $OLD_REV) ---"
REQ_HASH_BEFORE="$(sha1sum requirements-prod.txt 2>/dev/null | cut -d' ' -f1 || true)"
run_as_app git pull --ff-only origin master
NEW_REV="$(run_as_app git rev-parse --short HEAD)"
echo "commit: $OLD_REV -> $NEW_REV"

# --- 2. pip install if requirements changed ---------------------------------
REQ_HASH_AFTER="$(sha1sum requirements-prod.txt 2>/dev/null | cut -d' ' -f1 || true)"
if [[ "$REQ_HASH_BEFORE" != "$REQ_HASH_AFTER" ]]; then
  echo "--- requirements-prod.txt changed -> pip install ---"
  run_as_app "$PY" -m pip install -r requirements-prod.txt
else
  echo "--- requirements-prod.txt unchanged -> skip pip install ---"
fi

# --- 3. validate .env -------------------------------------------------------
echo "--- .env validation ---"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: $ENV_FILE missing" >&2
  exit 1
fi
missing=0
while IFS= read -r key; do
  [[ -z "$key" ]] && continue
  if ! grep -q "^${key}=" "$ENV_FILE"; then
    echo "ERROR: $ENV_FILE missing key: $key" >&2
    missing=1
  fi
done < <(grep -vE '^\s*#|^\s*$' "$ENV_EXAMPLE" | cut -d= -f1)
# flag duplicate keys (a duplicated FORECAST_RECENT_CORRECTION_MODE bit us once)
dupes="$(grep -vE '^\s*#|^\s*$' "$ENV_FILE" | cut -d= -f1 | sort | uniq -d || true)"
if [[ -n "$dupes" ]]; then
  echo "WARNING: duplicate keys in $ENV_FILE:" >&2
  echo "$dupes" >&2
fi
[[ "$missing" -eq 1 ]] && exit 1
echo ".env keys ok"

# --- 4. run the production inference service --------------------------------
if [[ "$RUN_SERVICE" -eq 1 ]]; then
  echo "--- starting $SERVICE (this takes ~9 min, memory-heavy) ---"
  systemctl start "$SERVICE"
  echo "service result:"
  systemctl show "$SERVICE" -p Result -p ExecMainStatus --no-pager
  RESULT="$(systemctl show "$SERVICE" -p Result --value)"
  if [[ "$RESULT" != "success" ]]; then
    echo "ERROR: service did not succeed (Result=$RESULT)" >&2
    journalctl -u "$SERVICE" -n 40 --no-pager >&2
    exit 1
  fi
else
  echo "--- --no-run: skipping service start ---"
fi

# --- 5. verify --------------------------------------------------------------
echo "--- verify prod state ---"
run_as_app "$PY" -m scripts.verify_prod_deploy --env-file "$ENV_FILE"

echo "=== deploy complete: $OLD_REV -> $NEW_REV ==="
