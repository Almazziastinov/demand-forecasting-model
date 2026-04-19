#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"

mkdir -p "$LOG_DIR"

cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  echo "Missing virtual environment: $ROOT_DIR/.venv" >&2
  exit 1
fi

source "$ROOT_DIR/.venv/bin/activate"

echo "[start] monthly benchmarks: $(date -Iseconds)"
echo "[info] root: $ROOT_DIR"

echo "[run] full_benchmark_monthly"
python src/experiments_v2/full_benchmark_monthly/run.py \
  2>&1 | tee "$LOG_DIR/full_benchmark_monthly.log"

echo "[run] sku_local_monthly"
python src/experiments_v2/sku_local_monthly/run.py --test-days 30 --min-train-rows 30 \
  2>&1 | tee "$LOG_DIR/sku_local_monthly.log"

echo "[done] monthly benchmarks: $(date -Iseconds)"
