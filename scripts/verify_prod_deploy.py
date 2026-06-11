"""Verify the production forecast deploy state.

Reads the production inference summary, queries the active run from ClickHouse,
and cross-checks the recent-correction mode against the .env. Exits non-zero if
anything looks inconsistent so it can gate a deploy script.

Usage:
    .venv/bin/python -m scripts.verify_prod_deploy --env-file .env
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = ROOT / ".env"
DEFAULT_SUMMARY_PATH = ROOT / "reports" / "production_inference_summary.json"

ENV_MODE_KEY = "FORECAST_RECENT_CORRECTION_MODE"
ENV_REFRESH_KEY = "FORECAST_REFRESH_DATASETS"


def read_env_value(env_path: Path, key: str) -> str | None:
    """Return the last value for a key from .env, or None if absent.

    If the key appears more than once we flag it: duplicate .env keys have
    previously caused silent production confusion.
    """
    if not env_path.exists():
        return None
    values = []
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(f"{key}="):
            values.append(line.split("=", 1)[1].strip())
    if len(values) > 1:
        print(f"WARNING: {key} appears {len(values)} times in {env_path}")
    return values[-1] if values else None


def _env_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify production forecast deploy state"
    )
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH))
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    args = parser.parse_args()

    env_path = Path(args.env_file)
    summary_path = Path(args.summary_path)
    problems: list[str] = []

    # --- .env mode -------------------------------------------------------
    env_mode = read_env_value(env_path, ENV_MODE_KEY)
    env_refresh = read_env_value(env_path, ENV_REFRESH_KEY)
    print(f".env {ENV_MODE_KEY} = {env_mode}")
    print(f".env {ENV_REFRESH_KEY} = {env_refresh}")

    # --- summary json ----------------------------------------------------
    summary_mode = None
    summary_refresh = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_mode = summary.get("recent_correction_mode")
        summary_refresh = summary.get("dataset_refresh")
        print(f"summary mode = {summary_mode}")
        print(f"summary days = {summary.get('recent_correction_days')}")
        print(f"summary table = {summary.get('recent_sales_table')}")
        if summary_refresh:
            print(
                "summary dataset_refresh = "
                f"{summary_refresh.get('history_start_date')}.."
                f"{summary_refresh.get('history_end_date')} | "
                f"base={summary_refresh.get('base_dataset_path')} | "
                f"uplifted={summary_refresh.get('uplifted_dataset_path')}"
            )
        else:
            print("summary dataset_refresh = none")
        for s in summary.get("scenarios", []):
            rows = s.get("loaded_rows", {})
            print(
                f"  scenario {s.get('scenario')} | {s.get('run_id')} "
                f"| activated={s.get('activated')} "
                f"| sku_day={rows.get('sku_day_rows')} "
                f"sku_hour={rows.get('sku_hour_rows')}"
            )
            if not s.get("activated"):
                problems.append(f"scenario {s.get('scenario')} was not activated")
    else:
        problems.append(f"summary not found: {summary_path}")

    # --- env vs summary consistency -------------------------------------
    if env_mode and summary_mode and env_mode != summary_mode:
        problems.append(
            f"mode mismatch: .env={env_mode} but last run used {summary_mode} "
            "(run may predate the .env change -- re-run the service)"
        )
    if _env_bool(env_refresh) and not summary_refresh:
        problems.append(
            f"{ENV_REFRESH_KEY} is enabled but last summary has no dataset_refresh "
            "(run may predate the refresh change -- re-run the service)"
        )

    # --- active run in ClickHouse ---------------------------------------
    try:
        from pipelines.forecast_publish.load_forecast_run import create_client

        client = create_client(args.env_file)
        df = client.query_df(
            """
            select run_id, status, horizon_start, horizon_end, generated_at, notes
            from forecast_runs_embedded
            where status = 'active'
            order by generated_at desc
            """
        )
        if df.empty:
            problems.append("no active run found in forecast_runs_embedded")
        else:
            print("\nactive run(s):")
            print(df.to_string(index=False))
    except Exception as exc:  # pragma: no cover - network/credentials dependent
        problems.append(f"could not query active run: {exc}")

    # --- verdict ---------------------------------------------------------
    print("\n" + "=" * 60)
    if problems:
        print("VERIFY FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("VERIFY OK: env, summary, and active run are consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
