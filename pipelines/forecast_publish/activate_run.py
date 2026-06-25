from __future__ import annotations

# ruff: noqa: E501
import argparse
import time
from pathlib import Path

import clickhouse_connect
import pandas as pd

from pipelines.forecast_publish.table_names import (
    get_table_suffix_from_env_file,
    table_name,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = ROOT / ".env"
RUNS_TABLE = "forecast_runs_embedded"


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_env_file(path: str | Path = DEFAULT_ENV_PATH) -> dict[str, str]:
    env: dict[str, str] = {}
    file_path = Path(path)
    if not file_path.exists():
        return env

    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def get_clickhouse_settings(env_path: str | Path = DEFAULT_ENV_PATH) -> dict[str, object]:
    env = load_env_file(env_path)
    return {
        "host": env.get("CLICKHOUSE_HOST") or env.get("HOST"),
        "port": int(env.get("CLICKHOUSE_PORT") or env.get("PORT") or "8443"),
        "username": env.get("CLICKHOUSE_USER") or env.get("USER"),
        "password": env.get("CLICKHOUSE_PASSWORD") or env.get("PASSWORD"),
        "database": env.get("CLICKHOUSE_DATABASE") or env.get("DATABASE"),
        "secure": _as_bool(env.get("CLICKHOUSE_SECURE") or env.get("SECURE"), default=True),
        "verify": _as_bool(env.get("CLICKHOUSE_VERIFY") or env.get("VERIFY"), default=False),
    }


def create_client(env_path: str | Path):
    settings = get_clickhouse_settings(env_path)
    return clickhouse_connect.get_client(
        host=settings["host"],
        port=int(settings["port"]),
        username=settings["username"],
        password=settings["password"],
        database=settings["database"],
        secure=bool(settings["secure"]),
        verify=bool(settings["verify"]),
    )


def fetch_run(client, run_id: str, table_suffix: str = "") -> dict | None:
    runs_table = table_name(RUNS_TABLE, table_suffix)
    df = client.query_df(
        f"""
        select run_id, model_version, profile_version, source_kind,
               horizon_start, horizon_end, generated_at, status, notes, is_bias_adjusted
        from {runs_table}
        where run_id = %(run_id)s
        order by generated_at desc
        limit 1
        """,
        parameters={"run_id": run_id},
    )
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def _wait_for_active_cleared(
    client,
    runs_table: str,
    *,
    timeout_seconds: int = 120,
    poll_seconds: float = 2.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        result = client.query(f"select count() from {runs_table} where status = 'active'")
        count = result.result_rows[0][0] if result.result_rows else 0
        if count == 0:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for status='active' rows to be cleared in {runs_table}"
            )
        time.sleep(poll_seconds)


def archive_current_active_runs(client, table_suffix: str = "") -> None:
    runs_table = table_name(RUNS_TABLE, table_suffix)
    active = client.query_df(
        f"""
        select run_id, model_version, profile_version, source_kind,
               horizon_start, horizon_end, generated_at, notes, is_bias_adjusted
        from {runs_table}
        where status = 'active'
        """
    )
    if active.empty:
        return

    archived = active.copy()
    archived["status"] = "archived"
    client.insert_df(runs_table, archived)
    client.command(f"alter table {runs_table} delete where status = 'active'")
    # Wait for the async mutation to complete before returning so that the
    # caller's subsequent INSERT of a new 'active' row is not deleted by the
    # still-running mutation (ClickHouse applies pending mutations to new parts).
    _wait_for_active_cleared(client, runs_table)


def activate_run(client, run_id: str, table_suffix: str = "") -> None:
    runs_table = table_name(RUNS_TABLE, table_suffix)
    run_row = fetch_run(client, run_id, table_suffix=table_suffix)
    if not run_row:
        raise ValueError(f"Run not found: {run_id}")

    archive_current_active_runs(client, table_suffix=table_suffix)

    active_row = {
        "run_id": run_row["run_id"],
        "model_version": run_row["model_version"],
        "profile_version": run_row["profile_version"],
        "source_kind": run_row["source_kind"],
        "horizon_start": run_row["horizon_start"],
        "horizon_end": run_row["horizon_end"],
        "generated_at": run_row["generated_at"],
        "status": "active",
        "notes": run_row.get("notes"),
        "is_bias_adjusted": run_row["is_bias_adjusted"],
    }
    client.insert_df(runs_table, pd.DataFrame([active_row]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Activate a ClickHouse forecast run for embedded serving")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH))
    args = parser.parse_args()

    client = create_client(args.env_file)
    table_suffix = get_table_suffix_from_env_file(args.env_file)
    activate_run(client, args.run_id, table_suffix=table_suffix)

    print("=" * 72)
    print("RUN ACTIVATED")
    print("=" * 72)
    print(f"run_id: {args.run_id}")


if __name__ == "__main__":
    main()
