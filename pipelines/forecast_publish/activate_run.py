from __future__ import annotations

import argparse
from pathlib import Path

import clickhouse_connect
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = ROOT / ".env"
RUNS_TABLE = "forecast_runs_embedded"


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


def create_client(env_path: str | Path):
    env = load_env_file(env_path)
    return clickhouse_connect.get_client(
        host=env["HOST"],
        port=int(env["PORT"]),
        username=env["USER"],
        password=env["PASSWORD"],
        database=env["DATABASE"],
    )


def fetch_run(client, run_id: str) -> dict | None:
    df = client.query_df(
        f"""
        select run_id, model_version, profile_version, source_kind,
               horizon_start, horizon_end, generated_at, status, notes, is_bias_adjusted
        from {RUNS_TABLE}
        where run_id = %(run_id)s
        order by generated_at desc
        limit 1
        """,
        parameters={"run_id": run_id},
    )
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def archive_current_active_runs(client) -> None:
    active = client.query_df(
        f"""
        select run_id, model_version, profile_version, source_kind,
               horizon_start, horizon_end, generated_at, notes, is_bias_adjusted
        from {RUNS_TABLE}
        where status = 'active'
        """
    )
    if active.empty:
        return

    archived = active.copy()
    archived["status"] = "archived"
    client.insert_df(RUNS_TABLE, archived)


def activate_run(client, run_id: str) -> None:
    run_row = fetch_run(client, run_id)
    if not run_row:
        raise ValueError(f"Run not found: {run_id}")

    archive_current_active_runs(client)

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
    client.insert_df(RUNS_TABLE, pd.DataFrame([active_row]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Activate a ClickHouse forecast run for embedded serving")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH))
    args = parser.parse_args()

    client = create_client(args.env_file)
    activate_run(client, args.run_id)

    print("=" * 72)
    print("RUN ACTIVATED")
    print("=" * 72)
    print(f"run_id: {args.run_id}")


if __name__ == "__main__":
    main()
