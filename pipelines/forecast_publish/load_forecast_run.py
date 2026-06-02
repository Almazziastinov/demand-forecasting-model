from __future__ import annotations

# ruff: noqa: E501

import argparse
from pathlib import Path
import time
from uuid import uuid4

import clickhouse_connect
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_PATH = ROOT / "apps" / "forecast_embedded" / "sql" / "schema.sql"
DEFAULT_BAKERY_PATH = ROOT / "data" / "processed" / "bakery_day_forecast.csv"
DEFAULT_SKU_DAY_PATH = ROOT / "data" / "processed" / "sku_day_forecast_future_smoothed_bias_adj.csv"
DEFAULT_SKU_HOUR_PATH = ROOT / "data" / "processed" / "sku_hour_forecast_future_smoothed_bias_adj.csv"
DEFAULT_PROFILE_PATH = ROOT / "data" / "processed" / "sku_hour_share_profile_smoothed.csv"
DEFAULT_ENV_PATH = ROOT / ".env"
DEFAULT_PROFILE_TABLE = "sku_hour_share_profile_smoothed_embedded"


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
        "host": env.get("HOST") or env.get("CLICKHOUSE_HOST"),
        "port": int(env.get("PORT") or env.get("CLICKHOUSE_PORT") or "8443"),
        "username": env.get("USER") or env.get("CLICKHOUSE_USER"),
        "password": env.get("PASSWORD") or env.get("CLICKHOUSE_PASSWORD"),
        "database": env.get("DATABASE") or env.get("CLICKHOUSE_DATABASE"),
        "secure": _as_bool(env.get("SECURE") or env.get("CLICKHOUSE_SECURE"), default=True),
        "verify": _as_bool(env.get("VERIFY") or env.get("CLICKHOUSE_VERIFY"), default=False),
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


def load_schema(client, schema_path: Path) -> None:
    sql_text = schema_path.read_text(encoding="utf-8")
    for statement in sql_text.split(";"):
        stmt = statement.strip()
        if stmt:
            client.command(stmt)


def infer_run_dates(bakery_df: pd.DataFrame) -> tuple[str, str]:
    dates = pd.to_datetime(bakery_df["date"], errors="coerce").dropna()
    return str(dates.min().date()), str(dates.max().date())


def build_run_id(prefix: str | None) -> str:
    return prefix or f"run_{uuid4().hex[:12]}"


def _quote_clickhouse_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "''")


def count_run_rows(client, run_id: str) -> dict[str, int]:
    tables = [
        "forecast_runs_embedded",
        "bakery_forecast_day_embedded",
        "sku_forecast_day_embedded",
        "sku_forecast_hour_embedded",
    ]
    counts: dict[str, int] = {}
    for table in tables:
        df = client.query_df(
            f"select count() as rows from {table} where run_id = %(run_id)s",
            parameters={"run_id": run_id},
        )
        counts[table] = int(df["rows"].iloc[0])
    return counts


def delete_forecast_run(client, run_id: str) -> None:
    safe_run_id = _quote_clickhouse_string(run_id)
    for table in [
        "forecast_runs_embedded",
        "bakery_forecast_day_embedded",
        "sku_forecast_day_embedded",
        "sku_forecast_hour_embedded",
    ]:
        client.command(f"alter table {table} delete where run_id = '{safe_run_id}'")


def wait_for_run_deleted(
    client,
    run_id: str,
    *,
    timeout_seconds: int = 120,
    poll_seconds: float = 2.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        counts = count_run_rows(client, run_id)
        if all(rows == 0 for rows in counts.values()):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for run cleanup: {run_id} counts={counts}")
        time.sleep(poll_seconds)


def load_product_lookup(profile_path: Path) -> pd.DataFrame:
    usecols = ["bakery_id", "product_id", "product_name", "category_name"]
    df = pd.read_csv(profile_path, encoding="utf-8-sig", usecols=usecols)
    return df.drop_duplicates(["bakery_id", "product_id"]).reset_index(drop=True)


def load_product_lookup_from_clickhouse(client, profile_table: str = DEFAULT_PROFILE_TABLE) -> pd.DataFrame:
    return client.query_df(
        f"""
        select
            bakery_id,
            product_id,
            any(product_name) as product_name,
            any(category_name) as category_name
        from {profile_table}
        group by bakery_id, product_id
        """
    ).drop_duplicates(["bakery_id", "product_id"]).reset_index(drop=True)


def prepare_bakery_day(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    work = df.copy()
    work["run_id"] = run_id
    work["forecast_date"] = pd.to_datetime(work["date"], errors="coerce").dt.date
    work["forecast_base"] = pd.to_numeric(work["bakery_day_forecast"], errors="coerce")
    work["forecast_final"] = pd.to_numeric(
        work.get("bakery_day_forecast_bias_adj", work["bakery_day_forecast"]),
        errors="coerce",
    )
    return work[
        [
            "run_id",
            "forecast_date",
            "bakery_id",
            "bakery_name",
            "city",
            "forecast_base",
            "forecast_final",
        ]
    ].dropna(subset=["forecast_date", "bakery_id", "forecast_final"])


def prepare_sku_day(df: pd.DataFrame, lookup: pd.DataFrame, run_id: str) -> pd.DataFrame:
    work = df.merge(lookup, on=["bakery_id", "product_id"], how="left", validate="many_to_one")
    work["run_id"] = run_id
    work["forecast_date"] = pd.to_datetime(work["date"], errors="coerce").dt.date
    work["forecast_qty"] = pd.to_numeric(work["sku_day_forecast"], errors="coerce")
    return work[
        [
            "run_id",
            "forecast_date",
            "bakery_id",
            "product_id",
            "product_name",
            "category_name",
            "forecast_qty",
        ]
    ].dropna(subset=["forecast_date", "bakery_id", "product_id", "forecast_qty"])


def prepare_sku_hour(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    work = df.copy()
    work["run_id"] = run_id
    work["forecast_date"] = pd.to_datetime(work["date"], errors="coerce").dt.date
    work["forecast_qty"] = pd.to_numeric(work["sku_hour_forecast"], errors="coerce")
    return work[
        [
            "run_id",
            "forecast_date",
            "bakery_id",
            "product_id",
            "hour",
            "forecast_qty",
        ]
    ].dropna(subset=["forecast_date", "bakery_id", "product_id", "hour", "forecast_qty"])


def insert_run_metadata(
    client,
    *,
    run_id: str,
    model_version: str,
    profile_version: str,
    horizon_start: str,
    horizon_end: str,
    notes: str | None,
) -> None:
    run_df = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "model_version": model_version,
                "profile_version": profile_version,
                "source_kind": "bakery_driven",
                "horizon_start": pd.to_datetime(horizon_start).date(),
                "horizon_end": pd.to_datetime(horizon_end).date(),
                "generated_at": pd.Timestamp.utcnow().tz_localize(None),
                "status": "draft",
                "notes": notes,
                "is_bias_adjusted": True,
            }
        ]
    )
    client.insert_df("forecast_runs_embedded", run_df)


def load_forecast_run(
    *,
    env_file: str | Path = DEFAULT_ENV_PATH,
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
    bakery_path: str | Path = DEFAULT_BAKERY_PATH,
    sku_day_path: str | Path = DEFAULT_SKU_DAY_PATH,
    sku_hour_path: str | Path = DEFAULT_SKU_HOUR_PATH,
    profile_path: str | Path | None = DEFAULT_PROFILE_PATH,
    profile_table: str = DEFAULT_PROFILE_TABLE,
    lookup_source: str = "csv",
    run_id: str | None = None,
    model_version: str = "bakery_day_lgbm_v1",
    profile_version: str = "smoothed_bias_adj_v1",
    notes: str | None = None,
    replace_existing: bool = False,
) -> dict[str, int | str]:
    client = create_client(env_file)
    load_schema(client, Path(schema_path))

    final_run_id = build_run_id(run_id)
    if replace_existing:
        delete_forecast_run(client, final_run_id)
        wait_for_run_deleted(client, final_run_id)

    bakery_raw = pd.read_csv(bakery_path, encoding="utf-8-sig")
    sku_day_raw = pd.read_csv(sku_day_path, encoding="utf-8-sig")
    sku_hour_raw = pd.read_csv(sku_hour_path, encoding="utf-8-sig")
    if lookup_source == "clickhouse":
        lookup = load_product_lookup_from_clickhouse(client, profile_table)
    elif lookup_source == "csv":
        if not profile_path:
            raise ValueError("profile_path is required when lookup_source='csv'")
        lookup = load_product_lookup(Path(profile_path))
    else:
        raise ValueError(f"Unsupported lookup_source: {lookup_source}")

    horizon_start, horizon_end = infer_run_dates(bakery_raw)
    insert_run_metadata(
        client,
        run_id=final_run_id,
        model_version=model_version,
        profile_version=profile_version,
        horizon_start=horizon_start,
        horizon_end=horizon_end,
        notes=notes,
    )

    bakery_day = prepare_bakery_day(bakery_raw, final_run_id)
    sku_day = prepare_sku_day(sku_day_raw, lookup, final_run_id)
    sku_hour = prepare_sku_hour(sku_hour_raw, final_run_id)

    client.insert_df("bakery_forecast_day_embedded", bakery_day)
    client.insert_df("sku_forecast_day_embedded", sku_day)
    client.insert_df("sku_forecast_hour_embedded", sku_hour)

    return {
        "run_id": final_run_id,
        "bakery_rows": len(bakery_day),
        "sku_day_rows": len(sku_day),
        "sku_hour_rows": len(sku_hour),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a forecast run into embedded app ClickHouse storage")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH))
    parser.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH))
    parser.add_argument("--bakery-path", default=str(DEFAULT_BAKERY_PATH))
    parser.add_argument("--sku-day-path", default=str(DEFAULT_SKU_DAY_PATH))
    parser.add_argument("--sku-hour-path", default=str(DEFAULT_SKU_HOUR_PATH))
    parser.add_argument("--profile-path", default=str(DEFAULT_PROFILE_PATH))
    parser.add_argument("--profile-table", default=DEFAULT_PROFILE_TABLE)
    parser.add_argument("--lookup-source", choices=["csv", "clickhouse"], default="csv")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--model-version", default="bakery_day_lgbm_v1")
    parser.add_argument("--profile-version", default="smoothed_bias_adj_v1")
    parser.add_argument("--notes", default=None)
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args()

    result = load_forecast_run(
        env_file=args.env_file,
        schema_path=args.schema_path,
        bakery_path=args.bakery_path,
        sku_day_path=args.sku_day_path,
        sku_hour_path=args.sku_hour_path,
        profile_path=args.profile_path,
        profile_table=args.profile_table,
        lookup_source=args.lookup_source,
        run_id=args.run_id,
        model_version=args.model_version,
        profile_version=args.profile_version,
        notes=args.notes,
        replace_existing=args.replace_existing,
    )

    print("=" * 72)
    print("FORECAST RUN LOADED")
    print("=" * 72)
    print(f"run_id: {result['run_id']}")
    print(f"bakery rows: {result['bakery_rows']}")
    print(f"sku day rows: {result['sku_day_rows']}")
    print(f"sku hour rows: {result['sku_hour_rows']}")


if __name__ == "__main__":
    main()
