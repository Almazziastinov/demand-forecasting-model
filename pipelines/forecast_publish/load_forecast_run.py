from __future__ import annotations

import argparse
from pathlib import Path
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


def load_product_lookup(profile_path: Path) -> pd.DataFrame:
    usecols = ["bakery_id", "product_id", "product_name", "category_name"]
    df = pd.read_csv(profile_path, encoding="utf-8-sig", usecols=usecols)
    return df.drop_duplicates(["bakery_id", "product_id"]).reset_index(drop=True)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a forecast run into embedded app ClickHouse storage")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH))
    parser.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH))
    parser.add_argument("--bakery-path", default=str(DEFAULT_BAKERY_PATH))
    parser.add_argument("--sku-day-path", default=str(DEFAULT_SKU_DAY_PATH))
    parser.add_argument("--sku-hour-path", default=str(DEFAULT_SKU_HOUR_PATH))
    parser.add_argument("--profile-path", default=str(DEFAULT_PROFILE_PATH))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--model-version", default="bakery_day_lgbm_v1")
    parser.add_argument("--profile-version", default="smoothed_bias_adj_v1")
    parser.add_argument("--notes", default=None)
    args = parser.parse_args()

    client = create_client(args.env_file)
    load_schema(client, Path(args.schema_path))

    run_id = build_run_id(args.run_id)
    bakery_raw = pd.read_csv(args.bakery_path, encoding="utf-8-sig")
    sku_day_raw = pd.read_csv(args.sku_day_path, encoding="utf-8-sig")
    sku_hour_raw = pd.read_csv(args.sku_hour_path, encoding="utf-8-sig")
    lookup = load_product_lookup(Path(args.profile_path))

    horizon_start, horizon_end = infer_run_dates(bakery_raw)
    insert_run_metadata(
        client,
        run_id=run_id,
        model_version=args.model_version,
        profile_version=args.profile_version,
        horizon_start=horizon_start,
        horizon_end=horizon_end,
        notes=args.notes,
    )

    bakery_day = prepare_bakery_day(bakery_raw, run_id)
    sku_day = prepare_sku_day(sku_day_raw, lookup, run_id)
    sku_hour = prepare_sku_hour(sku_hour_raw, run_id)

    client.insert_df("bakery_forecast_day_embedded", bakery_day)
    client.insert_df("sku_forecast_day_embedded", sku_day)
    client.insert_df("sku_forecast_hour_embedded", sku_hour)

    print("=" * 72)
    print("FORECAST RUN LOADED")
    print("=" * 72)
    print(f"run_id: {run_id}")
    print(f"bakery rows: {len(bakery_day)}")
    print(f"sku day rows: {len(sku_day)}")
    print(f"sku hour rows: {len(sku_hour)}")


if __name__ == "__main__":
    main()
