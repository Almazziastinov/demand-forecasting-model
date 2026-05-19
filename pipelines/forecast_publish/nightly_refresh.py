from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipelines.forecast_publish.activate_run import activate_run
from pipelines.forecast_publish.load_forecast_run import create_client as create_publish_client
from pipelines.forecast_publish.load_forecast_run import DEFAULT_ENV_PATH
from pipelines.forecast_publish.load_forecast_run import insert_run_metadata
from pipelines.forecast_publish.load_forecast_run import load_product_lookup
from pipelines.forecast_publish.load_forecast_run import load_schema
from pipelines.forecast_publish.load_forecast_run import prepare_bakery_day
from pipelines.forecast_publish.load_forecast_run import prepare_sku_day
from pipelines.forecast_publish.load_forecast_run import prepare_sku_hour
from pipelines.forecast_publish.sku_hour_profile_store import export_profile_from_clickhouse
from pipelines.forecast_publish.sku_hour_profile_store import PROFILE_TABLE
from scripts.export_clickhouse_checks import create_client as create_export_client
from scripts.export_clickhouse_checks import DEFAULT_OUTPUT as DEFAULT_RAW_OUTPUT
from scripts.export_clickhouse_checks import DEFAULT_SQL_TEMPLATE
from scripts.export_clickhouse_checks import export_windows
from src.experiments_v2.apply_bakery_profiles import apply_profiles
from src.experiments_v2.bakery_day_forecast import DEFAULT_BIAS_PATH
from src.experiments_v2.bakery_day_forecast import DEFAULT_FORECAST_PATH
from src.experiments_v2.bakery_day_forecast import DEFAULT_META_PATH
from src.experiments_v2.bakery_day_forecast import DEFAULT_MODEL_PATH
from src.experiments_v2.bakery_day_forecast import run_train_mode
from src.experiments_v2.bakery_day_forecast import run_forecast_mode
from src.experiments_v2.build_bakery_hour_profile import build_bakery_hour_profile
from src.experiments_v2.build_bakery_hour_profile import build_summary as build_hour_summary
from src.experiments_v2.build_bakery_hour_profile import save_outputs as save_hour_outputs
from src.experiments_v2.build_bakery_daily_dataset import build_bakery_daily_dataset
from src.experiments_v2.build_bakery_daily_dataset import build_summary as build_daily_summary
from src.experiments_v2.build_bakery_daily_dataset import save_outputs as save_daily_outputs


DEFAULT_SCHEMA_PATH = ROOT / "apps" / "forecast_embedded" / "sql" / "schema.sql"
DEFAULT_PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_BAKERY_HOUR_PROFILE_PATH = DEFAULT_PROCESSED_DIR / "bakery_hour_profile.csv"
DEFAULT_SKU_HOUR_PROFILE_PATH = DEFAULT_PROCESSED_DIR / "sku_hour_share_profile_smoothed.csv"
DEFAULT_SKU_HOUR_PROFILE_CACHE_PATH = DEFAULT_PROCESSED_DIR / "sku_hour_share_profile_smoothed.clickhouse.csv"
DEFAULT_SKU_DAY_CANONICAL_PATH = DEFAULT_PROCESSED_DIR / "sku_day_forecast_future_smoothed_bias_adj.csv"
DEFAULT_SKU_HOUR_CANONICAL_PATH = DEFAULT_PROCESSED_DIR / "sku_hour_forecast_future_smoothed_bias_adj.csv"
DEFAULT_MANIFEST_PATH = ROOT / "reports" / "nightly_forecast_last_run.json"


@dataclass(frozen=True)
class NightlyDates:
    history_end_date: str
    forecast_start_date: str


def resolve_default_dates(now: pd.Timestamp | None = None) -> NightlyDates:
    current = pd.Timestamp.utcnow().tz_localize(None).normalize() if now is None else pd.Timestamp(now).normalize()
    return NightlyDates(
        history_end_date=str((current - pd.Timedelta(days=1)).date()),
        forecast_start_date=str(current.date()),
    )


def build_run_id(*, forecast_start_date: str, horizon_days: int, prefix: str = "nightly") -> str:
    date_token = pd.Timestamp(forecast_start_date).strftime("%Y%m%d")
    return f"{prefix}_{date_token}_h{int(horizon_days):02d}"


def copy_file(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def publish_run(
    *,
    env_file: Path,
    schema_path: Path,
    bakery_path: Path,
    sku_day_path: Path,
    sku_hour_path: Path,
    profile_path: Path,
    run_id: str,
    model_version: str,
    profile_version: str,
    notes: str | None,
    activate: bool,
) -> dict[str, str]:
    client = create_publish_client(env_file)
    load_schema(client, schema_path)

    bakery_raw = pd.read_csv(bakery_path, encoding="utf-8-sig")
    sku_day_raw = pd.read_csv(sku_day_path, encoding="utf-8-sig")
    sku_hour_raw = pd.read_csv(sku_hour_path, encoding="utf-8-sig")
    lookup = load_product_lookup(profile_path)

    dates = pd.to_datetime(bakery_raw["date"], errors="coerce").dropna()
    horizon_start = str(dates.min().date())
    horizon_end = str(dates.max().date())

    insert_run_metadata(
        client,
        run_id=run_id,
        model_version=model_version,
        profile_version=profile_version,
        horizon_start=horizon_start,
        horizon_end=horizon_end,
        notes=notes,
    )

    bakery_day = prepare_bakery_day(bakery_raw, run_id)
    sku_day = prepare_sku_day(sku_day_raw, lookup, run_id)
    sku_hour = prepare_sku_hour(sku_hour_raw, run_id)

    client.insert_df("bakery_forecast_day_embedded", bakery_day)
    client.insert_df("sku_forecast_day_embedded", sku_day)
    client.insert_df("sku_forecast_hour_embedded", sku_hour)

    if activate:
        activate_run(client, run_id)

    return {
        "run_id": run_id,
        "horizon_start": horizon_start,
        "horizon_end": horizon_end,
        "bakery_rows": str(len(bakery_day)),
        "sku_day_rows": str(len(sku_day)),
        "sku_hour_rows": str(len(sku_hour)),
        "activated": str(bool(activate)).lower(),
    }


def build_parser() -> argparse.ArgumentParser:
    defaults = resolve_default_dates()
    parser = argparse.ArgumentParser(description="Nightly bakery forecast refresh and publish pipeline")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH))
    parser.add_argument("--sql-template", default=str(DEFAULT_SQL_TEMPLATE))
    parser.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH))
    parser.add_argument("--raw-output", default=str(DEFAULT_RAW_OUTPUT))
    parser.add_argument("--processed-dir", default=str(DEFAULT_PROCESSED_DIR))
    parser.add_argument("--history-start-date", default="2025-01-01")
    parser.add_argument("--history-end-date", default=defaults.history_end_date)
    parser.add_argument("--forecast-start-date", default=defaults.forecast_start_date)
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-id-prefix", default="nightly")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--meta-path", default=str(DEFAULT_META_PATH))
    parser.add_argument("--bias-path", default=str(DEFAULT_BIAS_PATH))
    parser.add_argument("--bakery-forecast-path", default=str(DEFAULT_FORECAST_PATH))
    parser.add_argument("--bakery-hour-profile-path", default=str(DEFAULT_BAKERY_HOUR_PROFILE_PATH))
    parser.add_argument("--sku-hour-profile-path", default=str(DEFAULT_SKU_HOUR_PROFILE_PATH))
    parser.add_argument("--sku-hour-profile-cache-path", default=str(DEFAULT_SKU_HOUR_PROFILE_CACHE_PATH))
    parser.add_argument("--profile-source", choices=["local", "clickhouse"], default="local")
    parser.add_argument("--profile-table", default=PROFILE_TABLE)
    parser.add_argument("--refresh-profile-cache", action="store_true")
    parser.add_argument("--sku-day-canonical-path", default=str(DEFAULT_SKU_DAY_CANONICAL_PATH))
    parser.add_argument("--sku-hour-canonical-path", default=str(DEFAULT_SKU_HOUR_CANONICAL_PATH))
    parser.add_argument("--model-version", default="bakery_day_lgbm_v1")
    parser.add_argument("--profile-version", default="smoothed_bias_adj_v1")
    parser.add_argument("--notes", default="nightly automated refresh")
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--train-model", action="store_true")
    parser.add_argument("--rebuild-bakery-hour-profile", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-publish", action="store_true")
    parser.add_argument("--skip-activate", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    env_file = Path(args.env_file)
    sql_template_path = Path(args.sql_template)
    raw_output = Path(args.raw_output)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest_path)

    run_id = args.run_id or build_run_id(
        forecast_start_date=args.forecast_start_date,
        horizon_days=args.horizon_days,
        prefix=args.run_id_prefix,
    )

    if not args.skip_export:
        sql_template_text = sql_template_path.read_text(encoding="utf-8")
        export_client = create_export_client(env_file)
        export_windows(
            client=export_client,
            sql_template_text=sql_template_text,
            output_path=raw_output,
            date_from=args.history_start_date,
            date_to=args.history_end_date,
            batch_mode="monthly",
            limit=None,
        )

    daily_df = build_bakery_daily_dataset(raw_output)
    daily_summary = build_daily_summary(daily_df)
    save_daily_outputs(processed_dir, daily_df, daily_summary)

    bakery_hour_profile_path = Path(args.bakery_hour_profile_path)
    if args.rebuild_bakery_hour_profile or not bakery_hour_profile_path.exists():
        bakery_hour_profile, bakery_hour_applied = build_bakery_hour_profile(raw_output)
        bakery_hour_summary = build_hour_summary(bakery_hour_profile, bakery_hour_applied)
        save_hour_outputs(processed_dir, bakery_hour_profile, bakery_hour_applied, bakery_hour_summary)
    else:
        bakery_hour_summary = None

    model_path = Path(args.model_path)
    meta_path = Path(args.meta_path)
    bias_path = Path(args.bias_path)
    if args.train_model or not (model_path.exists() and meta_path.exists() and bias_path.exists()):
        train_args = argparse.Namespace(
            dataset_path=str(processed_dir / "bakery_daily_sales.csv"),
            model_path=str(model_path),
            meta_path=str(meta_path),
            summary_path=str(ROOT / "reports" / "bakery_day_model_summary.json"),
            predictions_path=str(ROOT / "reports" / "bakery_day_model_holdout_predictions.csv"),
            bias_path=str(bias_path),
            test_days=int(args.test_days),
        )
        run_train_mode(train_args)

    forecast_args = argparse.Namespace(
        dataset_path=str(processed_dir / "bakery_daily_sales.csv"),
        model_path=str(model_path),
        meta_path=str(meta_path),
        bias_path=str(bias_path),
        horizon_days=int(args.horizon_days),
        start_date=args.forecast_start_date,
        apply_bias_correction=True,
        bias_clip_pct=0.15,
        output_path=str(Path(args.bakery_forecast_path)),
    )
    run_forecast_mode(forecast_args)

    profile_path = Path(args.sku_hour_profile_path)
    if args.profile_source == "clickhouse":
        cache_path = Path(args.sku_hour_profile_cache_path)
        if args.refresh_profile_cache or not cache_path.exists():
            export_profile_from_clickhouse(
                output_path=cache_path,
                env_file=env_file,
                table=args.profile_table,
            )
        profile_path = cache_path

    applied = apply_profiles(
        args.bakery_forecast_path,
        bakery_hour_profile_path,
        profile_path,
        processed_dir,
        forecast_col="bakery_day_forecast_bias_adj",
    )

    sku_day_canonical = copy_file(applied["sku_daily"], Path(args.sku_day_canonical_path))
    sku_hour_canonical = copy_file(applied["sku_hourly"], Path(args.sku_hour_canonical_path))

    manifest: dict[str, object] = {
        "run_id": run_id,
        "history_start_date": args.history_start_date,
        "history_end_date": args.history_end_date,
        "forecast_start_date": args.forecast_start_date,
        "horizon_days": int(args.horizon_days),
        "raw_output": str(raw_output),
        "bakery_forecast_path": str(Path(args.bakery_forecast_path)),
        "bakery_hour_profile_path": str(bakery_hour_profile_path),
        "profile_source": args.profile_source,
        "profile_path": str(profile_path),
        "sku_day_path": str(sku_day_canonical),
        "sku_hour_path": str(sku_hour_canonical),
        "daily_summary": daily_summary,
    }
    if bakery_hour_summary is not None:
        manifest["bakery_hour_summary"] = bakery_hour_summary

    if not args.skip_publish:
        publish_info = publish_run(
            env_file=env_file,
            schema_path=Path(args.schema_path),
            bakery_path=Path(args.bakery_forecast_path),
            sku_day_path=sku_day_canonical,
            sku_hour_path=sku_hour_canonical,
            profile_path=Path(args.sku_hour_profile_path),
            run_id=run_id,
            model_version=args.model_version,
            profile_version=args.profile_version,
            notes=args.notes,
            activate=not args.skip_activate,
        )
        manifest["publish"] = publish_info

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("NIGHTLY FORECAST REFRESH COMPLETE")
    print("=" * 72)
    print(f"run_id: {run_id}")
    print(f"history_end_date: {args.history_end_date}")
    print(f"forecast_start_date: {args.forecast_start_date}")
    print(f"horizon_days: {args.horizon_days}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
