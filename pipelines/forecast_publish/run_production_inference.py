from __future__ import annotations

# ruff: noqa: E501
import argparse
import json
import os
from argparse import Namespace
from pathlib import Path

import pandas as pd

from pipelines.forecast_publish.activate_run import activate_run
from pipelines.forecast_publish.load_forecast_run import (
    DEFAULT_ENV_PATH,
    DEFAULT_SCHEMA_PATH,
    create_client,
    get_clickhouse_settings,
    load_env_file,
    load_forecast_run,
)
from pipelines.forecast_publish.production_dataset_refresh import (
    DEFAULT_HISTORY_START_DATE,
    DEFAULT_RAW_OUTPUT,
    DEFAULT_REFRESH_SUMMARY_PATH,
    DEFAULT_SQL_TEMPLATE,
    DEFAULT_TIMEZONE,
    refresh_production_datasets,
    refresh_weather_features_with_fallback,
    resolve_default_refresh_dates,
)
from pipelines.forecast_publish.sku_hour_profile_store import (
    PROFILE_TABLE,
    UPLIFT_MULTIPLIER_TABLE,
)
from pipelines.forecast_publish.table_names import get_table_suffix_from_env_file
from src.experiments_v2.apply_bakery_profiles import DEFAULT_BAKERY_HOUR_PROFILE_PATH
from src.experiments_v2.apply_bakery_profiles_clickhouse import (
    DEFAULT_RECENT_ABSOLUTE_CAP_MULTIPLIER,
    DEFAULT_RECENT_UPWARD_CAP_CATEGORY_PATTERN,
    DEFAULT_RECENT_UPWARD_CAP_MULTIPLIER,
    allocate_from_clickhouse,
)
from src.experiments_v2.bakery_day_forecast import (
    DEFAULT_HORIZON_DAYS,
    FORECAST_BIAS_ADJ_COL,
    run_forecast_mode,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_DATASET_PATH = ROOT / "data" / "processed" / "bakery_daily_sales.csv"
DEFAULT_UPLIFTED_DATASET_PATH = ROOT / "data" / "processed" / "bakery_daily_sales_uplifted.csv"
DEFAULT_BASE_MODEL_PATH = ROOT / "models" / "bakery_day_model.joblib"
DEFAULT_BASE_META_PATH = ROOT / "models" / "bakery_day_meta.joblib"
DEFAULT_UPLIFTED_MODEL_PATH = ROOT / "models" / "bakery_day_model_uplifted.joblib"
DEFAULT_UPLIFTED_META_PATH = ROOT / "models" / "bakery_day_meta_uplifted.joblib"
DEFAULT_BASE_BIAS_PATH = ROOT / "models" / "bakery_day_bias.json"
DEFAULT_UPLIFTED_BIAS_PATH = ROOT / "models" / "bakery_day_bias_uplifted.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed"
DEFAULT_SUMMARY_PATH = ROOT / "reports" / "production_inference_summary.json"
DEFAULT_WEATHER_PATH = ROOT / "data" / "processed" / "bakery_weather_features.csv"
DEFAULT_RECENT_CORRECTION_MODE = os.getenv(
    "FORECAST_RECENT_CORRECTION_MODE",
    "runner_city_prior_soft_weekpart",
)
DEFAULT_RECENT_CORRECTION_DAYS = int(os.getenv("FORECAST_RECENT_CORRECTION_DAYS", "30"))
DEFAULT_RECENT_SALES_TABLE = os.getenv("FORECAST_RECENT_SALES_TABLE", "mart_sales_60d")
DEFAULT_REFRESH_DATASETS = os.getenv("FORECAST_REFRESH_DATASETS", "").strip().lower() in {"1", "true", "yes", "on"}
DEFAULT_REFRESH_WEATHER = os.getenv("FORECAST_REFRESH_WEATHER", "").strip().lower() in {"1", "true", "yes", "on"}


SCENARIOS = {
    "base_raw_uplift": {
        "description": "base bakery forecast + raw uplift SKU allocation",
        "run_id_suffix": "base_bakery_raw_uplift_sku",
        "dataset_attr": "base_dataset_path",
        "model_attr": "base_model_path",
        "meta_attr": "base_meta_path",
        "bias_attr": "base_bias_path",
        "forecast_name": "bakery_day_forecast_prod_base_raw_uplift.csv",
        "output_suffix": "prod_base_bakery_raw_uplift_sku",
        "model_version": "bakery_day_lgbm_base",
        "profile_version": "clickhouse_raw_uplift_sku",
        "use_raw_uplift_multiplier": True,
    },
    "uplifted_norm": {
        "description": "uplifted bakery forecast + normalized uplift SKU allocation",
        "run_id_suffix": "uplifted_bakery_norm_uplift_sku",
        "dataset_attr": "uplifted_dataset_path",
        "model_attr": "uplifted_model_path",
        "meta_attr": "uplifted_meta_path",
        "bias_attr": "uplifted_bias_path",
        "forecast_name": "bakery_day_forecast_prod_uplifted_norm.csv",
        "output_suffix": "prod_uplifted_bakery_norm_uplift_sku",
        "model_version": "bakery_day_lgbm_uplifted",
        "profile_version": "clickhouse_norm_uplift_sku",
        "use_raw_uplift_multiplier": False,
    },
    "base_no_sku_uplift": {
        "description": "base bakery forecast + raw SKU-hour profile allocation, no SKU-hour uplift multiplier",
        "run_id_suffix": "base_bakery_no_sku_uplift",
        "dataset_attr": "base_dataset_path",
        "model_attr": "base_model_path",
        "meta_attr": "base_meta_path",
        "bias_attr": "base_bias_path",
        "forecast_name": "bakery_day_forecast_prod_base_no_sku_uplift.csv",
        "output_suffix": "prod_base_bakery_no_sku_uplift",
        "model_version": "bakery_day_lgbm_base",
        "profile_version": "clickhouse_no_sku_uplift",
        "use_raw_uplift_multiplier": False,
    },
}


def _existing_path(path: str | Path, *, label: str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _build_run_id(prefix: str, scenario: dict, bakery_path: str | Path, horizon_days: int) -> str:
    forecast_dates = pd.to_datetime(
        pd.read_csv(bakery_path, encoding="utf-8-sig", usecols=["date"])["date"],
        errors="coerce",
    ).dropna()
    if forecast_dates.empty:
        raise ValueError(f"Cannot infer forecast dates from {bakery_path}")
    date_part = str(forecast_dates.min().date()).replace("-", "")
    return f"{prefix}_{scenario['run_id_suffix']}_{date_part}_h{horizon_days}"


def run_bakery_forecast(args: argparse.Namespace, scenario: dict) -> Path:
    dataset_path = _existing_path(getattr(args, scenario["dataset_attr"]), label="dataset")
    model_path = _existing_path(getattr(args, scenario["model_attr"]), label="model")
    meta_path = _existing_path(getattr(args, scenario["meta_attr"]), label="meta")
    bias_path = _existing_path(getattr(args, scenario["bias_attr"]), label="bias table")
    output_path = Path(args.output_dir) / scenario["forecast_name"]

    return run_forecast_mode(
        Namespace(
            dataset_path=str(dataset_path),
            model_path=str(model_path),
            meta_path=str(meta_path),
            output_path=str(output_path),
            horizon_days=args.horizon_days,
            start_date=args.start_date,
            weather_path=args.weather_path,
            apply_bias_correction=not args.no_bias_correction,
            bias_path=str(bias_path),
            bias_clip_pct=args.bias_clip_pct,
        )
    )


def run_scenario(args: argparse.Namespace, scenario_name: str) -> dict:
    scenario = SCENARIOS[scenario_name]
    print(f"running scenario: {scenario_name} ({scenario['description']})", flush=True)

    bakery_path = run_bakery_forecast(args, scenario)
    allocated_paths = allocate_from_clickhouse(
        bakery_forecast_path=bakery_path,
        bakery_hour_profile_path=args.bakery_hour_profile_path,
        output_dir=args.output_dir,
        env_file=args.env_file,
        profile_table=args.profile_table,
        uplift_table=args.uplift_table,
        forecast_col=FORECAST_BIAS_ADJ_COL,
        output_suffix=scenario["output_suffix"],
        use_raw_uplift_multiplier=bool(scenario["use_raw_uplift_multiplier"]),
        uplift_profile_version=args.uplift_profile_version,
        recent_correction_mode=args.recent_correction_mode,
        recent_correction_days=args.recent_correction_days,
        recent_sales_table=args.recent_sales_table,
        chunk_size=args.chunk_size,
        assortment_table=args.assortment_table,
        disable_assortment_filter=args.disable_assortment_filter,
        disable_assortment_renormalization=args.disable_assortment_renormalization,
        recent_category_upward_cap_pattern=(
            args.recent_category_upward_cap_pattern or None
        ),
        recent_category_upward_cap_multiplier=args.recent_category_upward_cap_multiplier,
        recent_category_absolute_cap_days=(
            args.recent_category_absolute_cap_days or None
        ),
        recent_category_absolute_cap_multiplier=args.recent_category_absolute_cap_multiplier,
    )

    run_id = _build_run_id(args.run_prefix, scenario, bakery_path, args.horizon_days)
    load_result = load_forecast_run(
        env_file=args.env_file,
        schema_path=args.schema_path,
        bakery_path=bakery_path,
        sku_day_path=allocated_paths["sku_daily"],
        sku_hour_path=allocated_paths["sku_hourly"],
        profile_table=args.profile_table,
        lookup_source="clickhouse",
        run_id=run_id,
        model_version=scenario["model_version"],
        profile_version=scenario["profile_version"],
        notes=args.notes or scenario["description"],
        replace_existing=not args.no_replace_existing,
        weather_path=args.weather_path,
    )

    if args.activate_run == scenario_name:
        client = create_client(args.env_file)
        activate_run(
            client,
            run_id,
            table_suffix=get_table_suffix_from_env_file(args.env_file),
        )

    summary = {
        "scenario": scenario_name,
        "description": scenario["description"],
        "run_id": run_id,
        "bakery_path": str(bakery_path),
        "sku_day_path": str(allocated_paths["sku_daily"]),
        "sku_hour_path": str(allocated_paths["sku_hourly"]),
        "allocation_summary_path": str(allocated_paths["summary"]),
        "loaded_rows": load_result,
        "activated": args.activate_run == scenario_name,
    }
    print(f"loaded run: {run_id}", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run production forecast inference and publish runs to ClickHouse")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH))
    parser.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH))
    parser.add_argument("--refresh-datasets", action="store_true", default=DEFAULT_REFRESH_DATASETS)
    parser.add_argument("--refresh-weather", action="store_true", default=DEFAULT_REFRESH_WEATHER)
    parser.add_argument("--refresh-timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--history-start-date", default=DEFAULT_HISTORY_START_DATE)
    parser.add_argument("--history-end-date", default=None)
    parser.add_argument("--raw-output", default=str(DEFAULT_RAW_OUTPUT))
    parser.add_argument("--sql-template", default=str(DEFAULT_SQL_TEMPLATE))
    parser.add_argument("--dataset-refresh-summary-path", default=str(DEFAULT_REFRESH_SUMMARY_PATH))
    parser.add_argument("--skip-weather-refresh", action="store_true")
    parser.add_argument("--base-dataset-path", default=str(DEFAULT_BASE_DATASET_PATH))
    parser.add_argument("--uplifted-dataset-path", default=str(DEFAULT_UPLIFTED_DATASET_PATH))
    parser.add_argument("--base-model-path", default=str(DEFAULT_BASE_MODEL_PATH))
    parser.add_argument("--base-meta-path", default=str(DEFAULT_BASE_META_PATH))
    parser.add_argument("--uplifted-model-path", default=str(DEFAULT_UPLIFTED_MODEL_PATH))
    parser.add_argument("--uplifted-meta-path", default=str(DEFAULT_UPLIFTED_META_PATH))
    parser.add_argument("--base-bias-path", default=str(DEFAULT_BASE_BIAS_PATH))
    parser.add_argument("--uplifted-bias-path", default=str(DEFAULT_UPLIFTED_BIAS_PATH))
    parser.add_argument("--bakery-hour-profile-path", default=str(DEFAULT_BAKERY_HOUR_PROFILE_PATH))
    parser.add_argument("--profile-table", default=PROFILE_TABLE)
    parser.add_argument("--uplift-table", default=UPLIFT_MULTIPLIER_TABLE)
    parser.add_argument("--uplift-profile-version", default=None)
    parser.add_argument(
        "--recent-correction-mode",
        choices=[
            "none",
            "dead_0d",
            "blend_recent_50",
            "core_recent_70",
            "runner_city_prior_soft_weekpart",
        ],
        default=DEFAULT_RECENT_CORRECTION_MODE,
    )
    parser.add_argument("--recent-correction-days", type=int, default=DEFAULT_RECENT_CORRECTION_DAYS)
    parser.add_argument("--recent-sales-table", default=DEFAULT_RECENT_SALES_TABLE)
    parser.add_argument("--assortment-table", default="assortment_city_products")
    parser.add_argument("--disable-assortment-filter", action="store_true")
    parser.add_argument("--disable-assortment-renormalization", action="store_true")
    parser.add_argument(
        "--recent-category-upward-cap-pattern",
        default=DEFAULT_RECENT_UPWARD_CAP_CATEGORY_PATTERN,
        help=(
            "Regex over category_name for costly categories that recent correction "
            "may not lift above base profile forecast. Empty string disables."
        ),
    )
    parser.add_argument(
        "--recent-category-upward-cap-multiplier",
        type=float,
        default=DEFAULT_RECENT_UPWARD_CAP_MULTIPLIER,
    )
    parser.add_argument(
        "--recent-category-absolute-cap-days",
        type=int,
        default=0,
        help=(
            "Calendar-day window for recent absolute cap. 0 means use "
            "--recent-correction-days."
        ),
    )
    parser.add_argument(
        "--recent-category-absolute-cap-multiplier",
        type=float,
        default=DEFAULT_RECENT_ABSOLUTE_CAP_MULTIPLIER,
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--weather-path", default=str(DEFAULT_WEATHER_PATH))
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--run-prefix", default="prod")
    parser.add_argument(
        "--require-nonprod-tables",
        action="store_true",
        help="Refuse to run unless APP_ENV is not prod and FORECAST_TABLE_SUFFIX is set.",
    )
    parser.add_argument("--scenario", choices=["both", *SCENARIOS.keys()], default="both")
    parser.add_argument("--activate-run", choices=["none", *SCENARIOS.keys()], default="none")
    parser.add_argument("--notes", default=None)
    parser.add_argument("--no-bias-correction", action="store_true")
    parser.add_argument("--no-replace-existing", action="store_true")
    parser.add_argument("--bias-clip-pct", type=float, default=0.15)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    return parser


def assert_nonprod_tables(env_file: str | Path) -> None:
    env = load_env_file(env_file)
    app_env = (env.get("APP_ENV") or os.getenv("APP_ENV") or "dev").strip().lower()
    settings = get_clickhouse_settings(env_file)
    database = str(settings.get("database") or "")
    table_suffix = env.get("FORECAST_TABLE_SUFFIX", "").strip()
    if app_env == "prod":
        raise RuntimeError(f"Refusing non-prod run with APP_ENV=prod in {env_file}")
    if not database:
        raise RuntimeError(f"CLICKHOUSE_DATABASE is required in {env_file}")
    if not table_suffix:
        raise RuntimeError(
            f"FORECAST_TABLE_SUFFIX is required for non-prod run in {env_file}"
        )
    get_table_suffix_from_env_file(env_file)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.require_nonprod_tables:
        assert_nonprod_tables(args.env_file)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    dataset_refresh_result = None
    if args.refresh_datasets:
        history_end_date = args.history_end_date or resolve_default_refresh_dates(
            timezone=args.refresh_timezone,
        ).history_end_date
        dataset_refresh_result = refresh_production_datasets(
            env_file=args.env_file,
            sql_template=args.sql_template,
            raw_output=args.raw_output,
            processed_dir=Path(args.base_dataset_path).parent,
            history_start_date=args.history_start_date,
            history_end_date=history_end_date,
            horizon_days=args.horizon_days,
            bakery_hour_profile_path=args.bakery_hour_profile_path,
            uplifted_output_path=args.uplifted_dataset_path,
            weather_path=args.weather_path,
            uplift_table=args.uplift_table,
            uplift_profile_version=args.uplift_profile_version,
            refresh_weather=not args.skip_weather_refresh,
        )
        refresh_summary_path = Path(args.dataset_refresh_summary_path)
        refresh_summary_path.parent.mkdir(parents=True, exist_ok=True)
        refresh_summary_path.write_text(
            json.dumps(dataset_refresh_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if args.refresh_weather and not args.refresh_datasets:
        # Refresh weather independently when full dataset refresh is disabled.
        # fetch_weather_features only calls Open-Meteo API — no ClickHouse dependency.
        weather_result = refresh_weather_features_with_fallback(
            dataset_paths=[
                Path(args.uplifted_dataset_path),
                Path(args.base_dataset_path),
            ],
            horizon_days=args.horizon_days,
            weather_path=args.weather_path,
        )
        print(
            f"weather refresh: {weather_result['weather_status']}"
            f" rows={weather_result['weather_rows']}"
            f" {weather_result['weather_start_date']} → {weather_result['weather_end_date']}",
            flush=True,
        )

    scenario_names = list(SCENARIOS) if args.scenario == "both" else [args.scenario]
    if args.activate_run != "none" and args.activate_run not in scenario_names:
        raise ValueError("--activate-run must be one of the selected scenarios")

    results = [run_scenario(args, scenario_name) for scenario_name in scenario_names]
    summary = {
        "horizon_days": args.horizon_days,
        "start_date": args.start_date,
        "profile_table": args.profile_table,
        "uplift_table": args.uplift_table,
        "uplift_profile_version": args.uplift_profile_version,
        "recent_correction_mode": args.recent_correction_mode,
        "recent_correction_days": args.recent_correction_days,
        "recent_sales_table": args.recent_sales_table,
        "recent_category_upward_cap_pattern": args.recent_category_upward_cap_pattern,
        "recent_category_upward_cap_multiplier": args.recent_category_upward_cap_multiplier,
        "recent_category_absolute_cap_days": args.recent_category_absolute_cap_days
        or args.recent_correction_days,
        "recent_category_absolute_cap_multiplier": args.recent_category_absolute_cap_multiplier,
        "dataset_refresh": dataset_refresh_result,
        "scenarios": results,
    }
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("PRODUCTION INFERENCE COMPLETE")
    print("=" * 72)
    print(f"summary: {summary_path}")
    for item in results:
        status = " active" if item["activated"] else ""
        print(f"{item['scenario']}: {item['run_id']}{status}")


if __name__ == "__main__":
    main()
