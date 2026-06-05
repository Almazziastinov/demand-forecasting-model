from __future__ import annotations

# ruff: noqa: E501

import argparse
import json
import os
from argparse import Namespace
from pathlib import Path

import pandas as pd

from pipelines.forecast_publish.activate_run import activate_run
from pipelines.forecast_publish.load_forecast_run import DEFAULT_ENV_PATH
from pipelines.forecast_publish.load_forecast_run import DEFAULT_SCHEMA_PATH
from pipelines.forecast_publish.load_forecast_run import create_client
from pipelines.forecast_publish.load_forecast_run import load_forecast_run
from pipelines.forecast_publish.sku_hour_profile_store import PROFILE_TABLE
from pipelines.forecast_publish.sku_hour_profile_store import UPLIFT_MULTIPLIER_TABLE
from src.experiments_v2.apply_bakery_profiles import DEFAULT_BAKERY_HOUR_PROFILE_PATH
from src.experiments_v2.apply_bakery_profiles_clickhouse import allocate_from_clickhouse
from src.experiments_v2.bakery_day_forecast import DEFAULT_HORIZON_DAYS
from src.experiments_v2.bakery_day_forecast import FORECAST_BIAS_ADJ_COL
from src.experiments_v2.bakery_day_forecast import run_forecast_mode


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
DEFAULT_RECENT_CORRECTION_MODE = os.getenv("FORECAST_RECENT_CORRECTION_MODE", "none")
DEFAULT_RECENT_CORRECTION_DAYS = int(os.getenv("FORECAST_RECENT_CORRECTION_DAYS", "30"))
DEFAULT_RECENT_SALES_TABLE = os.getenv("FORECAST_RECENT_SALES_TABLE", "mart_sales_60d")


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
        activate_run(client, run_id)

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
        choices=["none", "dead_0d", "blend_recent_50", "core_recent_70"],
        default=DEFAULT_RECENT_CORRECTION_MODE,
    )
    parser.add_argument("--recent-correction-days", type=int, default=DEFAULT_RECENT_CORRECTION_DAYS)
    parser.add_argument("--recent-sales-table", default=DEFAULT_RECENT_SALES_TABLE)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--weather-path", default=str(DEFAULT_WEATHER_PATH))
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--run-prefix", default="prod")
    parser.add_argument("--scenario", choices=["both", *SCENARIOS.keys()], default="both")
    parser.add_argument("--activate-run", choices=["none", *SCENARIOS.keys()], default="none")
    parser.add_argument("--notes", default=None)
    parser.add_argument("--no-bias-correction", action="store_true")
    parser.add_argument("--no-replace-existing", action="store_true")
    parser.add_argument("--bias-clip-pct", type=float, default=0.15)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

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
