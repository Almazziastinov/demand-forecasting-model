"""Build missing production lead-1 snapshots from the bakery model.

This is for gaps where no bakery-level lead-1 snapshot exists yet. For each
requested date the script forecasts one day using only history before that day,
allocates the bakery forecast to SKU/hour through ClickHouse profiles, and loads
a draft run whose snapshots have lead_days=1.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipelines.forecast_publish.load_forecast_run import (  # noqa: E402
    create_client,
    load_forecast_run,
)
from pipelines.forecast_publish.rolling_bakery_bias import (  # noqa: E402
    DEFAULT_MIN_DAYS as DEFAULT_ROLLING_BIAS_MIN_DAYS,
    DEFAULT_TRAILING_DAYS as DEFAULT_ROLLING_BIAS_DAYS,
    build_rolling_bias_table,
)
from pipelines.forecast_publish.table_names import table_name  # noqa: E402
from src.experiments_v2.apply_bakery_profiles_clickhouse import allocate_from_clickhouse  # noqa: E402
from src.experiments_v2.bakery_day_forecast import (  # noqa: E402
    DATE_COL,
    FORECAST_BIAS_ADJ_COL,
    FORECAST_COL,
    load_bias_table,
    run_forecast_mode,
)
from src.experiments_v2.build_bakery_daily_dataset import BAKERY_ID_COL  # noqa: E402

DEFAULT_DATASET_PATH = ROOT / "data" / "processed" / "bakery_daily_sales_uplifted.csv"
DEFAULT_MODEL_PATH = ROOT / "models" / "bakery_day_model_uplifted.joblib"
DEFAULT_META_PATH = ROOT / "models" / "bakery_day_meta_uplifted.joblib"
DEFAULT_BIAS_PATH = ROOT / "models" / "bakery_day_bias_uplifted.json"

BASE_DATASET_PATH = ROOT / "data" / "processed" / "bakery_daily_sales.csv"
BASE_MODEL_PATH = ROOT / "models" / "bakery_day_model.joblib"
BASE_META_PATH = ROOT / "models" / "bakery_day_meta.joblib"
BASE_BIAS_PATH = ROOT / "models" / "bakery_day_bias.json"
FALLBACK_BIAS_PATH = ROOT / "reports" / "bakery_day_model_bias_by_bakery.csv"
DEFAULT_WEATHER_PATH = ROOT / "data" / "processed" / "bakery_weather_features.csv"
DEFAULT_BAKERY_HOUR_PROFILE_PATH = (
    ROOT / "data" / "processed" / "bakery_hour_profile.csv"
)
DEFAULT_PROFILE_TABLE = "sku_hour_share_profile_smoothed_embedded"
DEFAULT_UPLIFT_TABLE = "sku_hour_uplift_multiplier_embedded"
DEFAULT_ASSORTMENT_TABLE = "assortment_city_products"
DEFAULT_RECENT_SALES_TABLE = "mart_sales_60d"


def daterange(start: str, end: str):
    current = date.fromisoformat(start)
    stop = date.fromisoformat(end)
    while current <= stop:
        yield str(current)
        current += timedelta(days=1)


def _existing_lead1_dates(client, date_from: str, date_to: str) -> set[str]:
    df = client.query_df(
        """
        select forecast_date
        from bakery_forecast_day_snapshots
        where lead_days = 1
          and forecast_date between %(date_from)s and %(date_to)s
        group by forecast_date
        """,
        parameters={"date_from": date_from, "date_to": date_to},
    )
    if df.empty:
        return set()
    return {
        pd.Timestamp(value).date().isoformat()
        for value in df["forecast_date"].tolist()
    }


def _load_dataset_until(
    dataset_path: Path,
    forecast_date: str,
    output_path: Path,
    bakery_ids: set[int] | None = None,
) -> int:
    df = pd.read_csv(dataset_path, encoding="utf-8-sig")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    cutoff = pd.Timestamp(forecast_date)
    history = df[df[DATE_COL] < cutoff].copy()
    if bakery_ids:
        history = history[history[BAKERY_ID_COL].isin(bakery_ids)].copy()
    if history.empty:
        raise ValueError(f"No history before {forecast_date}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(output_path, index=False, encoding="utf-8-sig")
    return int(len(history))


def _resolve_bias_path(path: str | Path | None) -> Path:
    if path:
        return Path(path)
    if DEFAULT_BIAS_PATH.exists():
        return DEFAULT_BIAS_PATH
    return FALLBACK_BIAS_PATH


def _resolve_backfill_bias_path(
    args: argparse.Namespace, forecast_date: str, history_path: Path, output_dir: Path
) -> Path:
    static_path = _resolve_bias_path(args.bias_path)
    if not getattr(args, "use_rolling_bias", False):
        return static_path

    filter_ids = getattr(args, "bakery_ids", None)
    bakery_ids = sorted(
        id_
        for id_ in pd.read_csv(history_path, encoding="utf-8-sig", usecols=[BAKERY_ID_COL])[
            BAKERY_ID_COL
        ]
        .dropna()
        .unique()
        .tolist()
        if not filter_ids or id_ in filter_ids
    )
    static_bias_df = load_bias_table(static_path)
    effective = build_rolling_bias_table(
        env_file=args.env_file,
        bakery_ids=bakery_ids,
        as_of_date=forecast_date,
        static_bias_df=static_bias_df,
        trailing_days=args.rolling_bias_days,
        min_days=args.rolling_bias_min_days,
        bakery_forecast_table=table_name(
            "bakery_forecast_day_snapshots", args.table_suffix
        ),
    )
    date_part = forecast_date.replace("-", "")
    effective_path = output_dir / f"_lead1_rolling_bias_{date_part}.csv"
    effective.to_csv(effective_path, index=False, encoding="utf-8-sig")
    return effective_path


def build_day(args: argparse.Namespace, forecast_date: str) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    date_part = forecast_date.replace("-", "")
    history_path = output_dir / f"_lead1_history_until_{date_part}.csv"
    bakery_path = output_dir / f"bakery_day_forecast_prod_lead1_{date_part}.csv"

    history_rows = _load_dataset_until(
        Path(args.dataset_path),
        forecast_date,
        history_path,
        bakery_ids=getattr(args, "bakery_ids", None),
    )
    apply_bias = not getattr(args, "no_bias_correction", False)
    bias_path = (
        _resolve_backfill_bias_path(args, forecast_date, history_path, output_dir)
        if apply_bias
        else None
    )
    run_forecast_mode(
        argparse.Namespace(
            dataset_path=str(history_path),
            model_path=args.model_path,
            meta_path=args.meta_path,
            bias_path=str(bias_path) if bias_path else "",
            output_path=str(bakery_path),
            weather_path=args.weather_path,
            horizon_days=1,
            start_date=forecast_date,
            apply_bias_correction=apply_bias,
            bias_clip_pct=args.bias_clip_pct,
        )
    )

    use_raw = getattr(args, "use_raw_uplift_multiplier", False)
    allocated = allocate_from_clickhouse(
        bakery_forecast_path=bakery_path,
        bakery_hour_profile_path=args.bakery_hour_profile_path,
        output_dir=output_dir,
        env_file=args.env_file,
        profile_table=table_name(args.profile_table, args.table_suffix),
        uplift_table=table_name(args.uplift_table, args.table_suffix),
        forecast_col=FORECAST_BIAS_ADJ_COL if apply_bias else FORECAST_COL,
        output_suffix=f"prod_lead1_{date_part}",
        uplift_profile_version=args.uplift_profile_version,
        recent_correction_mode=args.recent_correction_mode,
        recent_correction_days=args.recent_correction_days,
        recent_sales_table=args.recent_sales_table,
        assortment_table=table_name(args.assortment_table, args.table_suffix),
        disable_assortment_filter=args.disable_assortment_filter,
        disable_assortment_renormalization=args.disable_assortment_renormalization,
        use_raw_uplift_multiplier=use_raw,
        max_sku_uplift_ratio=args.max_sku_uplift_ratio,
        hierarchical_haircut_target_ratio=args.hierarchical_haircut_target_ratio,
        hierarchical_haircut_history_days=args.hierarchical_haircut_history_days,
        hierarchical_haircut_pair_prior_days=args.hierarchical_haircut_pair_prior_days,
        hierarchical_haircut_min_coefficient=args.hierarchical_haircut_min_coefficient,
    )

    model_is_base = Path(args.model_path).resolve() == BASE_MODEL_PATH.resolve()
    if not apply_bias:
        bias_suffix = "_nocorrection"
    elif getattr(args, "use_rolling_bias", False):
        bias_suffix = "_rollingbias"
    else:
        bias_suffix = ""
    if use_raw:
        run_id = f"backfill_base_bakery_raw_uplift_sku{bias_suffix}_{date_part}_h1"
        model_version = "bakery_day_lgbm_base_lead1_backfill"
    elif model_is_base:
        run_id = f"backfill_base_bakery_no_sku_uplift{bias_suffix}_{date_part}_h1"
        model_version = "bakery_day_lgbm_base_lead1_backfill"
    else:
        run_id = f"backfill_uplifted_bakery_norm_uplift_sku{bias_suffix}_{date_part}_h1"
        model_version = "bakery_day_lgbm_uplifted_lead1_backfill"

    loaded = load_forecast_run(
        env_file=args.env_file,
        bakery_path=bakery_path,
        sku_day_path=allocated["sku_daily"],
        sku_hour_path=allocated["sku_hourly"],
        profile_table=table_name(args.profile_table, args.table_suffix),
        lookup_source="clickhouse",
        run_id=run_id,
        model_version=model_version,
        profile_version=args.uplift_profile_version,
        notes=f"Lead-1 model backfill for {forecast_date}",
        replace_existing=True,
        weather_path=args.weather_path,
    )

    return {
        "date": forecast_date,
        "run_id": run_id,
        "history_rows": history_rows,
        "allocated": {key: str(value) for key, value in allocated.items()},
        "loaded": loaded,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--date-from", required=True)
    parser.add_argument("--date-to", required=True)
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--meta-path", default=str(DEFAULT_META_PATH))
    parser.add_argument("--bias-path", default=None)
    parser.add_argument("--weather-path", default=str(DEFAULT_WEATHER_PATH))
    parser.add_argument(
        "--bakery-hour-profile-path",
        default=str(DEFAULT_BAKERY_HOUR_PROFILE_PATH),
    )
    parser.add_argument("--profile-table", default=DEFAULT_PROFILE_TABLE)
    parser.add_argument("--uplift-table", default=DEFAULT_UPLIFT_TABLE)
    parser.add_argument("--uplift-profile-version", required=True)
    parser.add_argument("--assortment-table", default=DEFAULT_ASSORTMENT_TABLE)
    parser.add_argument(
        "--recent-correction-mode",
        default="runner_city_prior_soft_weekpart",
    )
    parser.add_argument("--recent-correction-days", type=int, default=30)
    parser.add_argument("--recent-sales-table", default=DEFAULT_RECENT_SALES_TABLE)
    parser.add_argument("--bias-clip-pct", type=float, default=0.15)
    parser.add_argument(
        "--no-bias-correction",
        action="store_true",
        help="Skip bias correction entirely and use the raw model output as-is.",
    )
    parser.add_argument(
        "--use-rolling-bias",
        action="store_true",
        help=(
            "Replace the static bias.json snapshot with a trailing-window "
            "rolling correction recomputed for each backfilled date (see "
            "pipelines/forecast_publish/rolling_bakery_bias.py)."
        ),
    )
    parser.add_argument(
        "--rolling-bias-days", type=int, default=DEFAULT_ROLLING_BIAS_DAYS
    )
    parser.add_argument(
        "--rolling-bias-min-days", type=int, default=DEFAULT_ROLLING_BIAS_MIN_DAYS
    )
    parser.add_argument("--table-suffix", default="")
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument(
        "--summary-path",
        default="reports/prod_lead1_model_backfill_summary.json",
    )
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument(
        "--bakery-ids",
        default=None,
        help="Comma-separated bakery IDs to restrict the backfill (e.g. 16,20,21). "
             "Omit to run all bakeries.",
        type=lambda s: {int(x.strip()) for x in s.split(",") if x.strip()},
    )
    parser.add_argument("--disable-assortment-filter", action="store_true")
    parser.add_argument("--disable-assortment-renormalization", action="store_true")
    parser.add_argument("--use-raw-uplift-multiplier", action="store_true",
                        help="Use base model + raw uplift (base_raw_uplift scenario)")
    parser.add_argument(
        "--max-sku-uplift-ratio",
        type=float,
        default=None,
        help=(
            "Cap each per-(date, bakery, product) daily forecast to this "
            "multiple of its recent rolling daily mean."
        ),
    )
    parser.add_argument("--hierarchical-haircut-target-ratio", type=float, default=None)
    parser.add_argument("--hierarchical-haircut-history-days", type=int, default=7)
    parser.add_argument("--hierarchical-haircut-pair-prior-days", type=float, default=7.0)
    parser.add_argument("--hierarchical-haircut-min-coefficient", type=float, default=0.85)
    args = parser.parse_args()

    # When using base_raw scenario, default paths to base model unless overridden
    if args.use_raw_uplift_multiplier:
        if args.dataset_path == str(DEFAULT_DATASET_PATH):
            args.dataset_path = str(BASE_DATASET_PATH)
        if args.model_path == str(DEFAULT_MODEL_PATH):
            args.model_path = str(BASE_MODEL_PATH)
        if args.meta_path == str(DEFAULT_META_PATH):
            args.meta_path = str(BASE_META_PATH)

    client = create_client(args.env_file)
    existing = (
        set()
        if args.replace_existing
        else _existing_lead1_dates(client, args.date_from, args.date_to)
    )
    results = []
    for day in daterange(args.date_from, args.date_to):
        if day in existing:
            print(f"SKIP {day}: lead-1 snapshot already exists")
            continue
        started = time.monotonic()
        print(f"\n{'=' * 72}\nBuilding lead-1 backfill for {day}", flush=True)
        result = build_day(args, day)
        result["elapsed_sec"] = round(time.monotonic() - started, 1)
        results.append(result)
        print(
            f"DONE {day}: {result['run_id']} "
            f"bakery={result['loaded']['bakery_rows']} "
            f"sku_day={result['loaded']['sku_day_rows']} "
            f"sku_hour={result['loaded']['sku_hour_rows']}",
            flush=True,
        )

    summary = {"date_from": args.date_from, "date_to": args.date_to, "days": results}
    summary_path = ROOT / args.summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSummary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
