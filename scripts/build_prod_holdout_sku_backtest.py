"""Build a 30-day SKU backtest for the current production allocation setup.

This is intentionally tied to the serving configuration used by the embedded
app:

* bakery forecast: uplifted bakery holdout predictions
* SKU allocation: ClickHouse-backed smoothed SKU profile, normalized within
  trusted bakery/dow/hour buckets, with the same fallback rules as production
* actuals: mart_sales_60d, the same raw fact source used by the embedded UI

Outputs are written under reports/prod_holdout_sku_backtest/.
"""

from __future__ import annotations

# ruff: noqa: E402,E501

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipelines.forecast_publish.load_forecast_run import DEFAULT_ENV_PATH
from pipelines.forecast_publish.load_forecast_run import create_client
from pipelines.forecast_publish.sku_hour_profile_store import PROFILE_TABLE
from src.experiments_v2.apply_bakery_profiles import DEFAULT_BAKERY_HOUR_PROFILE_PATH
from src.experiments_v2.apply_bakery_profiles_clickhouse import allocate_from_clickhouse


DEFAULT_HOLDOUT_PATH = REPO_ROOT / "reports" / "bakery_day_model_uplifted_holdout_predictions.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "prod_holdout_sku_backtest"
DEFAULT_PROCESSED_OUTPUT_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_START_DATE = "2026-05-02"
DEFAULT_END_DATE = "2026-05-31"
DEFAULT_SUFFIX = "holdout_30d_prod_uplifted_norm"
SALES_LINE_TABLE = "mart_sales_60d"


def _read_holdout(path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"date", "bakery_id", "bakery_name", "city", "bakery_sales", "bakery_day_forecast"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    if df.empty:
        raise ValueError(f"No holdout rows in {start_date}..{end_date}")

    return df.sort_values(["bakery_id", "date"]).reset_index(drop=True)


def _write_bakery_allocator_input(holdout: pd.DataFrame, path: Path) -> Path:
    work = holdout[
        ["date", "bakery_id", "bakery_name", "city", "bakery_day_forecast"]
    ].copy()
    work["date"] = work["date"].dt.date
    path.parent.mkdir(parents=True, exist_ok=True)
    work.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _query_actual_sku(
    *,
    env_file: str | Path,
    start_date: str,
    end_date: str,
    table: str,
) -> pd.DataFrame:
    client = create_client(env_file)
    query = f"""
        select
            check_date as date,
            toInt64OrNull(toString(bakery_id)) as bakery_id,
            any(bakery_name) as actual_bakery_name,
            any(city) as actual_city,
            toInt64OrNull(toString(product_id)) as product_id,
            any(product_name) as actual_product_name,
            any(category_name) as actual_category_name,
            sum(toFloat64(quantity)) as fact_qty,
            sum(toFloat64(line_amount)) as fact_revenue
        from {table}
        where check_date between %(start_date)s and %(end_date)s
          and toInt64OrNull(toString(bakery_id)) is not null
          and toInt64OrNull(toString(product_id)) is not null
        group by date, bakery_id, product_id
    """
    return client.query_df(
        query,
        parameters={"start_date": start_date, "end_date": end_date},
    )


def _query_product_lookup(*, env_file: str | Path, profile_table: str) -> pd.DataFrame:
    client = create_client(env_file)
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
    )


def _add_metrics(compare: pd.DataFrame) -> pd.DataFrame:
    work = compare.copy()
    work["fact_qty"] = pd.to_numeric(work["fact_qty"], errors="coerce").fillna(0.0)
    work["fact_revenue"] = pd.to_numeric(work["fact_revenue"], errors="coerce").fillna(0.0)
    work["forecast_qty"] = pd.to_numeric(work["forecast_qty"], errors="coerce").fillna(0.0)

    totals = (
        work.groupby(["date", "bakery_id"], as_index=False)
        .agg(
            bakery_fact_qty=("fact_qty", "sum"),
            bakery_forecast_qty=("forecast_qty", "sum"),
        )
    )
    work = work.merge(totals, on=["date", "bakery_id"], how="left", validate="many_to_one")
    work["fact_scale_to_forecast"] = np.where(
        work["bakery_fact_qty"] > 0,
        work["bakery_forecast_qty"] / work["bakery_fact_qty"],
        1.0,
    )
    work["fact_scaled_to_forecast_total"] = work["fact_qty"] * work["fact_scale_to_forecast"]

    work["err_raw_fact"] = work["forecast_qty"] - work["fact_qty"]
    work["abs_err_raw_fact"] = work["err_raw_fact"].abs()
    work["err_scaled_fact"] = work["forecast_qty"] - work["fact_scaled_to_forecast_total"]
    work["abs_err_scaled_fact"] = work["err_scaled_fact"].abs()

    work["forecast_share_in_bakery_day"] = np.where(
        work["bakery_forecast_qty"] > 0,
        work["forecast_qty"] / work["bakery_forecast_qty"],
        0.0,
    )
    work["fact_share_in_bakery_day"] = np.where(
        work["bakery_fact_qty"] > 0,
        work["fact_qty"] / work["bakery_fact_qty"],
        0.0,
    )
    work["share_delta"] = work["forecast_share_in_bakery_day"] - work["fact_share_in_bakery_day"]

    work["cell_type"] = np.select(
        [
            (work["forecast_qty"] > 0) & (work["fact_qty"] > 0),
            (work["forecast_qty"] > 0) & (work["fact_qty"] <= 0),
            (work["forecast_qty"] <= 0) & (work["fact_qty"] > 0),
        ],
        ["both_positive", "forecast_only_fact_zero", "fact_only_forecast_zero"],
        default="both_zero",
    )

    pair_totals = (
        work.groupby(["bakery_id", "product_id"], as_index=False)
        .agg(
            pair_window_fact_qty=("fact_qty", "sum"),
            pair_window_forecast_qty=("forecast_qty", "sum"),
            pair_window_days_fact_positive=("fact_qty", lambda s: int((s > 0).sum())),
            pair_window_days_forecast_positive=("forecast_qty", lambda s: int((s > 0).sum())),
        )
    )
    work = work.merge(pair_totals, on=["bakery_id", "product_id"], how="left", validate="many_to_one")
    work["dead_pair_window"] = (
        (work["pair_window_fact_qty"] <= 0)
        & (work["pair_window_forecast_qty"] > 0)
    )
    return work


def _metrics_block(df: pd.DataFrame, fact_col: str, abs_err_col: str, err_col: str) -> dict[str, float]:
    denom = float(df[fact_col].sum())
    return {
        "rows": int(len(df)),
        "fact_sum": denom,
        "forecast_sum": float(df["forecast_qty"].sum()),
        "mae": float(df[abs_err_col].mean()) if len(df) else 0.0,
        "wmape_pct": float(df[abs_err_col].sum() / denom * 100) if denom > 0 else None,
        "bias_mean": float(df[err_col].mean()) if len(df) else 0.0,
        "bias_sum": float(df[err_col].sum()),
    }


def _write_aggregates(compare: pd.DataFrame, out_dir: Path) -> dict:
    by_day = (
        compare.groupby("date", as_index=False)
        .agg(
            fact_qty=("fact_qty", "sum"),
            forecast_qty=("forecast_qty", "sum"),
            abs_err_raw_fact=("abs_err_raw_fact", "sum"),
            abs_err_scaled_fact=("abs_err_scaled_fact", "sum"),
            dead_pair_forecast_qty=("forecast_qty", lambda s: float(s[compare.loc[s.index, "dead_pair_window"]].sum())),
        )
    )
    by_day["wmape_raw_fact_pct"] = by_day["abs_err_raw_fact"] / by_day["fact_qty"].replace(0, np.nan) * 100
    by_day.to_csv(out_dir / "by_day.csv", index=False, encoding="utf-8-sig")

    by_bakery = (
        compare.groupby(["bakery_id", "bakery_name", "city"], as_index=False, dropna=False)
        .agg(
            fact_qty=("fact_qty", "sum"),
            forecast_qty=("forecast_qty", "sum"),
            abs_err_raw_fact=("abs_err_raw_fact", "sum"),
            abs_err_scaled_fact=("abs_err_scaled_fact", "sum"),
            dead_pair_forecast_qty=("forecast_qty", lambda s: float(s[compare.loc[s.index, "dead_pair_window"]].sum())),
        )
    )
    by_bakery["wmape_raw_fact_pct"] = by_bakery["abs_err_raw_fact"] / by_bakery["fact_qty"].replace(0, np.nan) * 100
    by_bakery["bias_qty"] = by_bakery["forecast_qty"] - by_bakery["fact_qty"]
    by_bakery.to_csv(out_dir / "by_bakery.csv", index=False, encoding="utf-8-sig")

    by_pair = (
        compare.groupby(
            ["bakery_id", "bakery_name", "city", "product_id", "product_name", "category_name"],
            as_index=False,
            dropna=False,
        )
        .agg(
            fact_qty=("fact_qty", "sum"),
            fact_revenue=("fact_revenue", "sum"),
            forecast_qty=("forecast_qty", "sum"),
            abs_err_raw_fact=("abs_err_raw_fact", "sum"),
            abs_err_scaled_fact=("abs_err_scaled_fact", "sum"),
            days_fact_positive=("fact_qty", lambda s: int((s > 0).sum())),
            days_forecast_positive=("forecast_qty", lambda s: int((s > 0).sum())),
        )
    )
    by_pair["bias_qty"] = by_pair["forecast_qty"] - by_pair["fact_qty"]
    by_pair["wmape_raw_fact_pct"] = by_pair["abs_err_raw_fact"] / by_pair["fact_qty"].replace(0, np.nan) * 100
    by_pair["dead_pair_window"] = (by_pair["fact_qty"] <= 0) & (by_pair["forecast_qty"] > 0)
    by_pair.to_csv(out_dir / "by_bakery_sku.csv", index=False, encoding="utf-8-sig")

    cell = (
        compare.groupby("cell_type", as_index=False)
        .agg(
            rows=("cell_type", "size"),
            fact_qty=("fact_qty", "sum"),
            forecast_qty=("forecast_qty", "sum"),
            abs_err_raw_fact=("abs_err_raw_fact", "sum"),
            abs_err_scaled_fact=("abs_err_scaled_fact", "sum"),
        )
        .sort_values("cell_type")
    )
    cell.to_csv(out_dir / "cell_type_summary.csv", index=False, encoding="utf-8-sig")

    top_under = by_pair[by_pair["fact_qty"] > 0].sort_values("bias_qty").head(100)
    top_under.to_csv(out_dir / "top_underforecast_bakery_sku.csv", index=False, encoding="utf-8-sig")

    dead = by_pair[by_pair["dead_pair_window"]].sort_values("forecast_qty", ascending=False)
    dead.head(200).to_csv(out_dir / "top_dead_pair_forecast.csv", index=False, encoding="utf-8-sig")

    return {
        "by_day_rows": int(len(by_day)),
        "by_bakery_rows": int(len(by_bakery)),
        "by_bakery_sku_rows": int(len(by_pair)),
        "cell_type_rows": int(len(cell)),
    }


def build_backtest(args: argparse.Namespace) -> dict:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_out = Path(args.processed_output_dir)
    processed_out.mkdir(parents=True, exist_ok=True)

    holdout = _read_holdout(Path(args.holdout_path), args.start_date, args.end_date)
    bakery_input_path = out_dir / "bakery_holdout_uplifted_allocator_input.csv"
    _write_bakery_allocator_input(holdout, bakery_input_path)

    allocated_paths = allocate_from_clickhouse(
        bakery_forecast_path=bakery_input_path,
        bakery_hour_profile_path=args.bakery_hour_profile_path,
        output_dir=processed_out,
        env_file=args.env_file,
        profile_table=args.profile_table,
        forecast_col="bakery_day_forecast",
        output_suffix=args.output_suffix,
        use_raw_uplift_multiplier=False,
        uplift_profile_version=args.uplift_profile_version,
        chunk_size=args.chunk_size,
    )

    sku_day = pd.read_csv(allocated_paths["sku_daily"], encoding="utf-8-sig")
    sku_day["date"] = pd.to_datetime(sku_day["date"], errors="coerce").dt.date
    sku_day = sku_day.rename(columns={"sku_day_forecast": "forecast_qty"})

    fact = _query_actual_sku(
        env_file=args.env_file,
        start_date=args.start_date,
        end_date=args.end_date,
        table=args.sales_table,
    )
    fact["date"] = pd.to_datetime(fact["date"], errors="coerce").dt.date

    lookup = _query_product_lookup(env_file=args.env_file, profile_table=args.profile_table)
    compare = sku_day.merge(
        lookup,
        on=["bakery_id", "product_id"],
        how="left",
        validate="many_to_one",
    )
    compare = compare.merge(
        fact,
        on=["date", "bakery_id", "product_id"],
        how="outer",
        validate="one_to_one",
    )

    bakery_lookup = holdout[["bakery_id", "bakery_name", "city"]].drop_duplicates("bakery_id")
    compare = compare.merge(bakery_lookup, on="bakery_id", how="left", validate="many_to_one")
    compare["bakery_name"] = compare["bakery_name"].fillna(compare["actual_bakery_name"])
    compare["city"] = compare["city"].fillna(compare["actual_city"])
    compare["product_name"] = compare["product_name"].fillna(compare["actual_product_name"])
    compare["category_name"] = compare["category_name"].fillna(compare["actual_category_name"])

    compare = _add_metrics(compare)
    keep_cols = [
        "date",
        "bakery_id",
        "bakery_name",
        "city",
        "product_id",
        "product_name",
        "category_name",
        "fact_qty",
        "fact_revenue",
        "forecast_qty",
        "fact_scaled_to_forecast_total",
        "err_raw_fact",
        "abs_err_raw_fact",
        "err_scaled_fact",
        "abs_err_scaled_fact",
        "bakery_fact_qty",
        "bakery_forecast_qty",
        "forecast_share_in_bakery_day",
        "fact_share_in_bakery_day",
        "share_delta",
        "cell_type",
        "pair_window_fact_qty",
        "pair_window_forecast_qty",
        "pair_window_days_fact_positive",
        "pair_window_days_forecast_positive",
        "dead_pair_window",
    ]
    compare = compare[keep_cols].sort_values(["bakery_id", "product_id", "date"]).reset_index(drop=True)
    compare_path = out_dir / "prod_holdout_sku_compare.csv"
    compare.to_csv(compare_path, index=False, encoding="utf-8-sig")

    aggregate_info = _write_aggregates(compare, out_dir)

    dead_pair_forecast_sum = float(compare.loc[compare["dead_pair_window"], "forecast_qty"].sum())
    summary = {
        "window": {"start": args.start_date, "end": args.end_date},
        "configuration": {
            "bakery_holdout_path": str(Path(args.holdout_path)),
            "scenario": "uplifted bakery forecast + normalized production SKU allocation",
            "profile_table": args.profile_table,
            "sales_table": args.sales_table,
            "fact_source_note": "Raw mart_sales_60d actuals, matching embedded UI actual_qty.",
        },
        "paths": {
            "bakery_allocator_input": str(bakery_input_path),
            "sku_day_forecast": str(allocated_paths["sku_daily"]),
            "sku_hour_forecast": str(allocated_paths["sku_hourly"]),
            "allocation_summary": str(allocated_paths["summary"]),
            "compare": str(compare_path),
        },
        "rows": {
            "compare": int(len(compare)),
            "days": int(pd.Series(compare["date"]).nunique()),
            "bakeries": int(compare["bakery_id"].nunique()),
            "products": int(compare["product_id"].nunique()),
            **aggregate_info,
        },
        "metrics_vs_raw_app_fact": _metrics_block(
            compare,
            "fact_qty",
            "abs_err_raw_fact",
            "err_raw_fact",
        ),
        "metrics_vs_fact_scaled_to_forecast_total": _metrics_block(
            compare,
            "fact_scaled_to_forecast_total",
            "abs_err_scaled_fact",
            "err_scaled_fact",
        ),
        "dead_pair_leakage": {
            "forecast_qty_on_pairs_with_zero_window_fact": dead_pair_forecast_sum,
            "forecast_share_pct": (
                dead_pair_forecast_sum / float(compare["forecast_qty"].sum()) * 100
                if compare["forecast_qty"].sum() > 0
                else 0.0
            ),
            "pairs": int(
                compare.loc[compare["dead_pair_window"], ["bakery_id", "product_id"]]
                .drop_duplicates()
                .shape[0]
            ),
        },
        "cell_type_summary": (
            compare.groupby("cell_type")
            .agg(
                rows=("cell_type", "size"),
                fact_qty=("fact_qty", "sum"),
                forecast_qty=("forecast_qty", "sum"),
                abs_err_raw_fact=("abs_err_raw_fact", "sum"),
            )
            .to_dict(orient="index")
        ),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build production-style SKU holdout backtest")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH))
    parser.add_argument("--holdout-path", default=str(DEFAULT_HOLDOUT_PATH))
    parser.add_argument("--bakery-hour-profile-path", default=str(DEFAULT_BAKERY_HOUR_PROFILE_PATH))
    parser.add_argument("--profile-table", default=PROFILE_TABLE)
    parser.add_argument("--sales-table", default=SALES_LINE_TABLE)
    parser.add_argument("--processed-output-dir", default=str(DEFAULT_PROCESSED_OUTPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-suffix", default=DEFAULT_SUFFIX)
    parser.add_argument("--uplift-profile-version", default=None)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_backtest(args)
    print("=" * 72)
    print("PRODUCTION SKU HOLDOUT BACKTEST")
    print("=" * 72)
    print(f"window: {summary['window']['start']} .. {summary['window']['end']}")
    print(f"compare: {summary['paths']['compare']}")
    print(f"rows: {summary['rows']['compare']}")
    print(f"bakeries: {summary['rows']['bakeries']}")
    print(f"products: {summary['rows']['products']}")
    print(f"raw fact WMAPE: {summary['metrics_vs_raw_app_fact']['wmape_pct']:.4f}%")
    print(
        "scaled fact WMAPE: "
        f"{summary['metrics_vs_fact_scaled_to_forecast_total']['wmape_pct']:.4f}%"
    )
    print(
        "dead-pair forecast share: "
        f"{summary['dead_pair_leakage']['forecast_share_pct']:.4f}%"
    )


if __name__ == "__main__":
    main()
