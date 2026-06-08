"""Evaluate post-allocation SKU share correction variants on holdout data.

The script starts from the production-style SKU holdout compare file built by
scripts/build_prod_holdout_sku_backtest.py. It does not retrain the bakery
model. Each variant changes only how the bakery-day forecast is distributed
across SKU for a bakery-day, then renormalizes to preserve the original
bakery-day forecast total.

Recent assortment statistics are queried from mart_sales_60d for the period
before the holdout window, so the experiment does not use holdout facts to build
the correction.
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


DEFAULT_COMPARE_PATH = (
    REPO_ROOT / "reports" / "prod_holdout_sku_backtest" / "prod_holdout_sku_compare.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "prod_holdout_sku_backtest_variants"
DEFAULT_START_DATE = "2026-05-02"
DEFAULT_END_DATE = "2026-05-31"
DEFAULT_RECENT_DAYS = 30
SALES_LINE_TABLE = "mart_sales_60d"


def _query_recent_stats(
    *,
    env_file: str | Path,
    recent_start: str,
    recent_end: str,
    table: str,
) -> pd.DataFrame:
    client = create_client(env_file)
    query = f"""
        select
            toInt64OrNull(toString(bakery_id)) as bakery_id,
            toInt64OrNull(toString(product_id)) as product_id,
            any(product_name) as recent_product_name,
            any(category_name) as recent_category_name,
            sum(toFloat64(quantity)) as recent_qty,
            uniqExact(check_date) as recent_days_sold
        from {table}
        where check_date between %(recent_start)s and %(recent_end)s
          and toInt64OrNull(toString(bakery_id)) is not null
          and toInt64OrNull(toString(product_id)) is not null
          and toFloat64(quantity) > 0
        group by bakery_id, product_id
    """
    stats = client.query_df(
        query,
        parameters={"recent_start": recent_start, "recent_end": recent_end},
    )
    bakery_totals = (
        stats.groupby("bakery_id", as_index=False)["recent_qty"]
        .sum()
        .rename(columns={"recent_qty": "bakery_recent_qty"})
    )
    stats = stats.merge(bakery_totals, on="bakery_id", how="left", validate="many_to_one")
    stats["recent_share"] = np.where(
        stats["bakery_recent_qty"] > 0,
        stats["recent_qty"] / stats["bakery_recent_qty"],
        0.0,
    )
    return stats


def _attach_recent(df: pd.DataFrame, recent: pd.DataFrame) -> pd.DataFrame:
    work = df.merge(
        recent,
        on=["bakery_id", "product_id"],
        how="left",
        validate="many_to_one",
    )
    work["recent_qty"] = pd.to_numeric(work["recent_qty"], errors="coerce").fillna(0.0)
    work["recent_days_sold"] = (
        pd.to_numeric(work["recent_days_sold"], errors="coerce").fillna(0).astype("int64")
    )
    work["recent_share"] = pd.to_numeric(work["recent_share"], errors="coerce").fillna(0.0)
    return work


def _renormalize_preserving_bakery_day(
    work: pd.DataFrame,
    raw_col: str,
    out_col: str,
) -> pd.DataFrame:
    raw_sum = (
        work.groupby(["date", "bakery_id"], as_index=False)[raw_col]
        .sum()
        .rename(columns={raw_col: "_raw_variant_sum"})
    )
    work = work.merge(raw_sum, on=["date", "bakery_id"], how="left", validate="many_to_one")
    base_sum = pd.to_numeric(work["bakery_forecast_qty"], errors="coerce").fillna(0.0)
    raw_sum_series = pd.to_numeric(work["_raw_variant_sum"], errors="coerce").fillna(0.0)
    fallback = pd.to_numeric(work["forecast_qty"], errors="coerce").fillna(0.0)
    work[out_col] = np.where(
        raw_sum_series > 0,
        pd.to_numeric(work[raw_col], errors="coerce").fillna(0.0) / raw_sum_series * base_sum,
        fallback,
    )
    return work.drop(columns=["_raw_variant_sum"])


def _apply_variant(df: pd.DataFrame, name: str) -> pd.DataFrame:
    work = df.copy()
    base = pd.to_numeric(work["forecast_qty"], errors="coerce").fillna(0.0)
    base_total = pd.to_numeric(work["bakery_forecast_qty"], errors="coerce").fillna(0.0)
    recent_share_qty = work["recent_share"] * base_total

    if name == "baseline":
        work["forecast_variant"] = base
        return work

    if name == "dead_0d":
        active = work["recent_days_sold"] > 0
        work["_raw_variant"] = np.where(active, base, 0.0)
    elif name == "active_3d":
        active = (work["recent_days_sold"] >= 3) | (work["recent_qty"] >= 10)
        work["_raw_variant"] = np.where(active, base, 0.0)
    elif name == "blend_recent_50":
        active = work["recent_days_sold"] > 0
        blended = 0.5 * base + 0.5 * recent_share_qty
        work["_raw_variant"] = np.where(active, blended, 0.0)
    elif name == "core_recent_70":
        active = work["recent_days_sold"] > 0
        core = (work["recent_days_sold"] >= 20) & (work["recent_share"] >= 0.01)
        blended_core = 0.3 * base + 0.7 * recent_share_qty
        blended_regular = 0.7 * base + 0.3 * recent_share_qty
        work["_raw_variant"] = np.where(core, blended_core, blended_regular)
        work["_raw_variant"] = np.where(active, work["_raw_variant"], 0.0)
    else:
        raise ValueError(f"Unknown variant: {name}")

    work = _renormalize_preserving_bakery_day(work, "_raw_variant", "forecast_variant")
    return work.drop(columns=["_raw_variant"])


def _score_variant(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, dict]:
    work = df.copy()
    fact = pd.to_numeric(work["fact_qty"], errors="coerce").fillna(0.0)
    forecast = pd.to_numeric(work["forecast_variant"], errors="coerce").fillna(0.0)

    bakery_totals = (
        work.assign(forecast_variant=forecast, fact_qty=fact)
        .groupby(["date", "bakery_id"], as_index=False)
        .agg(
            bakery_fact_qty=("fact_qty", "sum"),
            bakery_forecast_variant_qty=("forecast_variant", "sum"),
        )
    )
    work = work.drop(columns=["bakery_fact_qty"], errors="ignore").merge(
        bakery_totals,
        on=["date", "bakery_id"],
        how="left",
        validate="many_to_one",
    )
    scale = np.where(
        work["bakery_fact_qty"] > 0,
        work["bakery_forecast_variant_qty"] / work["bakery_fact_qty"],
        1.0,
    )
    scaled_fact = fact * scale
    err_raw = forecast - fact
    err_scaled = forecast - scaled_fact

    work["forecast_variant"] = forecast
    work["fact_scaled_to_variant_total"] = scaled_fact
    work["err_raw_fact"] = err_raw
    work["abs_err_raw_fact"] = np.abs(err_raw)
    work["err_scaled_fact"] = err_scaled
    work["abs_err_scaled_fact"] = np.abs(err_scaled)
    work["cell_type_variant"] = np.select(
        [
            (forecast > 0) & (fact > 0),
            (forecast > 0) & (fact <= 0),
            (forecast <= 0) & (fact > 0),
        ],
        ["both_positive", "forecast_only_fact_zero", "fact_only_forecast_zero"],
        default="both_zero",
    )

    pair_totals = (
        work.groupby(["bakery_id", "product_id"], as_index=False)
        .agg(
            pair_window_fact_qty=("fact_qty", "sum"),
            pair_window_forecast_variant_qty=("forecast_variant", "sum"),
        )
    )
    work = work.drop(
        columns=["pair_window_fact_qty", "pair_window_forecast_variant_qty"],
        errors="ignore",
    ).merge(pair_totals, on=["bakery_id", "product_id"], how="left", validate="many_to_one")
    work["dead_pair_window_variant"] = (
        (work["pair_window_fact_qty"] <= 0)
        & (work["pair_window_forecast_variant_qty"] > 0)
    )

    fact_sum = float(fact.sum())
    scaled_fact_sum = float(scaled_fact.sum())
    dead_sum = float(work.loc[work["dead_pair_window_variant"], "forecast_variant"].sum())
    summary = {
        "variant": name,
        "rows": int(len(work)),
        "fact_sum": fact_sum,
        "forecast_sum": float(forecast.sum()),
        "wmape_raw_fact_pct": float(work["abs_err_raw_fact"].sum() / fact_sum * 100) if fact_sum else None,
        "wmape_scaled_fact_pct": float(work["abs_err_scaled_fact"].sum() / scaled_fact_sum * 100)
        if scaled_fact_sum
        else None,
        "bias_sum": float(err_raw.sum()),
        "dead_pair_forecast_qty": dead_sum,
        "dead_pair_forecast_share_pct": float(dead_sum / forecast.sum() * 100) if forecast.sum() else 0.0,
        "forecast_only_fact_zero_qty": float(
            work.loc[work["cell_type_variant"] == "forecast_only_fact_zero", "forecast_variant"].sum()
        ),
        "fact_only_forecast_zero_qty": float(
            work.loc[work["cell_type_variant"] == "fact_only_forecast_zero", "fact_qty"].sum()
        ),
    }
    return work, summary


def _write_variant_artifacts(scored: pd.DataFrame, name: str, out_dir: Path) -> None:
    by_pair = (
        scored.groupby(
            ["bakery_id", "bakery_name", "city", "product_id", "product_name", "category_name"],
            as_index=False,
            dropna=False,
        )
        .agg(
            fact_qty=("fact_qty", "sum"),
            forecast_variant=("forecast_variant", "sum"),
            abs_err_raw_fact=("abs_err_raw_fact", "sum"),
            abs_err_scaled_fact=("abs_err_scaled_fact", "sum"),
            recent_qty=("recent_qty", "max"),
            recent_days_sold=("recent_days_sold", "max"),
        )
    )
    by_pair["bias_qty"] = by_pair["forecast_variant"] - by_pair["fact_qty"]
    by_pair["dead_pair_window"] = (
        (by_pair["fact_qty"] <= 0) & (by_pair["forecast_variant"] > 0)
    )
    by_pair.to_csv(out_dir / f"{name}_by_bakery_sku.csv", index=False, encoding="utf-8-sig")
    by_pair[by_pair["fact_qty"] > 0].sort_values("bias_qty").head(100).to_csv(
        out_dir / f"{name}_top_underforecast.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_pair[by_pair["dead_pair_window"]].sort_values(
        "forecast_variant",
        ascending=False,
    ).head(100).to_csv(
        out_dir / f"{name}_top_dead_pair_forecast.csv",
        index=False,
        encoding="utf-8-sig",
    )


def evaluate(args: argparse.Namespace) -> dict:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    holdout_start = pd.Timestamp(args.start_date)
    recent_end = holdout_start - pd.Timedelta(days=1)
    recent_start = holdout_start - pd.Timedelta(days=args.recent_days)

    base = pd.read_csv(args.compare_path, parse_dates=["date"])
    recent = _query_recent_stats(
        env_file=args.env_file,
        recent_start=str(recent_start.date()),
        recent_end=str(recent_end.date()),
        table=args.sales_table,
    )
    recent.to_csv(out_dir / "recent_assortment_stats.csv", index=False, encoding="utf-8-sig")
    base = _attach_recent(base, recent)

    variants = ["baseline", "dead_0d", "active_3d", "blend_recent_50", "core_recent_70"]
    summaries = []
    for name in variants:
        variant_df = _apply_variant(base, name)
        scored, summary = _score_variant(variant_df, name)
        summaries.append(summary)
        if name in {"baseline", "blend_recent_50", "core_recent_70"}:
            slim = scored[
                [
                    "date",
                    "bakery_id",
                    "bakery_name",
                    "city",
                    "product_id",
                    "product_name",
                    "category_name",
                    "fact_qty",
                    "forecast_variant",
                    "fact_scaled_to_variant_total",
                    "err_raw_fact",
                    "err_scaled_fact",
                    "cell_type_variant",
                    "recent_qty",
                    "recent_days_sold",
                    "recent_share",
                ]
            ].sort_values(["bakery_id", "product_id", "date"])
            slim.to_csv(out_dir / f"{name}_compare.csv", index=False, encoding="utf-8-sig")
        _write_variant_artifacts(scored, name, out_dir)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "variant_summary.csv", index=False, encoding="utf-8-sig")

    result = {
        "holdout_window": {"start": args.start_date, "end": args.end_date},
        "recent_window": {
            "start": str(recent_start.date()),
            "end": str(recent_end.date()),
            "days": args.recent_days,
        },
        "compare_path": str(Path(args.compare_path)),
        "output_dir": str(out_dir),
        "variants": summaries,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate SKU allocation variants")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH))
    parser.add_argument("--compare-path", default=str(DEFAULT_COMPARE_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sales-table", default=SALES_LINE_TABLE)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--recent-days", type=int, default=DEFAULT_RECENT_DAYS)
    return parser


def main() -> None:
    result = evaluate(build_parser().parse_args())
    summary = pd.DataFrame(result["variants"])
    print("=" * 72)
    print("SKU ALLOCATION VARIANT EVALUATION")
    print("=" * 72)
    print(f"recent window: {result['recent_window']['start']} .. {result['recent_window']['end']}")
    print(f"output_dir: {result['output_dir']}")
    print(
        summary[
            [
                "variant",
                "wmape_raw_fact_pct",
                "wmape_scaled_fact_pct",
                "dead_pair_forecast_share_pct",
                "forecast_only_fact_zero_qty",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
