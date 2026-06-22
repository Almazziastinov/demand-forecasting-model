"""Audit a dev forecast run for selected bakeries against deduplicated raw sales."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipelines.forecast_publish.load_forecast_run import create_client  # noqa: E402


SALES_EVENT_HEX = "D09FD180D0BED0B4D0B0D0B6D0B0"
DEFAULT_BAKERIES = (20, 21, 22, 222)


def _safe_pct(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else 100.0 * numerator / denominator


def _validate_suffix(value: str) -> str:
    if value and not re.fullmatch(r"_[A-Za-z0-9_]+", value):
        raise ValueError(f"Invalid table suffix: {value}")
    return value


def _id_sql(bakery_ids: list[int]) -> str:
    if not bakery_ids:
        raise ValueError("At least one bakery id is required")
    return ", ".join(str(int(value)) for value in bakery_ids)


def load_forecast(
    client,
    *,
    run_id: str,
    suffix: str,
    bakery_ids: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = _id_sql(bakery_ids)
    day = client.query_df(
        f"""
        select
            d.forecast_date as date,
            d.bakery_id,
            any(b.bakery_name) as bakery_name,
            any(b.city) as city,
            d.product_id,
            any(d.product_name) as product_name,
            any(d.category_name) as category_name,
            sum(d.forecast_qty) as forecast_qty
        from sku_forecast_day_embedded{suffix} d
        left join bakery_forecast_day_embedded{suffix} b
          on b.run_id = d.run_id
         and b.forecast_date = d.forecast_date
         and b.bakery_id = d.bakery_id
        where d.run_id = %(run_id)s and d.bakery_id in ({ids})
        group by d.forecast_date, d.bakery_id, d.product_id
        """,
        parameters={"run_id": run_id},
    )
    hour = client.query_df(
        f"""
        select
            forecast_date as date,
            bakery_id,
            product_id,
            hour,
            sum(forecast_qty) as forecast_qty
        from sku_forecast_hour_embedded{suffix}
        where run_id = %(run_id)s and bakery_id in ({ids})
        group by forecast_date, bakery_id, product_id, hour
        """,
        parameters={"run_id": run_id},
    )
    return day, hour


def load_actual(
    client,
    *,
    date_from: str,
    date_to: str,
    bakery_ids: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = _id_sql(bakery_ids)
    raw_source = f"""
        select distinct
            check_datetime,
            check_date,
            bakery_id,
            product_id,
            quantity,
            cash_event_type
        from Svezhar.fct_check_lines
        where hex(cash_event_type) = %(sales_event_hex)s
          and check_date between %(date_from)s and %(date_to)s
          and toInt64OrNull(toString(bakery_id)) in ({ids})
    """
    parameters = {
        "sales_event_hex": SALES_EVENT_HEX,
        "date_from": date_from,
        "date_to": date_to,
    }
    day = client.query_df(
        f"""
        select
            f.check_date as date,
            toInt64OrNull(toString(f.bakery_id)) as bakery_id,
            toInt64OrNull(toString(f.product_id)) as product_id,
            any(dp.product_name) as actual_product_name,
            any(dp.category_name) as actual_category_name,
            sum(toFloat64(f.quantity)) as fact_qty
        from ({raw_source}) f
        any left join Svezhar.dim_products dp on dp.product_id = f.product_id
        group by check_date, bakery_id, product_id
        """,
        parameters=parameters,
    )
    hour = client.query_df(
        f"""
        select
            check_date as date,
            toInt64OrNull(toString(bakery_id)) as bakery_id,
            toInt64OrNull(toString(product_id)) as product_id,
            toHour(check_datetime) as hour,
            sum(toFloat64(quantity)) as fact_qty
        from ({raw_source})
        group by check_date, bakery_id, product_id, hour
        """,
        parameters=parameters,
    )
    return day, hour


def build_metrics(
    forecast: pd.DataFrame,
    actual: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["date", "bakery_id", "product_id"]
    compare = forecast.merge(actual, on=keys, how="outer")
    for col in ["forecast_qty", "fact_qty"]:
        compare[col] = pd.to_numeric(compare[col], errors="coerce").fillna(0.0)
    for col, actual_col in [
        ("product_name", "actual_product_name"),
        ("category_name", "actual_category_name"),
    ]:
        compare[col] = compare[col].fillna(compare[actual_col])
    for col in ["bakery_name", "city", "product_name", "category_name"]:
        compare[col] = compare.groupby("bakery_id")[col].transform(
            lambda values: values.ffill().bfill()
        )
        compare[col] = compare[col].fillna("")
    totals = compare.groupby(["date", "bakery_id"], as_index=False).agg(
        bakery_forecast=("forecast_qty", "sum"), bakery_fact=("fact_qty", "sum")
    )
    compare = compare.merge(totals, on=["date", "bakery_id"], how="left")
    compare["scaled_fact_qty"] = np.where(
        compare["bakery_fact"] > 0,
        compare["fact_qty"] * compare["bakery_forecast"] / compare["bakery_fact"],
        0.0,
    )
    compare["abs_error"] = (compare["forecast_qty"] - compare["fact_qty"]).abs()
    compare["allocation_abs_error"] = (
        compare["forecast_qty"] - compare["scaled_fact_qty"]
    ).abs()
    text = (compare["product_name"] + " " + compare["category_name"]).str.casefold()
    compare["is_pie"] = text.str.contains("пирог|пирож", regex=True)

    rows = []
    for bakery_id, group in compare.groupby("bakery_id"):
        forecast_total = float(group["forecast_qty"].sum())
        fact_total = float(group["fact_qty"].sum())
        pie = group[group["is_pie"]]
        rows.append(
            {
                "bakery_id": int(bakery_id),
                "bakery_name": group["bakery_name"].iloc[0],
                "forecast_total": forecast_total,
                "fact_total": fact_total,
                "bias_pct": _safe_pct(forecast_total - fact_total, fact_total),
                "sku_wmape_pct": _safe_pct(float(group["abs_error"].sum()), fact_total),
                "allocation_wmape_pct": _safe_pct(
                    float(group["allocation_abs_error"].sum()),
                    float(group["scaled_fact_qty"].sum()),
                ),
                "pie_forecast": float(pie["forecast_qty"].sum()),
                "pie_fact": float(pie["fact_qty"].sum()),
                "pie_bias_pct": _safe_pct(
                    float(pie["forecast_qty"].sum() - pie["fact_qty"].sum()),
                    float(pie["fact_qty"].sum()),
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("bakery_id"), compare


def filter_actual_to_assortment(
    actual: pd.DataFrame,
    forecast: pd.DataFrame,
    assortment_path: str | Path,
) -> pd.DataFrame:
    assortment = pd.read_csv(assortment_path, encoding="utf-8-sig")
    assortment = assortment[
        pd.to_numeric(assortment["is_active"], errors="coerce").fillna(0).eq(1)
    ].copy()
    assortment["product_id"] = pd.to_numeric(
        assortment["product_id"], errors="coerce"
    ).astype("Int64")
    allowed = assortment[["city", "product_id"]].dropna().drop_duplicates()
    bakery_city = forecast[["bakery_id", "city"]].dropna().drop_duplicates(
        "bakery_id"
    )
    work = actual.merge(bakery_city, on="bakery_id", how="left")
    work["product_id"] = pd.to_numeric(work["product_id"], errors="coerce").astype(
        "Int64"
    )
    return work.merge(allowed, on=["city", "product_id"], how="inner").drop(
        columns=["city"]
    )


def add_edge_hour_metrics(
    metrics: pd.DataFrame,
    forecast_hour: pd.DataFrame,
    actual_hour: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for bakery_id in metrics["bakery_id"]:
        forecast = forecast_hour[forecast_hour["bakery_id"].eq(bakery_id)]
        actual = actual_hour[actual_hour["bakery_id"].eq(bakery_id)]
        forecast_total = float(forecast["forecast_qty"].sum())
        actual_total = float(actual["fact_qty"].sum())
        rows.append(
            {
                "bakery_id": bakery_id,
                "forecast_05_22_share_pct": _safe_pct(
                    float(
                        forecast.loc[
                            forecast["hour"].isin([5, 22]), "forecast_qty"
                        ].sum()
                    ),
                    forecast_total,
                ),
                "fact_05_22_share_pct": _safe_pct(
                    float(actual.loc[actual["hour"].isin([5, 22]), "fact_qty"].sum()),
                    actual_total,
                ),
            }
        )
    return metrics.merge(pd.DataFrame(rows), on="bakery_id", how="left")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=".env.dev")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--table-suffix", default="_dev")
    parser.add_argument(
        "--bakery-ids", nargs="+", type=int, default=list(DEFAULT_BAKERIES)
    )
    parser.add_argument("--output-dir", default="reports/dev_assortment_run_audit")
    parser.add_argument("--assortment-path")
    args = parser.parse_args()

    suffix = _validate_suffix(args.table_suffix)
    client = create_client(args.env_file)
    forecast_day, forecast_hour = load_forecast(
        client, run_id=args.run_id, suffix=suffix, bakery_ids=args.bakery_ids
    )
    if forecast_day.empty:
        raise ValueError(f"No forecast rows for run {args.run_id}")
    date_from = str(pd.to_datetime(forecast_day["date"]).min().date())
    date_to = str(pd.to_datetime(forecast_day["date"]).max().date())
    actual_day, actual_hour = load_actual(
        client, date_from=date_from, date_to=date_to, bakery_ids=args.bakery_ids
    )
    if args.assortment_path:
        actual_day = filter_actual_to_assortment(
            actual_day, forecast_day, args.assortment_path
        )
    metrics, compare = build_metrics(forecast_day, actual_day)
    metrics = add_edge_hour_metrics(metrics, forecast_hour, actual_hour)
    product_metrics = (
        compare.groupby(
            [
                "bakery_id",
                "bakery_name",
                "product_id",
                "product_name",
                "category_name",
                "is_pie",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            forecast_qty=("forecast_qty", "sum"),
            fact_qty=("fact_qty", "sum"),
            abs_error=("abs_error", "sum"),
            allocation_abs_error=("allocation_abs_error", "sum"),
        )
        .sort_values(["bakery_id", "allocation_abs_error"], ascending=[True, False])
    )
    product_metrics["bias_qty"] = (
        product_metrics["forecast_qty"] - product_metrics["fact_qty"]
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "bakery_metrics.csv", index=False, encoding="utf-8-sig")
    compare.to_csv(out_dir / "sku_day_compare.csv", index=False, encoding="utf-8-sig")
    product_metrics.to_csv(
        out_dir / "product_metrics.csv", index=False, encoding="utf-8-sig"
    )
    print(metrics.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
