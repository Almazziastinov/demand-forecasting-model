"""Diagnose whether confirmed stockout underforecasts come from SKU allocation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.export_clickhouse_checks import create_client  # noqa: E402

DEFAULT_INPUT = (
    ROOT / "reports" / "pilot_stockout_responsibility" / "stockout_cases_classified.csv"
)
DEFAULT_OUTPUT = ROOT / "reports" / "pilot_stockout_allocation_failures"
SALE_EVENT_HEX = "D09FD180D0BED0B4D0B0D0B6D0B0"


def enrich_cases(
    cases: pd.DataFrame, bakery: pd.DataFrame, sku_totals: pd.DataFrame
) -> pd.DataFrame:
    keys = ["date", "bakery_id"]
    result = cases.merge(bakery, on=keys, how="left", validate="many_to_one")
    result = result.merge(sku_totals, on=keys, how="left", validate="many_to_one")
    result["bakery_forecast_to_actual"] = result["bakery_forecast_qty"] / result[
        "bakery_actual_qty"
    ].replace(0.0, pd.NA)
    result["bakery_volume_sufficient"] = result["bakery_forecast_to_actual"].ge(0.95)
    result["observed_sku_share"] = result["daily_sold"] / result[
        "bakery_actual_qty"
    ].replace(0.0, pd.NA)
    result["forecast_sku_share"] = result["forecast_qty"] / result[
        "sku_forecast_total_qty"
    ].replace(0.0, pd.NA)
    result["allocation_share_ratio"] = result["forecast_sku_share"] / result[
        "observed_sku_share"
    ].replace(0.0, pd.NA)
    result["diagnosis"] = "mixed_or_top_level"
    result.loc[result["bakery_volume_sufficient"], "diagnosis"] = (
        "allocation_failure_likely"
    )
    result.loc[result["forecast_qty"].le(0.5), "diagnosis"] = "missing_or_near_zero_sku"
    return result


def build_sku_summary(cases: pd.DataFrame, all_rows: pd.DataFrame) -> pd.DataFrame:
    denominators = all_rows.groupby(
        ["product_id", "product_name", "category_name"], as_index=False
    ).agg(
        eligible_sku_days=("date", "size"), eligible_bakeries=("bakery_id", "nunique")
    )
    summary = (
        cases.groupby(["product_id", "product_name", "category_name"], as_index=False)
        .agg(
            underforecast_stockouts=("date", "size"),
            dates=("date", "nunique"),
            bakeries=("bakery_id", "nunique"),
            bakery_sku_pairs=("bakery_id", "size"),
            total_confirmed_shortfall=("confirmed_model_shortfall_qty", "sum"),
            median_shortfall=("confirmed_model_shortfall_qty", "median"),
            max_shortfall=("confirmed_model_shortfall_qty", "max"),
            allocation_likely_cases=(
                "bakery_volume_sufficient",
                "sum",
            ),
        )
        .merge(
            denominators, on=["product_id", "product_name", "category_name"], how="left"
        )
    )
    summary["case_rate"] = (
        summary["underforecast_stockouts"] / summary["eligible_sku_days"]
    )
    summary["systematic"] = (
        summary["underforecast_stockouts"].ge(3)
        & summary["dates"].ge(3)
        & (summary["bakeries"].ge(2) | summary["case_rate"].ge(0.10))
    )
    return summary.sort_values(
        ["underforecast_stockouts", "total_confirmed_shortfall"], ascending=False
    )


def assign_pipeline_regime(frame: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(frame["date"])
    source = frame["source_run_id"].fillna("").astype(str)
    regime = pd.Series("raw_uplift_pre_cap_haircut", index=frame.index)
    regime.loc[source.str.contains("no_sku_uplift") | dates.lt("2026-07-01")] = (
        "base_no_sku_uplift"
    )
    regime.loc[dates.ge("2026-07-15")] = "current_cap_haircut_stockout"
    return regime


def build_regime_summary(
    cases: pd.DataFrame, all_stockouts: pd.DataFrame
) -> pd.DataFrame:
    denominators = (
        all_stockouts.groupby("pipeline_regime").size().rename("accepted_stockouts")
    )
    summary = (
        cases.groupby("pipeline_regime", as_index=False)
        .agg(
            underforecast_cases=("date", "size"),
            confirmed_shortfall=("confirmed_model_shortfall_qty", "sum"),
            allocation_likely=("bakery_volume_sufficient", "sum"),
            median_bakery_ratio=("bakery_forecast_to_actual", "median"),
            median_allocation_share_ratio=("allocation_share_ratio", "median"),
        )
        .merge(denominators, on="pipeline_regime")
    )
    summary["underforecast_rate"] = (
        summary["underforecast_cases"] / summary["accepted_stockouts"]
    )
    summary["allocation_likely_share"] = (
        summary["allocation_likely"] / summary["underforecast_cases"]
    )
    return summary


def load_bakery_context(
    client, date_from: str, date_to: str, bakery_ids: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = {"date_from": date_from, "date_to": date_to, "bakery_ids": bakery_ids}
    forecasts = client.query_df(
        """
        select forecast_date as date, bakery_id,
               argMax(forecast_final, generated_at) as bakery_forecast_qty
        from bakery_forecast_day_snapshots
        where lead_days = 1 and forecast_date between %(date_from)s and %(date_to)s
          and bakery_id in %(bakery_ids)s
        group by date, bakery_id
        """,
        parameters=params,
    )
    actual = client.query_df(
        f"""
        select check_date as date, toInt64(m.bakery_id) as bakery_id,
               sum(quantity) as bakery_actual_qty
        from mart_sales_60d as m
        where check_date between %(date_from)s and %(date_to)s
          and toInt64OrNull(m.bakery_id) in %(bakery_ids)s
          and hex(cash_event_type) = '{SALE_EVENT_HEX}'
        group by date, bakery_id
        """,
        parameters=params,
    )
    bakery = actual.merge(forecasts, on=["date", "bakery_id"], how="outer")
    sku_totals = client.query_df(
        """
        select date, bakery_id, sum(forecast_qty) as sku_forecast_total_qty
        from (
            select forecast_date as date, bakery_id, product_id,
                   argMax(forecast_qty, generated_at) as forecast_qty
            from sku_forecast_day_snapshots
            where lead_days = 1 and forecast_date between %(date_from)s and %(date_to)s
              and bakery_id in %(bakery_ids)s
            group by date, bakery_id, product_id
        )
        group by date, bakery_id
        """,
        parameters=params,
    )
    for frame in [bakery, sku_totals]:
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return bakery, sku_totals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze allocation in stockout underforecasts"
    )
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_rows = pd.read_csv(args.input, encoding="utf-8-sig")
    all_rows["date"] = pd.to_datetime(all_rows["date"]).dt.normalize()
    cases = all_rows[
        all_rows["responsibility_group"].eq("confirmed_model_underforecast")
    ].copy()
    client = create_client(args.env_file)
    bakery, sku_totals = load_bakery_context(
        client,
        str(cases["date"].min().date()),
        str(cases["date"].max().date()),
        sorted(cases["bakery_id"].unique().tolist()),
    )
    enriched = enrich_cases(cases, bakery, sku_totals)
    all_rows["pipeline_regime"] = assign_pipeline_regime(all_rows)
    enriched["pipeline_regime"] = assign_pipeline_regime(enriched)
    sku = build_sku_summary(enriched, all_rows)
    regime = build_regime_summary(
        enriched, all_rows[all_rows["stockout_group"].eq("clear_stockout")]
    )
    sku_regime = (
        enriched.groupby(
            ["product_id", "product_name", "pipeline_regime"], as_index=False
        )
        .agg(
            cases=("date", "size"),
            total_shortfall=("confirmed_model_shortfall_qty", "sum"),
            allocation_likely=("bakery_volume_sufficient", "sum"),
        )
        .sort_values(["cases", "total_shortfall"], ascending=False)
    )
    pair = (
        enriched.groupby(
            ["bakery_id", "bakery_name", "product_id", "product_name"], as_index=False
        )
        .agg(
            cases=("date", "size"),
            dates=("date", "nunique"),
            total_shortfall=("confirmed_model_shortfall_qty", "sum"),
            allocation_likely_cases=("bakery_volume_sufficient", "sum"),
        )
        .sort_values(["cases", "total_shortfall"], ascending=False)
    )
    valid = enriched["bakery_forecast_to_actual"].notna()
    allocation = enriched["bakery_volume_sufficient"] & valid
    payload = {
        "cases": int(len(enriched)),
        "bakery_context_coverage": float(valid.mean()),
        "allocation_failure_likely_cases": int(allocation.sum()),
        "allocation_failure_likely_share": float(allocation.mean()),
        "top_level_underforecast_cases": int(
            (valid & ~enriched["bakery_volume_sufficient"]).sum()
        ),
        "missing_or_near_zero_sku_cases": int(enriched["forecast_qty"].le(0.5).sum()),
        "systematic_skus": int(sku["systematic"].sum()),
        "single_case_skus": int(sku["underforecast_stockouts"].eq(1).sum()),
        "median_bakery_forecast_to_actual": float(
            enriched.loc[valid, "bakery_forecast_to_actual"].median()
        ),
        "median_allocation_share_ratio": float(
            enriched["allocation_share_ratio"].median()
        ),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output / "case_details.csv", index=False, encoding="utf-8-sig")
    sku.to_csv(output / "sku_summary.csv", index=False, encoding="utf-8-sig")
    regime.to_csv(
        output / "pipeline_regime_summary.csv", index=False, encoding="utf-8-sig"
    )
    sku_regime.to_csv(
        output / "sku_pipeline_regime.csv", index=False, encoding="utf-8-sig"
    )
    pair.to_csv(output / "bakery_sku_summary.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
