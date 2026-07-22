"""Trace clear-stockout rows that are absent from the historical forecast grid.

ClickHouse access is read only.  Versioned assortment/bakeability tables are
evaluated as of the forecast date; the unversioned share profile is reported as
current-state evidence only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment_regime_aware_sku_allocation import choose_dominant_runs  # noqa: E402
from scripts.export_clickhouse_checks import create_client  # noqa: E402

DEFAULT_INPUT = ROOT / "reports/pilot_stockout_forecast_bias/sku_day_comparison.csv"
DEFAULT_OUTPUT = ROOT / "reports/zero_forecast_stockout_causes"


def classify_causes(cases: pd.DataFrame) -> pd.DataFrame:
    result = cases.copy()
    result["cause"] = "forecast_grid_drop"
    result.loc[~result["assortment_asof"], "cause"] = "excluded_by_assortment_asof"
    result.loc[result["assortment_asof"] & ~result["bakeable_asof"], "cause"] = (
        "excluded_by_bakeability_asof"
    )
    result.loc[
        result["assortment_asof"]
        & result["bakeable_asof"]
        & ~result["current_profile_present"],
        "cause",
    ] = "forecast_grid_drop_current_profile_missing"
    return result


def load_evidence(client, cases: pd.DataFrame) -> pd.DataFrame:
    bakery_ids = sorted(cases["bakery_id"].astype(int).unique().tolist())
    product_ids = sorted(cases["product_id"].astype(int).unique().tolist())
    date_from = str(cases["date"].min().date())
    date_to = str(cases["date"].max().date())

    bakeries = client.query_df(
        "select toInt64(bakery_id) bakery_id, city from dim_bakeries "
        "where toInt64(bakery_id) in %(bakery_ids)s",
        parameters={"bakery_ids": bakery_ids},
    ).drop_duplicates("bakery_id", keep="last")
    evidence = cases.merge(bakeries, on="bakery_id", how="left")

    assortment = client.query_df(
        """
        select city, toInt64(product_id) product_id, valid_from, valid_to
        from assortment_city_products
        where toInt64(product_id) in %(product_ids)s and is_active = 1
          and valid_from <= toDate(%(date_to)s)
          and (valid_to is null or valid_to >= toDate(%(date_from)s))
        """,
        parameters={
            "product_ids": product_ids,
            "date_from": date_from,
            "date_to": date_to,
        },
    )
    bakeable = client.query_df(
        """
        select city, toInt64(product_id) product_id, scope, bakery_id,
               valid_from, valid_to
        from bakeable_products
        where toInt64(product_id) in %(product_ids)s and is_active = 1
          and is_bakeable = 1 and valid_from <= toDate(%(date_to)s)
          and (valid_to is null or valid_to >= toDate(%(date_from)s))
        """,
        parameters={
            "product_ids": product_ids,
            "date_from": date_from,
            "date_to": date_to,
        },
    )
    profile = client.query_df(
        """
        select distinct bakery_id, product_id
        from sku_hour_share_profile_smoothed_embedded
        where bakery_id in %(bakery_ids)s and product_id in %(product_ids)s
        """,
        parameters={"bakery_ids": bakery_ids, "product_ids": product_ids},
    )
    profile_keys = set(
        map(tuple, profile[["bakery_id", "product_id"]].astype(int).values)
    )

    def active(
        frame: pd.DataFrame, row: pd.Series, include_scope: bool = False
    ) -> bool:
        subset = frame[
            frame["product_id"].eq(int(row.product_id))
            & frame["city"].eq(row.city)
            & (pd.to_datetime(frame["valid_from"]) <= row.date)
            & (
                frame["valid_to"].isna()
                | (pd.to_datetime(frame["valid_to"]) >= row.date)
            )
        ]
        if include_scope and not subset.empty:
            bakery_match = (
                subset["bakery_id"].fillna(-1).astype(int).eq(int(row.bakery_id))
            )
            city_scope = subset["scope"].astype(str).str.lower().ne("bakery")
            subset = subset[bakery_match | city_scope]
        return not subset.empty

    evidence["assortment_asof"] = evidence.apply(
        lambda row: active(assortment, row), axis=1
    )
    evidence["bakeable_asof"] = evidence.apply(
        lambda row: active(bakeable, row, include_scope=True), axis=1
    )
    evidence["current_profile_present"] = [
        (int(row.bakery_id), int(row.product_id)) in profile_keys
        for row in evidence.itertuples()
    ]
    return classify_causes(evidence)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    labels = pd.read_csv(args.input)
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    labels["has_forecast"] = labels["has_forecast"].fillna(False).astype(bool)
    cases = labels[
        labels["stockout_group"].eq("clear_stockout") & ~labels["has_forecast"]
    ].copy()
    choices = choose_dominant_runs(labels)
    cases = cases.drop(columns=["source_run_id"], errors="ignore").merge(
        choices[["date", "bakery_id", "source_run_id"]],
        on=["date", "bakery_id"],
        how="left",
    )
    result = load_evidence(create_client(args.env_file), cases)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "cases.csv", index=False)
    summary = {
        "cases": int(len(result)),
        "date_from": str(result["date"].min().date()),
        "date_to": str(result["date"].max().date()),
        "unique_bakeries": int(result["bakery_id"].nunique()),
        "unique_skus": int(result["product_id"].nunique()),
        "causes": {str(k): int(v) for k, v in result["cause"].value_counts().items()},
        "evidence_note": (
            "assortment_asof and bakeable_asof use validity intervals; "
            "current_profile_present is not historical because the profile "
            "table is unversioned"
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
