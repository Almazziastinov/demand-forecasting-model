"""Classify confirmed stockout misses by allocation versus lost demand.

ClickHouse is read only. All outputs are local diagnostic artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.export_clickhouse_checks import create_client  # noqa: E402

DEFAULT_INPUT = ROOT / "reports/pilot_stockout_allocation_failures/case_details.csv"
DEFAULT_ALL_STOCKOUTS = (
    ROOT / "reports/pilot_stockout_responsibility/stockout_cases_classified.csv"
)
DEFAULT_OUTPUT = ROOT / "reports/stockout_mechanism_classification"
SALE_EVENT_HEX = "D09FD180D0BED0B4D0B0D0B6D0B0"
WEEKDAY_LAGS = (7, 14, 21, 28, 35, 42)


def load_sales(
    client,
    *,
    bakery_ids: list[int],
    date_from: str,
    date_to: str,
) -> pd.DataFrame:
    frame = client.query_df(
        f"""
        select
            m.check_date as sales_date,
            toInt64(m.bakery_id) as bakery_id_int,
            toInt64(m.product_id) as product_id_int,
            any(m.product_name) as product_name,
            sum(m.quantity) as sold
        from mart_sales_60d as m
        where m.check_date between toDate(%(date_from)s) and toDate(%(date_to)s)
          and toInt64OrNull(m.bakery_id) in %(bakery_ids)s
          and m.quantity > 0
          and hex(m.cash_event_type) = '{SALE_EVENT_HEX}'
        group by sales_date, bakery_id_int, product_id_int
        """,
        parameters={
            "date_from": date_from,
            "date_to": date_to,
            "bakery_ids": bakery_ids,
        },
    )
    return frame.rename(
        columns={
            "sales_date": "date",
            "bakery_id_int": "bakery_id",
            "product_id_int": "product_id",
        }
    )


def add_weekday_counterfactual(
    frame: pd.DataFrame,
    *,
    keys: list[str],
    value: str,
    lags: tuple[int, ...] = WEEKDAY_LAGS,
) -> pd.DataFrame:
    work = frame.sort_values([*keys, "date"]).copy()
    lag_columns = []
    lookup = work.set_index([*keys, "date"])[value]
    for lag in lags:
        shifted_date = work["date"] - pd.Timedelta(days=lag)
        arrays = [work[key] for key in keys] + [shifted_date]
        index = pd.MultiIndex.from_arrays(arrays, names=[*keys, "date"])
        column = f"{value}_lag_{lag}"
        work[column] = lookup.reindex(index).to_numpy()
        lag_columns.append(column)
    work[f"expected_{value}"] = work[lag_columns].median(axis=1, skipna=True)
    work[f"sigma_{value}"] = work[lag_columns].std(axis=1, ddof=1)
    work[f"reference_days_{value}"] = work[lag_columns].notna().sum(axis=1)
    return work


def build_counterfactuals(
    sales: pd.DataFrame,
    *,
    date_from: pd.Timestamp,
    date_to: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = sales.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    dates = pd.date_range(date_from, date_to, freq="D")

    bakery_ids = sorted(work["bakery_id"].unique())
    bakery_grid = pd.MultiIndex.from_product(
        [bakery_ids, dates], names=["bakery_id", "date"]
    )
    bakery = (
        work.groupby(["bakery_id", "date"])["sold"]
        .sum()
        .reindex(bakery_grid, fill_value=0.0)
        .rename("actual_bakery")
        .reset_index()
    )
    bakery = add_weekday_counterfactual(
        bakery, keys=["bakery_id"], value="actual_bakery"
    )

    pair_index = work[["bakery_id", "product_id"]].drop_duplicates()
    sku_grids = [
        pd.DataFrame(
            {
                "bakery_id": bakery_id,
                "product_id": product_id,
                "date": dates,
            }
        )
        for bakery_id, product_id in pair_index.itertuples(index=False, name=None)
    ]
    sku_grid = pd.concat(sku_grids, ignore_index=True)
    sku_actual = work.groupby(["bakery_id", "product_id", "date"], as_index=False)[
        "sold"
    ].sum()
    sku = sku_grid.merge(
        sku_actual, on=["bakery_id", "product_id", "date"], how="left"
    )
    sku["sold"] = sku["sold"].fillna(0.0)
    sku = add_weekday_counterfactual(
        sku, keys=["bakery_id", "product_id"], value="sold"
    )
    return bakery, sku


def classify_cases(
    cases: pd.DataFrame,
    bakery: pd.DataFrame,
    sku: pd.DataFrame,
    *,
    normal_ratio: float = 0.95,
    loss_ratio: float = 0.85,
    noise_z: float = 0.5,
    loss_z: float = 1.0,
    substitution_threshold: float = 0.5,
    min_reference_days: int = 3,
) -> pd.DataFrame:
    result = cases.copy()
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    bakery_context = bakery[
        [
            "date",
            "bakery_id",
            "actual_bakery",
            "expected_actual_bakery",
            "sigma_actual_bakery",
            "reference_days_actual_bakery",
        ]
    ]
    sku_context = sku[
        [
            "date",
            "bakery_id",
            "product_id",
            "sold",
            "expected_sold",
            "sigma_sold",
            "reference_days_sold",
        ]
    ].rename(columns={"sold": "actual_sku"})
    result = result.merge(
        bakery_context, on=["date", "bakery_id"], how="left", validate="many_to_one"
    )
    result = result.merge(
        sku_context,
        on=["date", "bakery_id", "product_id"],
        how="left",
        validate="many_to_one",
    )

    sigma_floor = result["expected_actual_bakery"].abs() * 0.05
    result["bakery_sigma"] = result["sigma_actual_bakery"].fillna(0.0).clip(
        lower=sigma_floor
    )
    result["bakery_gap"] = (
        result["expected_actual_bakery"] - result["actual_bakery"]
    ).clip(lower=0.0)
    result["bakery_ratio"] = result["actual_bakery"] / result[
        "expected_actual_bakery"
    ].replace(0.0, np.nan)
    result["bakery_gap_z"] = result["bakery_gap"] / result["bakery_sigma"].replace(
        0.0, np.nan
    )
    result["sku_gap_estimate"] = pd.concat(
        [
            result["confirmed_model_shortfall_qty"].clip(lower=0.0),
            (result["expected_sold"] - result["daily_sold"]).clip(lower=0.0),
        ],
        axis=1,
    ).max(axis=1)
    other_actual = result["actual_bakery"] - result["daily_sold"]
    other_expected = result["expected_actual_bakery"] - result["expected_sold"]
    result["substitution_qty"] = (other_actual - other_expected).clip(lower=0.0)
    result["substitution_ratio"] = result["substitution_qty"] / result[
        "sku_gap_estimate"
    ].replace(0.0, np.nan)
    result["substitution_ratio_capped"] = result["substitution_ratio"].clip(0.0, 1.0)

    enough_reference = (
        result["reference_days_actual_bakery"].ge(min_reference_days)
        & result["reference_days_sold"].ge(min_reference_days)
    )
    normal_volume = result["bakery_ratio"].ge(normal_ratio) | result[
        "bakery_gap_z"
    ].le(noise_z)
    clear_loss = result["bakery_ratio"].le(loss_ratio) & result["bakery_gap_z"].ge(
        loss_z
    )
    material_substitution = result["substitution_ratio_capped"].ge(
        substitution_threshold
    )

    result["case_type"] = "uncertain"
    result.loc[enough_reference & normal_volume, "case_type"] = "allocation"
    result.loc[
        enough_reference & clear_loss & ~material_substitution, "case_type"
    ] = "demand_loss"
    result.loc[
        enough_reference & clear_loss & material_substitution, "case_type"
    ] = "mixed"
    result.loc[
        enough_reference
        & ~normal_volume
        & ~clear_loss
        & material_substitution,
        "case_type",
    ] = "mixed"

    result["case_confidence"] = 0.0
    result.loc[result["case_type"].eq("allocation"), "case_confidence"] = np.maximum(
        (result["bakery_ratio"] - normal_ratio).clip(lower=0.0)
        / max(1.0 - normal_ratio, 0.01),
        (noise_z - result["bakery_gap_z"]).clip(lower=0.0) / max(noise_z, 0.01),
    )
    result.loc[result["case_type"].eq("demand_loss"), "case_confidence"] = np.minimum(
        ((loss_ratio - result["bakery_ratio"]).clip(lower=0.0) / max(loss_ratio, 0.01))
        + 0.5,
        1.0,
    )
    result.loc[result["case_type"].eq("mixed"), "case_confidence"] = 0.5
    result["case_confidence"] = result["case_confidence"].clip(0.0, 1.0)
    return result


def build_sensitivity(
    cases: pd.DataFrame, bakery: pd.DataFrame, sku: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for normal_ratio, loss_ratio, substitution_threshold in product(
        [0.90, 0.95, 1.00], [0.80, 0.85, 0.90], [0.25, 0.50, 0.75]
    ):
        if loss_ratio >= normal_ratio:
            continue
        classified = classify_cases(
            cases,
            bakery,
            sku,
            normal_ratio=normal_ratio,
            loss_ratio=loss_ratio,
            substitution_threshold=substitution_threshold,
        )
        counts = classified["case_type"].value_counts()
        rows.append(
            {
                "normal_ratio": normal_ratio,
                "loss_ratio": loss_ratio,
                "substitution_threshold": substitution_threshold,
                **{
                    name: int(counts.get(name, 0))
                    for name in ["allocation", "demand_loss", "mixed", "uncertain"]
                },
            }
        )
    return pd.DataFrame(rows)


def build_manual_review(classified: pd.DataFrame, per_class: int = 20) -> pd.DataFrame:
    samples = []
    for case_type in ["allocation", "demand_loss", "mixed", "uncertain"]:
        group = classified[classified["case_type"].eq(case_type)].copy()
        if case_type == "uncertain":
            group["review_order"] = (group["bakery_ratio"] - 0.90).abs()
            group = group.sort_values("review_order")
        else:
            group = group.sort_values(
                ["case_confidence", "confirmed_model_shortfall_qty"],
                ascending=False,
            )
        samples.append(group.head(per_class))
    return pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--all-stockouts", default=str(DEFAULT_ALL_STOCKOUTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    cases = pd.read_csv(args.input, encoding="utf-8-sig")
    cases["date"] = pd.to_datetime(cases["date"]).dt.normalize()
    all_stockouts = pd.read_csv(args.all_stockouts, encoding="utf-8-sig")
    all_stockouts["date"] = pd.to_datetime(all_stockouts["date"]).dt.normalize()
    date_from = max(
        cases["date"].min() - pd.Timedelta(days=max(WEEKDAY_LAGS)),
        pd.Timestamp("2026-05-03"),
    )
    date_to = cases["date"].max()
    client = create_client(args.env_file)
    sales = load_sales(
        client,
        bakery_ids=sorted(cases["bakery_id"].unique().tolist()),
        date_from=str(date_from.date()),
        date_to=str(date_to.date()),
    )
    bakery, sku = build_counterfactuals(
        sales, date_from=date_from, date_to=date_to
    )
    classified = classify_cases(cases, bakery, sku)
    strict = classify_cases(
        cases,
        bakery,
        sku,
        normal_ratio=1.00,
        loss_ratio=0.80,
        substitution_threshold=0.25,
    )
    classified["robust_case_type"] = "uncertain"
    classified.loc[
        strict["case_type"].eq("allocation").to_numpy(), "robust_case_type"
    ] = "allocation"
    classified.loc[
        strict["case_type"].eq("demand_loss").to_numpy(), "robust_case_type"
    ] = "demand_loss"
    classified.loc[
        strict["case_type"].eq("mixed").to_numpy(), "robust_case_type"
    ] = "mixed"
    sensitivity = build_sensitivity(cases, bakery, sku)
    manual = build_manual_review(classified)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    classified.to_csv(output / "classified_cases.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(output / "threshold_sensitivity.csv", index=False)
    manual.to_csv(output / "manual_review_sample.csv", index=False, encoding="utf-8-sig")
    summary = {
        "cases": int(len(classified)),
        "date_from": str(cases["date"].min().date()),
        "date_to": str(cases["date"].max().date()),
        "counterfactual_history_from": str(date_from.date()),
        "class_counts": classified["case_type"].value_counts().to_dict(),
        "robust_class_counts": classified["robust_case_type"].value_counts().to_dict(),
        "class_shortfall": classified.groupby("case_type")[
            "confirmed_model_shortfall_qty"
        ].sum().to_dict(),
        "median_bakery_ratio": float(classified["bakery_ratio"].median()),
        "median_substitution_ratio": float(
            classified["substitution_ratio_capped"].median()
        ),
        "clickhouse_mode": "read_only",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
