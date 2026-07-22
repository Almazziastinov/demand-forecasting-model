"""Replay the assortment coverage guard on historical stockout cases.

Sales are read from the local pilot export. Assortment versions are read from
ClickHouse without mutations and selected as they were available at each
forecast run timestamp.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.export_clickhouse_checks import create_client  # noqa: E402

DEFAULT_CASES = ROOT / "reports/zero_forecast_stockout_causes/cases.csv"
DEFAULT_SALES = ROOT / "data/raw/pilot_stg_check_lines_2026-04-30_2026-07-19.csv"
DEFAULT_OUTPUT = ROOT / "reports/assortment_coverage_guard_backtest"


def select_allowed_products_asof(
    assortment: pd.DataFrame,
    *,
    city: str,
    forecast_date: pd.Timestamp,
    run_generated_at: pd.Timestamp,
) -> tuple[set[int], pd.Timestamp | None]:
    """Return the latest city batch available to a historical forecast run."""
    available = assortment[
        assortment["city"].eq(city)
        & assortment["valid_from"].le(forecast_date)
        & assortment["loaded_at"].le(run_generated_at)
        & (
            assortment["valid_to"].isna()
            | assortment["valid_to"].ge(forecast_date)
        )
    ]
    if available.empty:
        return set(), None
    batch_date = available["valid_from"].max()
    products = set(
        available.loc[available["valid_from"].eq(batch_date), "product_id"]
        .dropna()
        .astype(int)
    )
    return products, batch_date


def aggregate_daily_sales(path: Path, bakery_ids: set[int]) -> pd.DataFrame:
    sales = pd.read_csv(
        path,
        usecols=[
            "check_date",
            "bakery_id",
            "product_id",
            "product_name",
            "category_name",
            "quantity",
        ],
    )
    sales["check_date"] = pd.to_datetime(sales["check_date"]).dt.normalize()
    sales["bakery_id"] = pd.to_numeric(sales["bakery_id"], errors="coerce")
    sales["product_id"] = pd.to_numeric(sales["product_id"], errors="coerce")
    sales["quantity"] = pd.to_numeric(sales["quantity"], errors="coerce")
    sales = sales[
        sales["bakery_id"].isin(bakery_ids)
        & sales["bakery_id"].notna()
        & sales["product_id"].notna()
        & sales["quantity"].gt(0)
    ].copy()
    sales[["bakery_id", "product_id"]] = sales[
        ["bakery_id", "product_id"]
    ].astype(int)
    return (
        sales.groupby(
            ["check_date", "bakery_id", "product_id"], as_index=False
        )
        .agg(
            daily_qty=("quantity", "sum"),
            product_name=("product_name", "last"),
            category_name=("category_name", "last"),
        )
    )


def build_contexts(cases: pd.DataFrame) -> pd.DataFrame:
    contexts = cases[
        ["date", "bakery_id", "city", "source_run_id", "run_generated_at"]
    ].drop_duplicates(["date", "bakery_id"])
    duplicated = contexts.duplicated(["date", "bakery_id"], keep=False)
    if duplicated.any():
        raise ValueError("Multiple run contexts found for one bakery/date")
    return contexts.sort_values(["date", "bakery_id"]).reset_index(drop=True)


def replay_guard(
    cases: pd.DataFrame,
    daily_sales: pd.DataFrame,
    assortment: pd.DataFrame,
    *,
    recent_days: int = 7,
    min_days_sold: int = 2,
    min_qty: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    known_keys = set(
        map(
            tuple,
            cases[["date", "bakery_id", "product_id"]]
            .astype({"bakery_id": int, "product_id": int})
            .to_numpy(),
        )
    )
    contexts_output: list[dict] = []
    evaluated: list[pd.DataFrame] = []
    sales_date_max = daily_sales["check_date"].max()

    for context in build_contexts(cases).itertuples(index=False):
        recent_start = context.date - pd.Timedelta(days=recent_days)
        recent = daily_sales[
            daily_sales["bakery_id"].eq(int(context.bakery_id))
            & daily_sales["check_date"].between(
                recent_start, context.date - pd.Timedelta(days=1)
            )
        ]
        recent = (
            recent.groupby(["bakery_id", "product_id"], as_index=False)
            .agg(
                recent_qty=("daily_qty", "sum"),
                recent_days_sold=("check_date", "nunique"),
                product_name=("product_name", "last"),
                category_name=("category_name", "last"),
            )
        )
        allowed, batch_date = select_allowed_products_asof(
            assortment,
            city=str(context.city),
            forecast_date=context.date,
            run_generated_at=context.run_generated_at,
        )
        recent["date"] = context.date
        recent["city"] = context.city
        recent["source_run_id"] = context.source_run_id
        recent["run_generated_at"] = context.run_generated_at
        recent["batch_valid_from"] = batch_date
        recent["batch_present"] = batch_date is not None
        recent["assortment_present"] = recent["product_id"].isin(allowed)
        recent["guard_eligible"] = (
            recent["recent_days_sold"].ge(min_days_sold)
            & recent["recent_qty"].ge(min_qty)
        )
        recent["guard_blocks"] = (
            recent["guard_eligible"] & ~recent["assortment_present"]
        )
        recent["known_clear_stockout_case"] = [
            (context.date, int(context.bakery_id), int(product_id)) in known_keys
            for product_id in recent["product_id"]
        ]
        same_day = daily_sales[
            daily_sales["check_date"].eq(context.date)
            & daily_sales["bakery_id"].eq(int(context.bakery_id))
        ][["product_id", "daily_qty"]].rename(
            columns={"daily_qty": "forecast_date_sold"}
        )
        recent = recent.merge(same_day, on="product_id", how="left")
        recent["forecast_date_sold"] = recent["forecast_date_sold"].fillna(0.0)
        future_end = min(context.date + pd.Timedelta(days=7), sales_date_max)
        future = daily_sales[
            daily_sales["check_date"].between(
                context.date + pd.Timedelta(days=1), future_end
            )
            & daily_sales["bakery_id"].eq(int(context.bakery_id))
        ]
        future = (
            future.groupby("product_id", as_index=False)
            .agg(
                next_7d_qty=("daily_qty", "sum"),
                next_7d_days_sold=("check_date", "nunique"),
            )
        )
        recent = recent.merge(future, on="product_id", how="left")
        recent[["next_7d_qty", "next_7d_days_sold"]] = recent[
            ["next_7d_qty", "next_7d_days_sold"]
        ].fillna(0)
        recent["future_days_observed"] = max(
            0, int((future_end - context.date).days)
        )
        recent["active_same_or_next_7d"] = (
            recent["forecast_date_sold"].gt(0) | recent["next_7d_qty"].gt(0)
        )
        evaluated.append(recent)
        contexts_output.append(
            {
                "date": context.date,
                "bakery_id": int(context.bakery_id),
                "city": context.city,
                "source_run_id": context.source_run_id,
                "batch_present": batch_date is not None,
                "batch_valid_from": batch_date,
                "recent_pairs": int(len(recent)),
                "eligible_pairs": int(recent["guard_eligible"].sum()),
                "blocking_pairs": int(recent["guard_blocks"].sum()),
                "diagnostic_rare_missing_pairs": int(
                    (
                        (~recent["guard_eligible"])
                        & (~recent["assortment_present"])
                    ).sum()
                ),
            }
        )

    concat_ready = [
        frame.drop(columns="batch_valid_from")
        if frame["batch_valid_from"].isna().all()
        else frame
        for frame in evaluated
    ]
    return pd.DataFrame(contexts_output), pd.concat(concat_ready, ignore_index=True)


def build_known_case_results(
    cases: pd.DataFrame, evaluated: pd.DataFrame
) -> pd.DataFrame:
    result = cases.merge(
        evaluated[
            [
                "date",
                "bakery_id",
                "product_id",
                "recent_qty",
                "recent_days_sold",
                "batch_present",
                "batch_valid_from",
                "assortment_present",
                "guard_eligible",
                "guard_blocks",
                "forecast_date_sold",
            ]
        ],
        on=["date", "bakery_id", "product_id"],
        how="left",
        validate="many_to_one",
    )
    for col in ["recent_qty", "recent_days_sold", "forecast_date_sold"]:
        result[col] = result[col].fillna(0)
    for col in [
        "batch_present",
        "assortment_present",
        "guard_eligible",
        "guard_blocks",
    ]:
        result[col] = result[col].fillna(False).astype(bool)
    return result


def build_threshold_sensitivity(known: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for min_days, min_qty in [(1, 1.0), (2, 2.0), (3, 3.0)]:
        caught = known["recent_days_sold"].ge(min_days) & known["recent_qty"].ge(
            min_qty
        )
        rows.append(
            {
                "min_days_sold": min_days,
                "min_qty": min_qty,
                "known_cases_caught": int(caught.sum()),
                "known_cases_total": int(len(known)),
                "recall": float(caught.mean()) if len(known) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def load_assortment(client) -> pd.DataFrame:
    assortment = client.query_df(
        """
        select city, toInt64(product_id) product_id, valid_from, valid_to, loaded_at
        from assortment_city_products
        where is_active = 1
        """
    )
    assortment["valid_from"] = pd.to_datetime(
        assortment["valid_from"]
    ).dt.normalize()
    assortment["valid_to"] = pd.to_datetime(assortment["valid_to"])
    assortment["loaded_at"] = pd.to_datetime(
        assortment["loaded_at"], utc=True
    )
    return assortment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--sales", default=str(DEFAULT_SALES))
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    cases = pd.read_csv(args.cases)
    cases["date"] = pd.to_datetime(cases["date"]).dt.normalize()
    cases["run_generated_at"] = pd.to_datetime(
        cases["run_generated_at"], utc=True
    )
    cases[["bakery_id", "product_id"]] = cases[
        ["bakery_id", "product_id"]
    ].astype(int)
    daily_sales = aggregate_daily_sales(
        Path(args.sales), set(cases["bakery_id"].unique())
    )
    assortment = load_assortment(create_client(args.env_file))
    contexts, evaluated = replay_guard(cases, daily_sales, assortment)
    known = build_known_case_results(cases, evaluated)
    sensitivity = build_threshold_sensitivity(known)
    blockers = evaluated[evaluated["guard_blocks"]].copy()
    rare_missing = evaluated[
        ~evaluated["guard_eligible"] & ~evaluated["assortment_present"]
    ].copy()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    contexts.to_csv(output / "contexts.csv", index=False)
    blockers.to_csv(output / "blockers.csv", index=False)
    rare_missing.to_csv(output / "rare_missing_diagnostics.csv", index=False)
    known.to_csv(output / "known_cases.csv", index=False)
    sensitivity.to_csv(output / "threshold_sensitivity.csv", index=False)

    caught = int(known["guard_blocks"].sum())
    summary = {
        "known_cases": int(len(known)),
        "known_cases_caught": caught,
        "known_case_recall": round(caught / len(known), 6) if len(known) else 0.0,
        "known_cases_not_established_in_prior_7d": int(
            (~known["guard_eligible"]).sum()
        ),
        "contexts": int(len(contexts)),
        "contexts_without_historical_batch": int((~contexts["batch_present"]).sum()),
        "blocking_rows": int(len(blockers)),
        "blocking_rows_with_same_day_sales": int(
            blockers["forecast_date_sold"].gt(0).sum()
        ),
        "blocking_rows_with_same_or_next_7d_sales": int(
            blockers["active_same_or_next_7d"].sum()
        ),
        "blocking_rows_with_full_future_window": int(
            blockers["future_days_observed"].eq(7).sum()
        ),
        "fully_observed_blocking_rows_without_same_or_next_7d_sales": int(
            (
                blockers["future_days_observed"].eq(7)
                & ~blockers["active_same_or_next_7d"]
            ).sum()
        ),
        "blocking_rows_known_clear_stockout": int(
            blockers["known_clear_stockout_case"].sum()
        ),
        "blocking_rows_with_historical_batch": int(blockers["batch_present"].sum()),
        "blocking_unique_bakery_sku": int(
            blockers[["bakery_id", "product_id"]].drop_duplicates().shape[0]
        ),
        "rare_missing_diagnostic_rows_not_blocked": int(len(rare_missing)),
        "rare_missing_rows_with_same_day_sales": int(
            rare_missing["forecast_date_sold"].gt(0).sum()
        ),
        "note": (
            "A blocking row outside the known-case set is not labelled a false "
            "positive without independent assortment intent. Same-day sales are "
            "reported as evidence that blocking was appropriate."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
