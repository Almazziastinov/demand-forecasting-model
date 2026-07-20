"""Walk-forward SKU allocation using shares from confirmed non-stockout days."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_stockout_allocation_failures import (  # noqa: E402
    SALE_EVENT_HEX,
)
from scripts.export_clickhouse_checks import create_client  # noqa: E402

DEFAULT_INPUT = (
    ROOT / "reports" / "pilot_stockout_forecast_bias" / "sku_day_comparison.csv"
)
DEFAULT_OUTPUT = ROOT / "reports" / "non_stockout_share_allocation_experiment"


def add_bakery_actual(frame: pd.DataFrame, client) -> pd.DataFrame:
    params = {
        "date_from": str(frame["date"].min().date()),
        "date_to": str(frame["date"].max().date()),
        "bakery_ids": sorted(frame["bakery_id"].unique().tolist()),
    }
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
    actual["date"] = pd.to_datetime(actual["date"]).dt.normalize()
    return frame.merge(
        actual, on=["date", "bakery_id"], how="left", validate="many_to_one"
    )


def allocate_from_non_stockout_shares(
    frame: pd.DataFrame,
    *,
    lookback_days: int,
    min_history_days: int,
    prior_days: float,
    use_weekday: bool = True,
) -> pd.DataFrame:
    """Reallocate each evaluated bakery-day total without using future rows."""
    work = (
        frame.copy()
        .sort_values(["date", "bakery_id", "product_id"])
        .reset_index(drop=True)
    )
    baseline_total = work.groupby(["date", "bakery_id"])["forecast_qty"].transform(
        "sum"
    )
    work["baseline_share"] = work["forecast_qty"] / baseline_total.replace(0.0, np.nan)
    work["observed_bakery_share"] = work["daily_sold"] / work[
        "bakery_actual_qty"
    ].replace(0.0, np.nan)
    work["profile_share"] = np.nan
    work["profile_days"] = 0

    group_columns = ["bakery_id", "product_id"]
    if use_weekday:
        group_columns.append("dow")
    for _, indexes in work.groupby(group_columns, sort=False).groups.items():
        group = work.loc[indexes].sort_values("date")
        for index, row in group.iterrows():
            start = row["date"] - pd.Timedelta(days=lookback_days)
            history = group[
                (group["date"] < row["date"])
                & (group["date"] >= start)
                & group["stockout_group"].eq("confirmed_non_stockout")
            ]
            shares = history["observed_bakery_share"].dropna()
            work.at[index, "profile_days"] = len(shares)
            if len(shares) >= min_history_days:
                work.at[index, "profile_share"] = float(shares.median())

    alpha = work["profile_days"] / (work["profile_days"] + prior_days)
    work["allocation_weight"] = np.where(
        work["profile_share"].notna(),
        (1.0 - alpha) * work["baseline_share"] + alpha * work["profile_share"],
        work["baseline_share"],
    )
    weight_total = work.groupby(["date", "bakery_id"])["allocation_weight"].transform(
        "sum"
    )
    work["adjusted_share"] = work["allocation_weight"] / weight_total.replace(
        0.0, np.nan
    )
    work["adjusted_forecast_qty"] = baseline_total * work["adjusted_share"]
    return work


def evaluate(frame: pd.DataFrame) -> dict[str, float | int]:
    stockout = frame[frame["stockout_group"].eq("clear_stockout")]
    normal = frame[frame["stockout_group"].eq("confirmed_non_stockout")]
    before = (stockout["daily_sold"] - stockout["forecast_qty"]).clip(lower=0.0)
    after = (stockout["daily_sold"] - stockout["adjusted_forecast_qty"]).clip(lower=0.0)
    normal_sales = float(normal["daily_sold"].sum())
    original_totals = frame.groupby(["date", "bakery_id"])["forecast_qty"].sum()
    adjusted_totals = frame.groupby(["date", "bakery_id"])[
        "adjusted_forecast_qty"
    ].sum()
    return {
        "profile_coverage": float(frame["profile_share"].notna().mean()),
        "baseline_underforecast_cases": int((before > 0.5).sum()),
        "adjusted_underforecast_cases": int((after > 0.5).sum()),
        "underforecast_cases_removed": int(((before > 0.5) & (after <= 0.5)).sum()),
        "new_underforecast_cases": int(((before <= 0.5) & (after > 0.5)).sum()),
        "baseline_shortfall_qty": float(before.sum()),
        "adjusted_shortfall_qty": float(after.sum()),
        "normal_baseline_forecast_to_sales": float(normal["forecast_qty"].sum())
        / normal_sales,
        "normal_adjusted_forecast_to_sales": float(
            normal["adjusted_forecast_qty"].sum()
        )
        / normal_sales,
        "max_bakery_day_total_difference": float(
            (original_totals - adjusted_totals).abs().max()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test non-stockout SKU share allocation"
    )
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input, encoding="utf-8-sig")
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame[
        frame["stockout_group"].isin(["clear_stockout", "confirmed_non_stockout"])
    ]
    frame = add_bakery_actual(frame, create_client(args.env_file))
    rows = []
    details = {}
    # The weekday-specific grid is intentionally excluded here: it was tested
    # separately and had too little coverage. This grid evaluates the broader
    # bakery-product fallback without repeating the expensive calculation.
    for use_weekday in [False]:
        for lookback in [28, 56]:
            for minimum in [2, 3]:
                for prior in [3.0, 7.0, 14.0]:
                    mode = "pair_dow" if use_weekday else "pair_all"
                    name = f"{mode}_lb{lookback}_min{minimum}_prior{int(prior)}"
                    adjusted = allocate_from_non_stockout_shares(
                        frame,
                        lookback_days=lookback,
                        min_history_days=minimum,
                        prior_days=prior,
                        use_weekday=use_weekday,
                    )
                    result = {
                        "scenario": name,
                        "profile_mode": mode,
                        "lookback_days": lookback,
                        "min_history_days": minimum,
                        "prior_days": prior,
                    }
                    result.update(evaluate(adjusted))
                    rows.append(result)
                    details[name] = adjusted
    scenarios = pd.DataFrame(rows)
    scenarios["net_cases_improved"] = (
        scenarios["underforecast_cases_removed"] - scenarios["new_underforecast_cases"]
    )
    scenarios = scenarios.sort_values(
        ["net_cases_improved", "adjusted_shortfall_qty"], ascending=[False, True]
    )
    best = str(scenarios.iloc[0]["scenario"])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scenarios.to_csv(
        output / "scenario_comparison.csv", index=False, encoding="utf-8-sig"
    )
    details[best].to_csv(
        output / "best_scenario_rows.csv", index=False, encoding="utf-8-sig"
    )
    payload = {"best_scenario": best, "scenarios": scenarios.to_dict(orient="records")}
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
