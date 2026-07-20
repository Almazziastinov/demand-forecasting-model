"""Evaluate stockout reconstruction by hiding known non-stockout sales."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments_v2.stockout_demand_preprocessing import (  # noqa: E402
    build_bakery_share_reference,
    build_uncensored_hour_reference,
    reconstruct_stockout_demand,
    reconstruct_stockout_demand_from_bakery_share,
)

TRAIN_END = pd.Timestamp("2026-03-21")
HOLDOUT_START = pd.Timestamp("2026-03-22")


def prepare_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    bool_cols = [
        "balance_is_consistent",
        "is_inventory_stockout",
        "is_production_observed",
        "is_stockout_day",
    ]
    for column in bool_cols:
        frame[column] = frame[column].astype(bool)
    return frame


def build_synthetic_cases(
    frame: pd.DataFrame, *, gap_hours: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdout = frame[frame["date"] >= HOLDOUT_START].copy()
    daily = holdout.groupby(["date", "bakery_id", "product_id"], as_index=False).agg(
        daily_sold_hourly=("sold", "sum"),
        daily_sold_balance=("daily_sold", "first"),
        balance_consistent=("balance_is_consistent", "first"),
        inventory_stockout=("is_inventory_stockout", "first"),
        last_sale_hour=("hour", lambda h: h[holdout.loc[h.index, "sold"] > 0].max()),
    )
    daily["sales_agree"] = (
        daily["daily_sold_hourly"] - daily["daily_sold_balance"]
    ).abs() <= 1.0
    daily["cutoff_hour"] = daily["last_sale_hour"] - gap_hours
    candidates = daily[
        daily["balance_consistent"]
        & ~daily["inventory_stockout"]
        & daily["sales_agree"]
        & daily["last_sale_hour"].notna()
        & (daily["cutoff_hour"] >= 6)
    ].copy()
    synthetic = holdout.merge(
        candidates[
            ["date", "bakery_id", "product_id", "cutoff_hour", "daily_sold_hourly"]
        ],
        on=["date", "bakery_id", "product_id"],
        how="inner",
    )
    synthetic["true_sold"] = synthetic["sold"]
    synthetic["is_hidden_hour"] = synthetic["hour"] > synthetic["cutoff_hour"]
    synthetic["sold"] = np.where(synthetic["is_hidden_hour"], 0.0, synthetic["sold"])
    synthetic["is_stockout_day"] = True
    synthetic["is_production_observed"] = True
    return synthetic, candidates


def evaluate_method(
    reconstructed: pd.DataFrame,
    *,
    method: str,
    gap_hours: int,
) -> pd.DataFrame:
    hidden = reconstructed[reconstructed["is_hidden_hour"]].copy()
    cases = hidden.groupby(["date", "bakery_id", "product_id"], as_index=False).agg(
        true_hidden=("true_sold", "sum"),
        predicted_hidden=("sold_demand", "sum"),
        daily_sold=("daily_sold_hourly", "first"),
        predicted_hours=("is_censored_hour", "sum"),
    )
    cases["error"] = cases["predicted_hidden"] - cases["true_hidden"]
    cases["abs_error"] = cases["error"].abs()
    cases["method"] = method
    cases["gap_hours"] = gap_hours
    cases["volume_band"] = np.where(cases["daily_sold"] <= 10, "<=10", ">10")
    return cases


def build_guarded_hybrid(
    good: pd.DataFrame,
    share: pd.DataFrame,
    *,
    max_case_uplift_ratio: float = 0.75,
    max_case_uplift_units: float = 20.0,
) -> pd.DataFrame:
    """Combine both estimates and cap total imputation using observed demand."""
    hybrid = share.copy()
    hybrid["sold_demand"] = np.maximum(good["sold_demand"], share["sold_demand"])
    hybrid["imputed_demand"] = hybrid["sold_demand"] - hybrid["sold"]
    keys = ["date", "bakery_id", "product_id"]
    observed = hybrid.groupby(keys)["sold"].transform("sum")
    imputed = hybrid.groupby(keys)["imputed_demand"].transform("sum")
    case_cap = np.minimum(
        max_case_uplift_units,
        np.maximum(observed, 4.0) * max_case_uplift_ratio,
    )
    scale = np.minimum(1.0, case_cap / imputed.replace(0.0, np.nan)).fillna(0.0)
    hybrid["sold_demand"] = hybrid["sold"] + hybrid["imputed_demand"] * scale
    hybrid["imputed_demand"] = hybrid["sold_demand"] - hybrid["sold"]
    return hybrid


def summarize(cases: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in cases.groupby(["gap_hours", "method", "volume_band"]):
        gap, method, band = keys
        true_total = group["true_hidden"].sum()
        predicted_total = group["predicted_hidden"].sum()
        rows.append(
            {
                "gap_hours": int(gap),
                "method": method,
                "volume_band": band,
                "cases": int(len(group)),
                "true_hidden_units": float(true_total),
                "predicted_units": float(predicted_total),
                "recovery_ratio": float(predicted_total / true_total),
                "bias_pct": float(100 * (predicted_total - true_total) / true_total),
                "wape_pct": float(100 * group["abs_error"].sum() / true_total),
                "mae": float(group["abs_error"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["gap_hours", "volume_band", "method"])


def main() -> None:
    frame = prepare_frame(
        ROOT / "reports/inventory_stockout_hourly_10/hourly_frame.csv"
    )
    train = frame[(frame["date"] <= TRAIN_END) & frame["balance_is_consistent"]].copy()
    good_reference = build_uncensored_hour_reference(train)
    share_reference = build_bakery_share_reference(train)

    results = []
    for gap_hours in [2, 3, 4]:
        synthetic, _ = build_synthetic_cases(frame, gap_hours=gap_hours)
        good = reconstruct_stockout_demand(synthetic, good_reference)
        share = reconstruct_stockout_demand_from_bakery_share(
            synthetic, share_reference
        )
        hybrid = build_guarded_hybrid(good, share)
        results.append(evaluate_method(good, method="good_day", gap_hours=gap_hours))
        results.append(
            evaluate_method(share, method="bakery_share", gap_hours=gap_hours)
        )
        results.append(
            evaluate_method(hybrid, method="guarded_hybrid", gap_hours=gap_hours)
        )

    cases = pd.concat(results, ignore_index=True)
    summary = summarize(cases)
    output_dir = ROOT / "reports/pseudo_stockout_backtest_10"
    output_dir.mkdir(parents=True, exist_ok=True)
    cases.to_csv(output_dir / "cases.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    payload = {
        "train_end": str(TRAIN_END.date()),
        "holdout_start": str(HOLDOUT_START.date()),
        "rows": summary.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
