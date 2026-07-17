"""Build long-history baseline and demand-adjusted profiles for ten bakeries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.backtest_pseudo_stockout_reconstruction import (  # noqa: E402
    build_synthetic_cases,
    evaluate_method,
)
from src.experiments_v2.stockout_demand_preprocessing import (  # noqa: E402
    reconstruct_stockout_demand_from_bakery_share,
)

PILOT_BAKERY_IDS = {20, 21, 22, 28, 80, 89, 107, 221, 222, 257}
DATE_FROM = "2025-06-01"
TRAIN_END = pd.Timestamp("2026-03-21")
HOLDOUT_START = pd.Timestamp("2026-03-22")
PROFILE_KEYS = ["bakery_id", "product_id", "dow", "hour"]


def _numeric_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype("int64")


def load_long_hourly(path: Path, product_ids: set[int]) -> pd.DataFrame:
    usecols = [
        "check_datetime",
        "check_date",
        "cash_event_type",
        "quantity",
        "bakery_id",
        "product_id",
    ]
    parts = []
    bakery_parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=750_000):
        chunk["bakery_id"] = _numeric_id(chunk["bakery_id"])
        chunk["product_id"] = _numeric_id(chunk["product_id"])
        chunk = chunk[
            chunk["bakery_id"].isin(PILOT_BAKERY_IDS)
            & (chunk["cash_event_type"] == "Продажа")
            & (chunk["check_date"] >= DATE_FROM)
            & (chunk["check_date"] <= str(TRAIN_END.date()))
        ].copy()
        if chunk.empty:
            continue
        chunk["date"] = pd.to_datetime(chunk["check_date"], errors="coerce")
        chunk["hour"] = (
            pd.to_datetime(chunk["check_datetime"], errors="coerce", utc=True)
            .dt.tz_convert("Europe/Moscow")
            .dt.hour
        )
        chunk["sold"] = pd.to_numeric(chunk["quantity"], errors="coerce").fillna(0.0)
        bakery_parts.append(
            chunk.groupby(["date", "bakery_id", "hour"], as_index=False)["sold"].sum()
        )
        chunk = chunk[chunk["product_id"].isin(product_ids)]
        if chunk.empty:
            continue
        parts.append(
            chunk.groupby(["date", "bakery_id", "product_id", "hour"], as_index=False)[
                "sold"
            ].sum()
        )
    hourly = (
        pd.concat(parts, ignore_index=True)
        .groupby(["date", "bakery_id", "product_id", "hour"], as_index=False)["sold"]
        .sum()
    )
    bakery_hour = (
        pd.concat(bakery_parts, ignore_index=True)
        .groupby(["date", "bakery_id", "hour"], as_index=False)["sold"]
        .sum()
        .rename(columns={"sold": "bakery_hour_sales"})
    )
    sku_days = hourly[["date", "bakery_id", "product_id"]].drop_duplicates()
    frame = sku_days.merge(pd.DataFrame({"hour": range(6, 24)}), how="cross")
    frame = frame.merge(
        hourly, on=["date", "bakery_id", "product_id", "hour"], how="left"
    ).merge(bakery_hour, on=["date", "bakery_id", "hour"], how="left")
    frame["sold"] = frame["sold"].fillna(0.0)
    frame["bakery_hour_sales"] = frame["bakery_hour_sales"].fillna(0.0)
    frame["dow"] = frame["date"].dt.dayofweek
    return frame


def build_profile(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    work = frame.copy()
    work["share"] = work[value_col] / work["bakery_hour_sales"].replace(0.0, np.nan)
    profile = work.groupby(PROFILE_KEYS, as_index=False).agg(
        profile_share=("share", "mean"), profile_days=("date", "nunique")
    )
    return profile


def evaluate_profile(profile: pd.DataFrame, holdout: pd.DataFrame) -> dict[str, float]:
    work = holdout.copy()
    work["actual_share"] = work["sold"] / work["bakery_hour_sales"].replace(0.0, np.nan)
    work = work.merge(profile, on=PROFILE_KEYS, how="left")
    valid = work[work["actual_share"].notna() & work["profile_share"].notna()]
    error = (valid["profile_share"] - valid["actual_share"]).abs()
    return {
        "share_mae": float(error.mean()),
        "weighted_share_mae": float(
            np.average(error, weights=valid["bakery_hour_sales"])
        ),
        "coverage": float(len(valid) / len(work)),
    }


def pseudo_metrics(
    march_frame: pd.DataFrame,
    profile: pd.DataFrame,
    *,
    method: str,
) -> list[dict[str, float | int | str]]:
    reference = profile.rename(columns={"profile_share": "mean_sku_share"})[
        PROFILE_KEYS + ["mean_sku_share", "profile_days"]
    ].rename(columns={"profile_days": "reference_days"})
    rows = []
    for gap in [2, 3, 4]:
        synthetic, _ = build_synthetic_cases(march_frame, gap_hours=gap)
        reconstructed = reconstruct_stockout_demand_from_bakery_share(
            synthetic, reference
        )
        cases = evaluate_method(reconstructed, method=method, gap_hours=gap)
        true_total = cases["true_hidden"].sum()
        pred_total = cases["predicted_hidden"].sum()
        rows.append(
            {
                "gap_hours": gap,
                "method": method,
                "cases": int(len(cases)),
                "recovery_ratio": float(pred_total / true_total),
                "bias_pct": float(100 * (pred_total - true_total) / true_total),
                "wape_pct": float(100 * cases["abs_error"].sum() / true_total),
            }
        )
    return rows


def main() -> None:
    march_frame = pd.read_csv(
        ROOT / "reports/inventory_stockout_hourly_10/hourly_frame.csv"
    )
    march_frame["date"] = pd.to_datetime(march_frame["date"])
    for column in [
        "balance_is_consistent",
        "is_inventory_stockout",
        "is_production_observed",
        "is_stockout_day",
    ]:
        march_frame[column] = march_frame[column].astype(bool)
    product_ids = set(march_frame["product_id"].astype(int).unique())

    history = load_long_hourly(ROOT / "data/raw/sales_stg_2025_2026.csv", product_ids)
    adjustments = pd.read_csv(
        ROOT / "reports/demand_adjusted_profile_10/adjusted_hours_audit.csv",
        usecols=["date", "bakery_id", "product_id", "hour", "imputed_demand"],
    )
    adjustments["date"] = pd.to_datetime(adjustments["date"])
    adjustments = adjustments.groupby(
        ["date", "bakery_id", "product_id", "hour"], as_index=False
    )["imputed_demand"].sum()
    history = history.merge(
        adjustments,
        on=["date", "bakery_id", "product_id", "hour"],
        how="left",
    )
    history["imputed_demand"] = history["imputed_demand"].fillna(0.0)
    history["sold_demand"] = history["sold"] + history["imputed_demand"]

    baseline = build_profile(history, "sold")
    demand = build_profile(history, "sold_demand")
    holdout = march_frame[march_frame["date"] >= HOLDOUT_START].copy()
    holdout_sum = holdout.groupby(["date", "bakery_id", "product_id"])[
        "sold"
    ].transform("sum")
    reliable = holdout[
        holdout["balance_is_consistent"]
        & ~holdout["is_inventory_stockout"]
        & ((holdout_sum - holdout["daily_sold"]).abs() <= 1)
    ].copy()

    comparison = baseline.merge(
        demand, on=PROFILE_KEYS, suffixes=("_baseline", "_demand")
    )
    comparison["share_delta"] = (
        comparison["profile_share_demand"] - comparison["profile_share_baseline"]
    )
    pseudo = pseudo_metrics(march_frame, baseline, method="long_baseline")
    pseudo += pseudo_metrics(march_frame, demand, method="long_demand")

    window_rows = []
    for window_days in [28, 42, 56, 84, 120, 293]:
        start = TRAIN_END - pd.Timedelta(days=window_days - 1)
        window_history = history[history["date"] >= start]
        for variant, value_col in [("baseline", "sold"), ("demand", "sold_demand")]:
            window_profile = build_profile(window_history, value_col)
            holdout_metrics = evaluate_profile(window_profile, reliable)
            for pseudo_row in pseudo_metrics(
                march_frame,
                window_profile,
                method=f"{window_days}d_{variant}",
            ):
                window_rows.append(
                    {
                        "window_days": window_days,
                        "variant": variant,
                        **holdout_metrics,
                        **pseudo_row,
                    }
                )

    output_dir = ROOT / "reports/long_demand_adjusted_profile_10"
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(output_dir / "baseline_profile.csv", index=False)
    demand.to_csv(output_dir / "demand_profile.csv", index=False)
    comparison.to_csv(output_dir / "profile_comparison.csv", index=False)
    pd.DataFrame(pseudo).to_csv(output_dir / "pseudo_stockout_metrics.csv", index=False)
    pd.DataFrame(window_rows).to_csv(output_dir / "window_comparison.csv", index=False)
    summary = {
        "history_date_min": str(history["date"].min().date()),
        "history_date_max": str(history["date"].max().date()),
        "history_rows": int(len(history)),
        "history_days": int(history["date"].nunique()),
        "imputed_units": float(history["imputed_demand"].sum()),
        "changed_profile_rows": int((comparison["share_delta"].abs() > 1e-12).sum()),
        "mean_abs_profile_delta": float(comparison["share_delta"].abs().mean()),
        "p99_abs_profile_delta": float(comparison["share_delta"].abs().quantile(0.99)),
        "max_abs_profile_delta": float(comparison["share_delta"].abs().max()),
        "baseline_holdout": evaluate_profile(baseline, reliable),
        "demand_holdout": evaluate_profile(demand, reliable),
        "pseudo_stockout": pseudo,
        "window_comparison": window_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
