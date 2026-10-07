"""Validate lost-demand reconstruction with synthetic sales censoring."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.calibrate_post_last_sale_demand import build_cases  # noqa: E402


HOURLY = ROOT / ".codex_tmp/rolling_hourly_sales_20260601_20260823.parquet"
PANEL = ROOT / ".codex_tmp/historical_clickhouse_panel/causal_panel.parquet"
INVENTORY = ROOT / "reports/inventory_stockout_hourly_10/hourly_frame.csv"
TOP_PRODUCTS = ROOT / "reports/top_loss_product_diagnostic_20260913/top20_product_diagnostic.csv"
OUTPUT = ROOT / "reports/synthetic_lost_demand_diagnostic_20260914"
CUTOFFS = [12, 15, 18]
FOLD_STARTS = pd.date_range("2026-06-22", "2026-08-17", freq="7D")


def score_rolling_cases(cases: pd.DataFrame) -> pd.DataFrame:
    scored = []
    for test_start in FOLD_STARTS:
        calibration_start = test_start - pd.Timedelta(days=21)
        calibration_end = test_start - pd.Timedelta(days=1)
        test_end = min(test_start + pd.Timedelta(days=6), cases["date"].max())
        calibration = cases[cases["date"].between(calibration_start, calibration_end)]
        test = cases[cases["date"].between(test_start, test_end)].copy()
        fitted = calibration.groupby("cutoff", as_index=False).agg(
            calibration_true=("true_hidden", "sum"),
            calibration_raw=("raw_prediction", "sum"),
            calibration_cases=("date", "size"),
        )
        fitted["multiplier"] = fitted["calibration_true"] / fitted["calibration_raw"]
        test = test.merge(fitted, on="cutoff", how="left", validate="many_to_one")
        test["prediction"] = test["raw_prediction"] * test["multiplier"]
        test["error"] = test["prediction"] - test["true_hidden"]
        test["abs_error"] = test["error"].abs()
        test["fold"] = test_start.date().isoformat()
        scored.append(test)
    return pd.concat(scored, ignore_index=True)


def summarize(scored: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    records = []
    grouper: str | list[str] = keys[0] if len(keys) == 1 else keys
    for values, group in scored.groupby(grouper, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        true = float(group["true_hidden"].sum())
        prediction = float(group["prediction"].sum())
        record = dict(zip(keys, values, strict=True))
        record.update(
            {
                "cases": len(group),
                "true_hidden": true,
                "prediction": prediction,
                "recovery_pct": 100 * prediction / true if true else np.nan,
                "bias_pct": 100 * (prediction - true) / true if true else np.nan,
                "wape_pct": 100 * float(group["abs_error"].sum()) / true if true else np.nan,
                "mae_units": float(group["abs_error"].mean()),
                "within_1_unit_pct": 100 * float(group["abs_error"].le(1.0).mean()),
                "within_3_units_pct": 100 * float(group["abs_error"].le(3.0).mean()),
                "underpredicted_cases_pct": 100 * float(group["error"].lt(0.0).mean()),
                "case_correlation": group["prediction"].corr(group["true_hidden"]),
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def add_dimensions(scored: pd.DataFrame) -> pd.DataFrame:
    panel = pd.read_parquet(
        PANEL, columns=["bakery_id", "bakery_name", "city", "product_id", "product_name", "category_name"]
    )
    bakeries = panel[["bakery_id", "bakery_name", "city"]].drop_duplicates("bakery_id", keep="last")
    products = panel[["product_id", "product_name", "category_name"]].drop_duplicates(
        "product_id", keep="last"
    )
    return scored.merge(bakeries, on="bakery_id", how="left", validate="many_to_one").merge(
        products, on="product_id", how="left", validate="many_to_one"
    )


def last_sale_identification_proxy() -> pd.DataFrame:
    frame = pd.read_csv(INVENTORY)
    frame["date"] = pd.to_datetime(frame["date"])
    positive = frame["sold"].gt(0.0)
    daily = frame.groupby(["date", "bakery_id", "product_id"], as_index=False).agg(
        daily_sold=("daily_sold", "first"),
        last_sale_hour=("hour", lambda hour: hour[positive.loc[hour.index]].max()),
        balance_consistent=("balance_is_consistent", "first"),
        production_observed=("is_production_observed", "first"),
        inventory_stockout=("is_inventory_stockout", "first"),
    )
    for column in ["balance_consistent", "production_observed", "inventory_stockout"]:
        daily[column] = daily[column].astype(bool)
    eligible = daily[
        daily["balance_consistent"]
        & daily["production_observed"]
        & daily["daily_sold"].gt(0.0)
        & daily["last_sale_hour"].notna()
    ].copy()
    records = []
    for cutoff in CUTOFFS:
        early = eligible["last_sale_hour"].le(cutoff)
        known_stockout = eligible["inventory_stockout"]
        records.append(
            {
                "cutoff": cutoff,
                "eligible_days": len(eligible),
                "early_last_sale_days": int(early.sum()),
                "early_known_stockout_days": int((early & known_stockout).sum()),
                "early_known_non_stockout_days": int((early & ~known_stockout).sum()),
                "last_sale_only_precision_proxy_pct": 100
                * float((early & known_stockout).sum())
                / float(early.sum())
                if early.any()
                else np.nan,
                "known_stockout_recall_pct": 100
                * float((early & known_stockout).sum())
                / float(known_stockout.sum())
                if known_stockout.any()
                else np.nan,
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    hourly = pd.read_parquet(HOURLY)
    cases = build_cases(hourly, CUTOFFS)
    scored = add_dimensions(score_rolling_cases(cases))
    scored["volume_band"] = pd.cut(
        scored["true_hidden"], bins=[-np.inf, 1, 3, 10, np.inf], labels=["<=1", "1-3", "3-10", ">10"]
    )

    summarize(scored, ["cutoff"]).to_csv(
        OUTPUT / "by_cutoff.csv", index=False, encoding="utf-8-sig"
    )
    summarize(scored, ["fold", "cutoff"]).to_csv(
        OUTPUT / "by_week.csv", index=False, encoding="utf-8-sig"
    )
    summarize(scored, ["volume_band", "cutoff"]).to_csv(
        OUTPUT / "by_volume_band.csv", index=False, encoding="utf-8-sig"
    )
    summarize(scored, ["city", "cutoff"]).to_csv(
        OUTPUT / "by_city.csv", index=False, encoding="utf-8-sig"
    )
    summarize(scored, ["bakery_id", "bakery_name", "cutoff"]).to_csv(
        OUTPUT / "by_bakery.csv", index=False, encoding="utf-8-sig"
    )
    summarize(scored, ["product_id", "product_name", "category_name", "cutoff"]).to_csv(
        OUTPUT / "by_product.csv", index=False, encoding="utf-8-sig"
    )

    top_ids = set(pd.read_csv(TOP_PRODUCTS, encoding="utf-8-sig")["product_id"].astype(int))
    top = scored[scored["product_id"].isin(top_ids)]
    summarize(top, ["product_id", "product_name", "category_name", "cutoff"]).to_csv(
        OUTPUT / "top20_by_product.csv", index=False, encoding="utf-8-sig"
    )
    summarize(top, ["cutoff"]).to_csv(
        OUTPUT / "top20_by_cutoff.csv", index=False, encoding="utf-8-sig"
    )
    proxy = last_sale_identification_proxy()
    proxy.to_csv(OUTPUT / "last_sale_identification_proxy.csv", index=False, encoding="utf-8-sig")
    scored.to_parquet(OUTPUT / "synthetic_cases.parquet", index=False)

    print("Network by cutoff")
    print(summarize(scored, ["cutoff"]).to_string(index=False))
    print("\nTop-20 by cutoff")
    print(summarize(top, ["cutoff"]).to_string(index=False))
    print("\nLast-sale identification proxy")
    print(proxy.to_string(index=False))


if __name__ == "__main__":
    main()
