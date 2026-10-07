"""Classify the products responsible for most of Direct's August GP loss."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/three_prod_models_august_holdout_20260911"
STOCK = ROOT / "reports/direct_old_stock_credit_20260911"
MAPPING = ROOT / "reports/markup_price_mapping_20260826/mapped_products.csv"
OUTPUT = ROOT / "reports/top_loss_product_diagnostic_20260913"
KEYS = ["date", "bakery_id", "product_id"]
DIRECT_PLAN = "plan_1"
TOP_N = 20


def load_features() -> pd.DataFrame:
    columns = KEYS + [
        "recent_trend",
        "presence_28",
        "historical_volume",
        "cold_start_fallback",
        "tail_cap_applied",
        "predicted_stockout_probability",
    ]
    frames = [
        pd.read_parquet(path, columns=columns)
        for path in sorted((SOURCE / "direct_days").glob("*/shadow_rows.parquet"))
    ]
    features = pd.concat(frames, ignore_index=True)
    features["date"] = pd.to_datetime(features["date"]).dt.normalize()
    return features.drop_duplicates(KEYS, keep="last")


def load_names() -> pd.DataFrame:
    mapping = pd.read_csv(MAPPING, encoding="utf-8-sig")
    mapping = mapping[mapping["valid_economics"].astype(bool)].copy()
    mapping = mapping.sort_values("unit_price").drop_duplicates("product_id", keep="last")
    mapping["product_id"] = mapping["product_id"].astype(int)
    return mapping[["product_id", "product_name", "category_name"]]


def safe_pct(numerator: float, denominator: float) -> float:
    return 100 * numerator / denominator if denominator else np.nan


def classify(record: dict[str, float | int | str | bool]) -> str:
    flags: list[str] = []
    h1 = float(record["bias_h1_pct"])
    h2 = float(record["bias_h2_pct"])
    if h1 < -5 and h2 < -5:
        flags.append("устойчивый недопрогноз")
    elif h1 > 5 and h2 > 5:
        flags.append("устойчивый перепрогноз")
    elif h1 * h2 < 0:
        flags.append("смена направления ошибки")

    if float(record["regime_signal_rows_pct"]) >= 35:
        flags.append("нестабильный recent-trend")
    if float(record["reconstructed_lost_share_pct"]) >= 15:
        flags.append("сильная зависимость от восстановленного спроса")
    if float(record["missing_plan_demand_share_pct"]) >= 2:
        flags.append("пропуски/почти нулевой план")
    if (
        float(record["cold_start_rows_pct"]) >= 10
        or float(record["low_presence_rows_pct"]) >= 20
    ):
        flags.append("редкая позиция/холодный старт")
    if abs(float(record["no_old_credit_gp_delta"])) >= 10_000:
        flags.append("чувствительность к переходящему остатку")
    return "; ".join(flags) if flags else "тонкая SKU-калибровка"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = pd.read_parquet(SOURCE / "evaluation_input.parquet")
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    actual = pd.read_parquet(SOURCE / "economics_actual_state.parquet")
    direct = pd.read_parquet(SOURCE / "economics_direct_alpha_025_v1.parquet")
    full_credit = pd.read_parquet(STOCK / "economics_current_full_credit.parquet")
    no_credit = pd.read_parquet(STOCK / "economics_no_old_stock_credit.parquet")

    paired = rows[KEYS + ["demand", "observed_sales_qty", DIRECT_PLAN]].merge(
        actual[KEYS + ["production", "lost", "writeoff", "gross_profit"]],
        on=KEYS,
        validate="one_to_one",
    )
    paired = paired.rename(
        columns={
            "production": "actual_production",
            "lost": "actual_lost",
            "writeoff": "actual_writeoff",
            "gross_profit": "actual_gp",
        }
    )
    paired = paired.merge(
        direct[KEYS + ["production", "lost", "writeoff", "gross_profit"]],
        on=KEYS,
        validate="one_to_one",
    ).rename(
        columns={
            "production": "direct_production",
            "lost": "direct_lost",
            "writeoff": "direct_writeoff",
            "gross_profit": "direct_gp",
        }
    )
    paired = paired.merge(load_features(), on=KEYS, how="left", validate="one_to_one")

    stock = full_credit[KEYS + ["gross_profit"]].merge(
        no_credit[KEYS + ["gross_profit"]],
        on=KEYS,
        suffixes=("_full_credit", "_no_credit"),
        validate="one_to_one",
    )
    stock["no_old_credit_gp_delta"] = (
        stock["gross_profit_no_credit"] - stock["gross_profit_full_credit"]
    )
    paired = paired.merge(
        stock[KEYS + ["no_old_credit_gp_delta"]], on=KEYS, validate="one_to_one"
    )

    paired["gp_delta"] = paired["direct_gp"] - paired["actual_gp"]
    product_loss = paired.groupby("product_id")["gp_delta"].sum().sort_values()
    top_ids = product_loss.head(TOP_N).index
    top = paired[paired["product_id"].isin(top_ids)].copy()
    top["half"] = np.where(top["date"].le(pd.Timestamp("2026-08-27")), "h1", "h2")

    records = []
    for product_id, group in top.groupby("product_id"):
        demand = float(group["demand"].sum())
        reconstructed = float((group["demand"] - group["observed_sales_qty"]).clip(lower=0).sum())
        plan = float(group[DIRECT_PLAN].sum())
        h = group.groupby("half")[[DIRECT_PLAN, "demand"]].sum()
        h1_demand = float(h.loc["h1", "demand"]) if "h1" in h.index else 0.0
        h2_demand = float(h.loc["h2", "demand"]) if "h2" in h.index else 0.0
        h1_bias = float(h.loc["h1", DIRECT_PLAN] - h1_demand) if "h1" in h.index else 0.0
        h2_bias = float(h.loc["h2", DIRECT_PLAN] - h2_demand) if "h2" in h.index else 0.0
        positive_demand = group["demand"].gt(0.0)
        missing_plan = positive_demand & group[DIRECT_PLAN].le(0.05)
        actual_case = group["actual_lost"].ge(1.0)
        direct_case = group["direct_lost"].ge(1.0)
        record: dict[str, float | int | str | bool] = {
            "product_id": int(product_id),
            "gp_delta": float(group["gp_delta"].sum()),
            "demand_qty": demand,
            "observed_sales_qty": float(group["observed_sales_qty"].sum()),
            "reconstructed_lost_qty": reconstructed,
            "reconstructed_lost_share_pct": safe_pct(reconstructed, demand),
            "direct_plan_qty": plan,
            "plan_bias_qty": plan - demand,
            "plan_bias_pct": safe_pct(plan - demand, demand),
            "plan_wape_pct": safe_pct(float((group[DIRECT_PLAN] - group["demand"]).abs().sum()), demand),
            "bias_h1_pct": safe_pct(h1_bias, h1_demand),
            "bias_h2_pct": safe_pct(h2_bias, h2_demand),
            "actual_lost_qty": float(group["actual_lost"].sum()),
            "direct_lost_qty": float(group["direct_lost"].sum()),
            "lost_delta_qty": float((group["direct_lost"] - group["actual_lost"]).sum()),
            "actual_lost_cases": int(actual_case.sum()),
            "direct_lost_cases": int(direct_case.sum()),
            "fixed_cases": int((actual_case & ~direct_case).sum()),
            "new_cases": int((~actual_case & direct_case).sum()),
            "direct_writeoff_qty": float(group["direct_writeoff"].sum()),
            "writeoff_delta_qty": float((group["direct_writeoff"] - group["actual_writeoff"]).sum()),
            "missing_plan_demand_qty": float(group.loc[missing_plan, "demand"].sum()),
            "missing_plan_demand_share_pct": safe_pct(float(group.loc[missing_plan, "demand"].sum()), demand),
            "cold_start_rows_pct": 100 * float(group["cold_start_fallback"].fillna(False).mean()),
            "low_presence_rows_pct": 100 * float(group["presence_28"].fillna(0).lt(0.5).mean()),
            "regime_signal_rows_pct": 100
            * float((~group["recent_trend"].fillna(1.0).between(0.8, 1.25)).mean()),
            "median_recent_trend": float(group["recent_trend"].median()),
            "tail_cap_rows_pct": 100 * float(group["tail_cap_applied"].fillna(False).mean()),
            "no_old_credit_gp_delta": float(group["no_old_credit_gp_delta"].sum()),
        }
        record["diagnostic_flags"] = classify(record)
        records.append(record)

    result = pd.DataFrame(records).merge(load_names(), on="product_id", how="left")
    result = result.sort_values("gp_delta")
    ordered = ["product_id", "product_name", "category_name", "diagnostic_flags"]
    result = result[ordered + [column for column in result if column not in ordered]]
    result.to_csv(OUTPUT / "top20_product_diagnostic.csv", index=False, encoding="utf-8-sig")

    flag_counts = (
        result.assign(flag=result["diagnostic_flags"].str.split("; "))
        .explode("flag")
        .groupby("flag", as_index=False)
        .agg(products=("product_id", "nunique"), gp_loss=("gp_delta", lambda value: -value.sum()))
        .sort_values("gp_loss", ascending=False)
    )
    flag_counts.to_csv(OUTPUT / "flag_summary.csv", index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))
    print("\nFlag summary\n", flag_counts.to_string(index=False))


if __name__ == "__main__":
    main()
