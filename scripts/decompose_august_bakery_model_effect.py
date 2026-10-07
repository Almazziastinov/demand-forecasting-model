"""Isolate the bakery-level model change on the trusted August backtest scope."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from simulate_two_day_economics import simulate_group  # noqa: E402


TRUSTED = ROOT / "reports/prod_direct_end_of_day_economics_august_folds_20260910/daily_rows.parquet"
NEW_TOTALS = ROOT / "reports/exact_bakery_p50_loss075_full_20260910/bakery_totals.parquet"
NETWORK_TOTALS = ROOT / "reports/network_trained_bakery_p50_loss075_full_20260910/bakery_totals.parquet"
OUTPUT = ROOT / "reports/august_decomposition_bakery_model_only_20260911"
DISCOUNT = 0.30


def economics(rows: pd.DataFrame, variant: str) -> dict[str, float | str]:
    rows = rows.copy()
    rows["revenue"] = rows["sold_fresh"] * rows["unit_price"] + rows[
        "sold_yesterday"
    ] * rows["unit_price"] * (1 - DISCOUNT)
    rows["production_cost"] = rows["production"] * rows["unit_cost"]
    rows["gross_profit"] = rows["revenue"] - rows["production_cost"]
    return {
        "variant": variant,
        "demand": rows["demand"].sum(),
        "production": rows["production"].sum(),
        "served": rows["served"].sum(),
        "lost": rows["lost"].sum(),
        "expired_strategy_stock": rows["expired_strategy_stock"].sum(),
        "revenue": rows["revenue"].sum(),
        "production_cost": rows["production_cost"].sum(),
        "gross_profit": rows["gross_profit"].sum(),
    }


def main() -> None:
    all_rows = pd.read_parquet(TRUSTED)
    all_rows["date"] = pd.to_datetime(all_rows["date"]).dt.normalize()
    actual = all_rows[all_rows["variant"].eq("actual_state")].copy()
    old = all_rows[all_rows["variant"].eq("p50_loss_075")].copy()
    keys = ["date", "bakery_id", "product_id"]

    # Preserve the exact trusted SKU allocation; change only its bakery total.
    old["old_bakery_total"] = old.groupby(["date", "bakery_id"])["production"].transform("sum")
    old["sku_share"] = old["production"] / old["old_bakery_total"].where(old["old_bakery_total"].gt(0), 1)
    factual = actual[keys + [
        "demand", "sold_yesterday_initial_stock", "expired_initial_stock"
    ]]

    def run_totals(path: Path) -> pd.DataFrame:
        totals = pd.read_parquet(path)
        totals["date"] = pd.to_datetime(totals["date"]).dt.normalize()
        source = old[keys + ["sku_share", "unit_price", "unit_cost"]].merge(
            totals, on=["date", "bakery_id"], how="left", validate="many_to_one"
        )
        source["new_plan"] = source["sku_share"] * source["p50_total_exact"]
        source = source.merge(factual, on=keys, validate="one_to_one")
        source["opening_stock"] = (
            source["sold_yesterday_initial_stock"] + source["expired_initial_stock"]
        )
        source["received"] = 0.0
        source["sent"] = 0.0
        source["produced"] = 0.0
        return pd.concat(
            [simulate_group(group, "new_plan") for _, group in source.groupby(["bakery_id", "product_id"], sort=False)],
            ignore_index=True,
        ).merge(source[keys + ["unit_price", "unit_cost"]], on=keys, validate="one_to_one")

    simulated = run_totals(NEW_TOTALS)
    network_simulated = run_totals(NETWORK_TOTALS)

    records = [
        economics(actual, "actual_state"),
        economics(old, "old_p50_loss075"),
        economics(simulated, "new_pilot_trained_p50_loss075"),
        economics(network_simulated, "network_trained_p50_loss075"),
    ]
    summary = pd.DataFrame(records)
    actual_gp = summary.loc[summary["variant"].eq("actual_state"), "gross_profit"].iloc[0]
    summary["gp_delta_vs_actual"] = summary["gross_profit"] - actual_gp
    summary["gp_delta_vs_actual_pct"] = 100 * summary["gp_delta_vs_actual"] / actual_gp
    summary["service_pct"] = 100 * summary["served"] / summary["demand"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT / "summary.csv", index=False, encoding="utf-8-sig")
    simulated.to_parquet(OUTPUT / "new_model_daily_rows.parquet", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
