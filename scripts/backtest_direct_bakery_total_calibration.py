"""Backtest Direct bakery-total calibration with stateful stock and batch rounding."""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "apps/forecast_embedded"))

from app.db import get_client  # noqa: E402


DETAIL = ROOT / ".codex_tmp/direct_recalibration_28d/detail.csv"
MAPPING = ROOT / "reports/markup_price_mapping_20260826/mapped_products.csv"
OUTPUT = ROOT / "reports/direct_bakery_total_calibration_20260914"
KEYS = ["date", "bakery_id", "product_id"]
EVAL_START = pd.Timestamp("2026-09-01")
EVAL_END = pd.Timestamp("2026-09-13")
DISCOUNT = 0.30

# Known-good run actually available for each day. Incident-generated runs are excluded.
RUN_BY_DATE = {
    "2026-09-01": "prod_direct_alpha_025_20260831_h14",
    "2026-09-02": "prod_direct_alpha_025_20260831_h14",
    "2026-09-03": "prod_direct_alpha_025_20260903_h14",
    "2026-09-04": "prod_direct_alpha_025_20260903_h14",
    "2026-09-05": "prod_direct_alpha_025_20260905_h14",
    "2026-09-06": "prod_direct_alpha_025_20260905_h14",
    "2026-09-07": "prod_direct_alpha_025_20260907_h14",
    "2026-09-08": "prod_direct_alpha_025_20260908_h14",
    "2026-09-09": "prod_direct_alpha_025_20260909_h14",
    "2026-09-10": "prod_direct_alpha_025_20260910_h14",
    "2026-09-11": "prod_direct_alpha_025_20260911_h14",
    "2026-09-12": "prod_direct_alpha_025_20260912_h14",
    "2026-09-13": "prod_direct_alpha_025_20260913_h14",
}


def round_up(value: float, multiple: int) -> int:
    if value <= 0:
        return 0
    return int(math.ceil(value / multiple - 1e-9) * multiple)


def load_forecasts(client) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = tuple(sorted(set(RUN_BY_DATE.values())))
    direct = client.query_df(
        """
        select forecast_date date, bakery_id, product_id, run_id, sum(forecast_qty) direct_target
        from sku_forecast_day_embedded
        where run_id in %(runs)s
          and forecast_date between toDate(%(date_from)s) and toDate(%(date_to)s)
        group by date, bakery_id, product_id, run_id
        """,
        parameters={"runs": runs, "date_from": str(EVAL_START.date()), "date_to": str(EVAL_END.date())},
    )
    direct["date"] = pd.to_datetime(direct["date"]).dt.normalize()
    direct = direct[
        direct.apply(lambda row: RUN_BY_DATE.get(str(row["date"].date())) == row["run_id"], axis=1)
    ].drop(columns="run_id")

    run_meta = client.query_df(
        "select run_id, any(notes) notes from forecast_runs_embedded where run_id in %(runs)s group by run_id",
        parameters={"runs": runs},
    )
    base_by_direct: dict[str, str] = {}
    for row in run_meta.to_dict("records"):
        match = re.search(r"from (prod_base_bakery_norm_recent_\d{8}_h14)", str(row.get("notes") or ""))
        if not match:
            raise RuntimeError(f"Cannot resolve base run for {row['run_id']}")
        base_by_direct[str(row["run_id"])] = match.group(1)
    base_runs = tuple(sorted(set(base_by_direct.values())))
    base = client.query_df(
        """
        select forecast_date date, bakery_id, run_id, sum(forecast_qty) base_total
        from sku_forecast_day_embedded
        where run_id in %(runs)s
          and forecast_date between toDate(%(date_from)s) and toDate(%(date_to)s)
        group by date, bakery_id, run_id
        """,
        parameters={"runs": base_runs, "date_from": str(EVAL_START.date()), "date_to": str(EVAL_END.date())},
    )
    base["date"] = pd.to_datetime(base["date"]).dt.normalize()
    expected_base = {
        date: base_by_direct[direct_run] for date, direct_run in RUN_BY_DATE.items()
    }
    base = base[
        base.apply(lambda row: expected_base.get(str(row["date"].date())) == row["run_id"], axis=1)
    ].drop(columns="run_id")
    return direct, base


def load_multiples(client, product_ids: list[int]) -> tuple[dict[int, int], dict[tuple[int, int], int]]:
    meta = client.query_df(
        """
        select product_id, bakery_id, kratnost, scope
        from baking_sku_meta final
        where is_active = 1 and product_id in %(product_ids)s
        """,
        parameters={"product_ids": [f"{product_id:09d}" for product_id in product_ids]},
    )
    base: dict[int, int] = {}
    bakery: dict[tuple[int, int], int] = {}
    for row in meta.to_dict("records"):
        product_id = int(row["product_id"])
        multiple = max(int(row.get("kratnost") or 1), 1)
        if row.get("scope") == "bakery" and pd.notna(row.get("bakery_id")):
            bakery[(int(row["bakery_id"]), product_id)] = multiple
        else:
            base[product_id] = multiple
    return base, bakery


def add_causal_calibration(rows: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    result = rows.copy()
    bakery_day_demand = history.groupby(["date", "bakery_id"], as_index=False)["demand"].sum()
    lag7 = bakery_day_demand.rename(columns={"date": "lag_date", "demand": "demand_lag7"})
    lag14 = bakery_day_demand.rename(columns={"date": "lag_date", "demand": "demand_lag14"})
    result["lag_date"] = result["date"] - pd.Timedelta(days=7)
    result = result.merge(lag7, on=["lag_date", "bakery_id"], how="left").drop(columns="lag_date")
    result["lag_date"] = result["date"] - pd.Timedelta(days=14)
    result = result.merge(lag14, on=["lag_date", "bakery_id"], how="left").drop(columns="lag_date")
    result["same_dow_demand"] = result[["demand_lag7", "demand_lag14"]].mean(axis=1, skipna=True)
    direct_total = result.groupby(["date", "bakery_id"])["direct_target"].transform("sum")
    base_total = result.groupby(["date", "bakery_id"])["volume_neutral_target"].transform("sum")
    causal_factor = (result["same_dow_demand"] / direct_total).clip(lower=0.85, upper=1.05).fillna(1.0)
    result["calibration_factor"] = causal_factor
    result["calibrated_target"] = result["direct_target"] * causal_factor
    raw_factor = (result["same_dow_demand"] / direct_total).fillna(1.0)
    result["gentle25_factor"] = (1.0 + 0.25 * (raw_factor - 1.0)).clip(lower=0.95, upper=1.02)
    result["gentle25_target"] = result["direct_target"] * result["gentle25_factor"]
    result["moderate50_factor"] = (1.0 + 0.50 * (raw_factor - 1.0)).clip(lower=0.90, upper=1.03)
    result["moderate50_target"] = result["direct_target"] * result["moderate50_factor"]
    # Historical SKU demand is used only by the economic last-batch gate.
    sku_history = history[KEYS + ["demand", "sold_qty"]].copy()
    sku_history["historical_lost"] = (sku_history["demand"] - sku_history["sold_qty"]).clip(lower=0.0)
    for days, column in [(7, "sku_demand_lag7"), (14, "sku_demand_lag14")]:
        lost_column = f"sku_lost_lag{days}"
        lag = sku_history.rename(
            columns={"date": "lag_date", "demand": column, "historical_lost": lost_column}
        )[["lag_date", "bakery_id", "product_id", column, lost_column]]
        result["lag_date"] = result["date"] - pd.Timedelta(days=days)
        result = result.merge(lag, on=["lag_date", "bakery_id", "product_id"], how="left").drop(columns="lag_date")
    history_points = result[["sku_demand_lag7", "sku_demand_lag14"]].notna().sum(axis=1)
    strict_avg2_demand = result[["sku_demand_lag7", "sku_demand_lag14"]].fillna(0.0).mean(axis=1)
    observed_avg2_demand = result[["sku_demand_lag7", "sku_demand_lag14"]].mean(axis=1, skipna=True)
    result["avg2_demand_strict_target"] = strict_avg2_demand
    result["avg2_demand_cold_start_target"] = observed_avg2_demand.where(
        history_points.gt(0), result["direct_target"]
    )
    result["avg2_demand_blend25_target"] = (
        0.75 * result["direct_target"] + 0.25 * result["avg2_demand_cold_start_target"]
    )
    result["avg2_demand_blend50_target"] = (
        0.50 * result["direct_target"] + 0.50 * result["avg2_demand_cold_start_target"]
    )
    result["same_dow_sku_demand"] = observed_avg2_demand.fillna(result["volume_neutral_target"])
    historical_floor = np.minimum(result["direct_target"], result["same_dow_sku_demand"])
    result["gentle25_demand_floor_target"] = np.maximum(result["gentle25_target"], historical_floor)
    prior_lost = result[["sku_lost_lag7", "sku_lost_lag14"]].fillna(0.0).max(axis=1)
    result["gentle25_loss_protected_target"] = result["gentle25_demand_floor_target"].where(
        prior_lost.le(0.5), result["direct_target"]
    )
    # Keep this invariant visible in the output.
    result["base_total"] = base_total
    return result


def simulate_group(group: pd.DataFrame, target_column: str | None, *, batch_gate: bool = False) -> pd.DataFrame:
    group = group.sort_values("date")
    carry = 0.0
    previous_date = None
    rows = []
    for record in group.itertuples(index=False):
        if previous_date is None or record.date != previous_date + pd.Timedelta(days=1):
            carry = max(float(record.opening_stock), 0.0)
        received = max(float(record.received), 0.0)
        sent = max(float(record.sent), 0.0)
        from_old_for_transfer = min(carry, sent)
        old_after_transfer = carry - from_old_for_transfer
        remaining_transfer = sent - from_old_for_transfer
        if target_column is None:
            production = max(float(record.produced), 0.0)
        else:
            target = max(float(getattr(record, target_column)), 0.0)
            net_need = max(target + remaining_transfer - old_after_transfer - received, 0.0)
            production = float(round_up(net_need, int(record.kratnost)))
            if batch_gate:
                base_target = max(float(record.volume_neutral_target), 0.0)
                base_need = max(base_target + remaining_transfer - old_after_transfer - received, 0.0)
                base_production = float(round_up(base_need, int(record.kratnost)))
                extra = max(production - base_production, 0.0)
                base_available = old_after_transfer + received + base_production - remaining_transfer
                expected_shortfall = max(float(record.same_dow_sku_demand) - base_available, 0.0)
                expected_revenue = min(extra, expected_shortfall) * float(record.unit_price)
                extra_cost = extra * float(record.unit_cost)
                if extra > 0 and expected_revenue <= extra_cost:
                    production = base_production
        fresh = max(production + received - remaining_transfer, 0.0)
        demand = max(float(record.demand), 0.0)
        sold_old = min(old_after_transfer, demand)
        sold_fresh = min(fresh, demand - sold_old)
        served = sold_old + sold_fresh
        expired_old = old_after_transfer - sold_old
        carry = fresh - sold_fresh
        rows.append(
            {
                "date": record.date,
                "bakery_id": record.bakery_id,
                "product_id": record.product_id,
                "demand": demand,
                "production": production,
                "served": served,
                "lost": demand - served,
                "lost_case": demand - served > 1e-9,
                "expired_old": expired_old,
                "ending_carry": carry,
                "revenue": sold_fresh * record.unit_price + sold_old * record.unit_price * (1 - DISCOUNT),
                "production_cost": production * record.unit_cost,
            }
        )
        previous_date = record.date
    return pd.DataFrame(rows)


def summarize(label: str, simulation: pd.DataFrame) -> dict[str, float | str]:
    return {
        "variant": label,
        "production": float(simulation["production"].sum()),
        "demand": float(simulation["demand"].sum()),
        "served": float(simulation["served"].sum()),
        "lost": float(simulation["lost"].sum()),
        "lost_cases": int(simulation["lost_case"].sum()),
        "expired_old": float(simulation["expired_old"].sum()),
        "terminal_carry": float(simulation.groupby(["bakery_id", "product_id"])["ending_carry"].last().sum()),
        "revenue": float(simulation["revenue"].sum()),
        "production_cost": float(simulation["production_cost"].sum()),
        "gross_profit": float((simulation["revenue"] - simulation["production_cost"]).sum()),
    }


def main() -> None:
    raw = pd.read_csv(DETAIL, encoding="utf-8-sig", low_memory=False)
    raw["date"] = pd.to_datetime(raw["business_date"]).dt.normalize()
    raw = raw.drop_duplicates(KEYS)
    numeric = [
        "produced_qty", "sold_qty", "received_qty", "sent_qty", "available_to_sell_qty",
        "demand_qty", "price",
    ]
    for column in numeric:
        raw[column] = pd.to_numeric(raw[column], errors="coerce").fillna(0.0)
    raw["demand"] = raw["demand_qty"].clip(lower=raw["sold_qty"])
    raw["opening_stock"] = (
        raw["available_to_sell_qty"] - raw["produced_qty"] - raw["received_qty"] + raw["sent_qty"]
    ).clip(lower=0.0)
    raw["produced"] = raw["produced_qty"]
    raw["received"] = raw["received_qty"]
    raw["sent"] = raw["sent_qty"]

    client = get_client()
    direct, base = load_forecasts(client)
    eval_facts = raw[raw["date"].between(EVAL_START, EVAL_END)].copy()
    rows = direct.merge(base, on=["date", "bakery_id"], how="inner", validate="many_to_one")
    rows = rows.merge(
        eval_facts[KEYS + ["demand", "opening_stock", "produced", "received", "sent", "price"]],
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    for column in ["demand", "opening_stock", "produced", "received", "sent"]:
        rows[column] = rows[column].fillna(0.0)
    direct_total = rows.groupby(["date", "bakery_id"])["direct_target"].transform("sum")
    rows["volume_neutral_target"] = rows["direct_target"] * rows["base_total"] / direct_total.replace(0.0, np.nan)
    rows = add_causal_calibration(rows, raw)

    mapping = pd.read_csv(MAPPING, encoding="utf-8-sig")
    mapping = mapping[mapping["valid_economics"].fillna(False).astype(bool)].copy()
    mapping["product_id"] = mapping["product_id"].astype(int)
    mapping = mapping.sort_values("unit_price").drop_duplicates("product_id", keep="last")
    rows = rows.merge(mapping[["product_id", "unit_price", "unit_cost"]], on="product_id", how="inner", validate="many_to_one")
    rows["unit_price"] = rows["price"].where(rows["price"].gt(0)).fillna(rows["unit_price"])
    base_multiple, bakery_multiple = load_multiples(client, sorted(rows["product_id"].astype(int).unique()))
    rows["kratnost"] = [
        bakery_multiple.get((int(bakery_id), int(product_id)), base_multiple.get(int(product_id), 1))
        for bakery_id, product_id in zip(rows["bakery_id"], rows["product_id"])
    ]

    variants = {
        "actual_state": (None, False),
        "direct": ("direct_target", False),
        "avg2_demand_strict": ("avg2_demand_strict_target", False),
        "avg2_demand_cold_start": ("avg2_demand_cold_start_target", False),
        "avg2_demand_blend25": ("avg2_demand_blend25_target", False),
        "avg2_demand_blend50": ("avg2_demand_blend50_target", False),
        "volume_neutral": ("volume_neutral_target", False),
        "bakery_total_gentle25": ("gentle25_target", False),
        "gentle25_demand_floor": ("gentle25_demand_floor_target", False),
        "gentle25_loss_protected": ("gentle25_loss_protected_target", False),
        "bakery_total_moderate50": ("moderate50_target", False),
        "bakery_total_calibrated": ("calibrated_target", False),
        "calibrated_batch_gate": ("calibrated_target", True),
    }
    simulations = []
    summaries = []
    for label, (column, gate) in variants.items():
        simulation = pd.concat(
            [simulate_group(group, column, batch_gate=gate) for _, group in rows.groupby(["bakery_id", "product_id"], sort=False)],
            ignore_index=True,
        )
        simulation["variant"] = label
        simulations.append(simulation)
        summaries.append(summarize(label, simulation))
    summary = pd.DataFrame(summaries)
    direct_gp = float(summary.loc[summary["variant"].eq("direct"), "gross_profit"].iloc[0])
    direct_lost = float(summary.loc[summary["variant"].eq("direct"), "lost"].iloc[0])
    summary["gross_profit_delta_vs_direct"] = summary["gross_profit"] - direct_gp
    summary["lost_delta_vs_direct"] = summary["lost"] - direct_lost
    summary["service_level_pct"] = 100 * summary["served"] / summary["demand"]
    calibration = rows.groupby(["date", "bakery_id"], as_index=False).agg(
        direct_total=("direct_target", "sum"), base_total=("volume_neutral_target", "sum"),
        calibrated_total=("calibrated_target", "sum"), factor=("calibration_factor", "first"),
        same_dow_demand=("same_dow_demand", "first"),
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT / "summary.csv", index=False, encoding="utf-8-sig")
    calibration.to_csv(OUTPUT / "bakery_day_calibration.csv", index=False, encoding="utf-8-sig")
    pd.concat(simulations, ignore_index=True).to_parquet(OUTPUT / "rows.parquet", index=False)
    print(summary.to_string(index=False))
    print("\nCalibration factors")
    print(calibration["factor"].describe().to_string())


if __name__ == "__main__":
    main()
