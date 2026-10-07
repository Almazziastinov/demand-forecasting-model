"""Compare Direct, fixed guards, and causal prior-year regime-aware guards."""

from __future__ import annotations

import csv
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.backtest_direct_demand_guardrail import KEYS, add_guardrails  # noqa: E402
from scripts.backtest_direct_demand_guardrail_aug24_sep09 import (  # noqa: E402
    actual_rows,
    aggregate,
    load_demand_history,
    load_rows,
    simulate_stateful,
)


OUTPUT = ROOT / "reports/regime_aware_guardrail_aug24_sep09_20260915"
RAW_HISTORY = ROOT / "data/raw/sales_stg_2025_2026.csv"
TARGET_MONTHS = (8, 9)
SEASONAL_MIN_RATIO = 1.05
CURRENT_MIN_RATIO = 1.05
MAX_SEASONAL_RATIO = 1.50
SEASONAL_DECAY_DAYS = 14


def load_prior_year_daily(rows: pd.DataFrame) -> pd.DataFrame:
    """Stream only scoped 2025 Jul-Sep sales rows through ripgrep."""
    pairs = set(zip(rows["bakery_id"].astype(int), rows["product_id"].astype(int)))
    products = sorted({product_id for _, product_id in pairs})
    product_pattern = "|".join(f"{product_id:09d}" for product_id in products)
    pattern = rf"^2025-(07|08|09)-.*,(?:{product_pattern}),"
    command = ["rg", "--no-filename", pattern, str(RAW_HISTORY)]
    process = subprocess.Popen(  # noqa: S603
        command,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1024 * 1024,
    )
    if process.stdout is None:
        raise RuntimeError("Failed to capture ripgrep output")

    daily: defaultdict[tuple[str, int, int], float] = defaultdict(float)
    for record in csv.reader(process.stdout):
        if len(record) < 13:
            continue
        try:
            date = record[1]
            bakery_id = int(record[4])
            product_id = int(record[7])
            quantity = float(record[3])
        except (TypeError, ValueError):
            continue
        if (bakery_id, product_id) in pairs:
            daily[(date, bakery_id, product_id)] += quantity
    process.stdout.close()
    return_code = process.wait()
    if return_code not in (0, 1):
        raise RuntimeError(f"ripgrep failed with exit code {return_code}")

    history = pd.DataFrame(
        [(*key, quantity) for key, quantity in daily.items()],
        columns=["date", "bakery_id", "product_id", "sales_qty"],
    )
    history["date"] = pd.to_datetime(history["date"]).dt.normalize()
    history["month"] = history["date"].dt.month
    history["is_weekend"] = history["date"].dt.dayofweek.ge(5)
    return history


def summarized_levels(history: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    return (
        history.groupby(group_columns + ["month", "is_weekend"], as_index=False)
        .agg(level=("sales_qty", "mean"), days=("date", "nunique"))
    )


def ratio_lookup(
    levels: pd.DataFrame,
    group_columns: list[str],
) -> dict[tuple[object, ...], dict[str, float]]:
    output: dict[tuple[object, ...], dict[str, float]] = {}
    for target_month in TARGET_MONTHS:
        previous_month = target_month - 1
        previous = levels[levels["month"].eq(previous_month)].rename(
            columns={"level": "previous_level", "days": "previous_days"}
        )
        target = levels[levels["month"].eq(target_month)].rename(
            columns={"level": "target_level", "days": "target_days"}
        )
        join_columns = group_columns + ["is_weekend"]
        paired = previous[join_columns + ["previous_level", "previous_days"]].merge(
            target[join_columns + ["target_level", "target_days"]],
            on=join_columns,
            how="inner",
        )
        for record in paired.itertuples(index=False):
            values = record._asdict()
            previous_level = float(values["previous_level"])
            if previous_level <= 0.0:
                continue
            key = tuple(values[column] for column in join_columns) + (target_month,)
            output[key] = {
                "ratio": float(values["target_level"]) / previous_level,
                "previous_days": float(values["previous_days"]),
                "target_days": float(values["target_days"]),
            }
    return output


def build_seasonal_priors(rows: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    pair_lookup = ratio_lookup(
        summarized_levels(history, ["bakery_id", "product_id"]),
        ["bakery_id", "product_id"],
    )
    product_lookup = ratio_lookup(
        summarized_levels(history, ["product_id"]),
        ["product_id"],
    )
    network_lookup = ratio_lookup(summarized_levels(history, []), [])

    records = []
    for row in rows[KEYS].itertuples(index=False):
        target_month = int(row.date.month)
        is_weekend = bool(row.date.dayofweek >= 5)
        pair = pair_lookup.get((row.bakery_id, row.product_id, is_weekend, target_month))
        product = product_lookup.get((row.product_id, is_weekend, target_month))
        network = network_lookup.get((is_weekend, target_month))

        fallback_ratio = 1.0
        if network is not None:
            fallback_ratio = float(network["ratio"])
        if product is not None:
            product_days = min(product["previous_days"], product["target_days"])
            product_weight = product_days / (product_days + 7.0)
            fallback_ratio = math.exp(
                product_weight * math.log(max(float(product["ratio"]), 1e-6))
                + (1.0 - product_weight) * math.log(max(fallback_ratio, 1e-6))
            )

        seasonal_ratio = fallback_ratio
        pair_days = 0.0
        if pair is not None:
            pair_days = min(pair["previous_days"], pair["target_days"])
            pair_weight = pair_days / (pair_days + 7.0)
            seasonal_ratio = math.exp(
                pair_weight * math.log(max(float(pair["ratio"]), 1e-6))
                + (1.0 - pair_weight) * math.log(max(fallback_ratio, 1e-6))
            )
        seasonal_ratio = float(np.clip(seasonal_ratio, 1.0, MAX_SEASONAL_RATIO))
        elapsed = min(max(int(row.date.day) - 1, 0), SEASONAL_DECAY_DAYS)
        seasonal_weight = 1.0 - elapsed / SEASONAL_DECAY_DAYS
        effective_ratio = 1.0 + (seasonal_ratio - 1.0) * seasonal_weight
        records.append(
            {
                "date": row.date,
                "bakery_id": row.bakery_id,
                "product_id": row.product_id,
                "prior_year_seasonal_ratio": seasonal_ratio,
                "seasonal_weight": seasonal_weight,
                "effective_seasonal_ratio": effective_ratio,
                "prior_year_pair_days": pair_days,
            }
        )
    return pd.DataFrame(records)


def build_current_growth(rows: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    pair_history = {
        key: group.sort_values("date")
        for key, group in history.groupby(["bakery_id", "product_id"], sort=False)
    }
    records = []
    for row in rows[KEYS].itertuples(index=False):
        pair = pair_history.get((row.bakery_id, row.product_id))
        current_ratio = 1.0
        comparable = False
        if pair is not None:
            age = (row.date - pair["date"]).dt.days
            same_type = pair["is_weekend"].eq(row.date.dayofweek >= 5)
            recent = pair.loc[same_type & age.between(1, 14), "demand"]
            previous = pair.loc[same_type & age.between(15, 28), "demand"]
            if len(recent) >= 2 and len(previous) >= 2 and float(previous.mean()) > 0.0:
                current_ratio = float(recent.mean()) / float(previous.mean())
                comparable = True
        records.append(
            {
                "date": row.date,
                "bakery_id": row.bakery_id,
                "product_id": row.product_id,
                "current_growth_ratio": current_ratio,
                "current_growth_comparable": comparable,
            }
        )
    return pd.DataFrame(records)


def add_regime_variants(rows: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    guards = add_guardrails(
        rows[KEYS + ["forecast_qty"]],
        history[KEYS + ["demand", "dow", "is_weekend"]],
    )
    work = rows.merge(
        guards[KEYS + ["plain_anchor_28", "plain_fixed_lower_28", "plain_fixed_upper_28"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    prior_year = load_prior_year_daily(work)
    work = work.merge(build_seasonal_priors(work, prior_year), on=KEYS, validate="one_to_one")
    work = work.merge(build_current_growth(work, history), on=KEYS, validate="one_to_one")

    direct = work["forecast_qty"]
    lower = work["plain_fixed_lower_28"].fillna(direct)
    fixed_upper = work["plain_fixed_upper_28"].fillna(direct)
    work["floor_only"] = np.maximum(direct, lower)
    work["symmetric_guard_28"] = direct.clip(lower=lower, upper=fixed_upper)

    anchor = work["plain_anchor_28"].fillna(direct)
    seasonal_anchor = anchor * work["effective_seasonal_ratio"]
    seasonal_width = np.maximum(0.15 * seasonal_anchor, 2.0)
    work["seasonal_upper"] = seasonal_anchor + seasonal_width
    work["seasonal_guard"] = work["floor_only"].clip(upper=work["seasonal_upper"])

    work["regime_confirmed"] = (
        work["prior_year_seasonal_ratio"].ge(SEASONAL_MIN_RATIO)
        & work["current_growth_comparable"]
        & work["current_growth_ratio"].ge(CURRENT_MIN_RATIO)
    )
    work["regime_upper"] = fixed_upper
    confirmed = work["regime_confirmed"]
    work.loc[confirmed, "regime_upper"] = work.loc[confirmed, "seasonal_upper"]
    work["regime_aware_guard"] = work["floor_only"].clip(upper=work["regime_upper"])
    return work


def main() -> None:
    rows = load_rows()
    history = load_demand_history(rows)
    rows = add_regime_variants(rows, history)
    variants = {
        "actual_state": actual_rows(rows),
        "direct": simulate_stateful(rows, "forecast_qty"),
        "floor_only": simulate_stateful(rows, "floor_only"),
        "symmetric_guard_28": simulate_stateful(rows, "symmetric_guard_28"),
        "seasonal_guard": simulate_stateful(rows, "seasonal_guard"),
        "regime_aware_guard": simulate_stateful(rows, "regime_aware_guard"),
    }
    source_period = rows[KEYS + ["source_period"]]
    for variant in variants:
        if variant == "actual_state":
            continue
        variants[variant] = variants[variant].merge(
            source_period, on=KEYS, how="left", validate="one_to_one"
        )

    periods = {
        "aug24_31": (pd.Timestamp("2026-08-24"), pd.Timestamp("2026-08-31")),
        "sep01_09": (pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-09")),
        "combined": (pd.Timestamp("2026-08-24"), pd.Timestamp("2026-09-09")),
    }
    summary_rows = []
    for period, (start, end) in periods.items():
        for variant, simulation in variants.items():
            summary_rows.append(aggregate(simulation[simulation["date"].between(start, end)], variant, period))
    summary = pd.DataFrame(summary_rows)
    for period, indices in summary.groupby("period").groups.items():
        block = summary.loc[indices]
        actual = block[block["variant"].eq("actual_state")].iloc[0]
        direct = block[block["variant"].eq("direct")].iloc[0]
        summary.loc[indices, "gross_profit_delta_vs_actual"] = block["gross_profit"] - actual["gross_profit"]
        summary.loc[indices, "gross_profit_delta_vs_direct"] = block["gross_profit"] - direct["gross_profit"]
        summary.loc[indices, "production_delta_vs_actual"] = block["production"] - actual["production"]
        summary.loc[indices, "lost_delta_vs_direct"] = block["lost"] - direct["lost"]
        summary.loc[indices, "writeoff_delta_vs_direct"] = block["writeoff"] - direct["writeoff"]

    daily_parts = []
    for variant, simulation in variants.items():
        daily = simulation.groupby("date", as_index=False).agg(
            production=("production", "sum"),
            lost=("lost", "sum"),
            writeoff=("writeoff", "sum"),
            gross_profit=("gross_profit", "sum"),
        )
        daily["variant"] = variant
        daily_parts.append(daily)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT / "summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(daily_parts, ignore_index=True).to_csv(
        OUTPUT / "daily.csv", index=False, encoding="utf-8-sig"
    )
    decision_columns = KEYS + [
        "source_period",
        "forecast_qty",
        "demand",
        "plain_anchor_28",
        "plain_fixed_lower_28",
        "plain_fixed_upper_28",
        "prior_year_seasonal_ratio",
        "seasonal_weight",
        "effective_seasonal_ratio",
        "current_growth_ratio",
        "current_growth_comparable",
        "regime_confirmed",
        "floor_only",
        "symmetric_guard_28",
        "seasonal_upper",
        "seasonal_guard",
        "regime_upper",
        "regime_aware_guard",
    ]
    rows[decision_columns].to_parquet(OUTPUT / "decision_rows.parquet", index=False)
    for variant, simulation in variants.items():
        simulation.to_parquet(OUTPUT / f"economics_{variant}.parquet", index=False)
    print(summary.to_string(index=False))
    print(
        {
            "rows": len(rows),
            "seasonal_adjusted_rows": int(rows["effective_seasonal_ratio"].gt(1.0 + 1e-9).sum()),
            "current_growth_comparable_rows": int(rows["current_growth_comparable"].sum()),
            "regime_confirmed_rows": int(rows["regime_confirmed"].sum()),
        }
    )


if __name__ == "__main__":
    main()
