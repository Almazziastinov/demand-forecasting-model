"""Guarded two-day economics for partner-facing pilot reporting.

The calculation is deliberately separate from forecast-quality metrics.  It
compares two production scenarios against the same estimated demand:

* actual production;
* the production quantity from the plan that was published to the bakery.

Products live for two selling days.  Yesterday's units are sold first at a
discount; fresh units are sold at full price.  The result is an estimate of
gross profit, not a causal claim about guaranteed revenue.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class EconomicConfig:
    yesterday_discount: float = 0.30
    required_run_prefix: str = "prod_direct_alpha_025_"

    def __post_init__(self) -> None:
        if not 0 <= self.yesterday_discount <= 1:
            raise ValueError("yesterday_discount must be between 0 and 1")


REQUIRED_COLUMNS = {
    "business_date",
    "bakery_id",
    "product_id",
    "demand_qty",
    "produced_qty",
    "forecast_qty",
    "unit_price",
    "unit_cost",
}

SIMULATION_COLUMNS = [
    "business_date",
    "bakery_id",
    "product_id",
    "product_name",
    "category_name",
    "scenario",
    "demand_qty",
    "production_qty",
    "served_qty",
    "underproduction_qty",
    "ending_carry_qty",
    "expired_qty",
    "discount_loss",
    "revenue",
    "production_cost",
    "gross_profit",
]


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).clip(lower=0.0)


def eligible_economic_rows(
    detail: pd.DataFrame,
    config: EconomicConfig = EconomicConfig(),
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Return comparable rows and transparent coverage diagnostics."""
    missing = REQUIRED_COLUMNS.difference(detail.columns)
    if missing:
        return pd.DataFrame(), {
            "total_rows": int(len(detail)),
            "eligible_rows": 0,
            "coverage_ratio": 0.0,
            "excluded_rows": int(len(detail)),
        }

    work = detail.copy()
    work["business_date"] = pd.to_datetime(work["business_date"], errors="coerce")
    for column in (
        "demand_qty",
        "produced_qty",
        "forecast_qty",
        "unit_price",
        "unit_cost",
    ):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    valid = (
        work["business_date"].notna()
        & work["demand_qty"].notna()
        & work["produced_qty"].notna()
        & work["forecast_qty"].notna()
        & work["unit_price"].gt(0)
        & work["unit_cost"].ge(0)
    )
    if "eligible_lost_demand" in work.columns:
        valid &= work["eligible_lost_demand"].fillna(False).astype(bool)
    if "forecast_run_id" in work.columns and config.required_run_prefix:
        valid &= (
            work["forecast_run_id"]
            .fillna("")
            .astype(str)
            .str.startswith(config.required_run_prefix)
        )
    eligible = work.loc[valid].copy()
    for column in ("issued_yesterday_stock", "received_qty", "sent_qty"):
        eligible[column] = _numeric(eligible, column)
    total_demand = float(_numeric(work, "demand_qty").sum())
    eligible_demand = float(_numeric(eligible, "demand_qty").sum())
    coverage = eligible_demand / total_demand if total_demand > 0 else 0.0
    return eligible, {
        "total_rows": int(len(work)),
        "eligible_rows": int(len(eligible)),
        "excluded_rows": int(len(work) - len(eligible)),
        "coverage_ratio": coverage,
    }


def _simulate(
    detail: pd.DataFrame, production_column: str, scenario: str, config: EconomicConfig
) -> pd.DataFrame:
    outputs: list[dict] = []
    ordered = detail.sort_values(["bakery_id", "product_id", "business_date"])
    for (bakery_id, product_id), group in ordered.groupby(
        ["bakery_id", "product_id"], sort=False
    ):
        carry = 0.0
        first = True
        previous_date: pd.Timestamp | None = None
        for row in group.itertuples(index=False):
            current_date = pd.Timestamp(row.business_date)
            if first:
                opening = getattr(row, "issued_yesterday_stock", 0.0)
                carry = 0.0 if pd.isna(opening) else max(float(opening), 0.0)
                first = False
            elif previous_date is not None and (current_date - previous_date).days > 1:
                carry = 0.0
            received = max(float(getattr(row, "received_qty", 0.0) or 0.0), 0.0)
            sent = max(float(getattr(row, "sent_qty", 0.0) or 0.0), 0.0)
            demand = max(float(row.demand_qty), 0.0)
            if production_column == "forecast_net_need":
                target = max(float(row.forecast_qty), 0.0)
                production = max(target + sent - carry - received, 0.0)
            else:
                production = max(float(getattr(row, production_column)), 0.0)

            old_sent = min(carry, sent)
            old_available = carry - old_sent
            fresh_available = max(production + received - (sent - old_sent), 0.0)
            sold_old = min(old_available, demand)
            sold_fresh = min(fresh_available, max(demand - sold_old, 0.0))
            served = sold_old + sold_fresh
            ending_carry = max(fresh_available - sold_fresh, 0.0)
            expired = max(old_available - sold_old, 0.0)
            revenue = sold_fresh * float(row.unit_price) + sold_old * float(
                row.unit_price
            ) * (1 - config.yesterday_discount)
            production_cost = production * float(row.unit_cost)
            outputs.append(
                {
                    "business_date": row.business_date,
                    "bakery_id": int(bakery_id),
                    "product_id": int(product_id),
                    "product_name": getattr(row, "product_name", None),
                    "category_name": getattr(row, "fact_category_name", None),
                    "scenario": scenario,
                    "demand_qty": demand,
                    "production_qty": production,
                    "served_qty": served,
                    "underproduction_qty": max(demand - served, 0.0),
                    "ending_carry_qty": ending_carry,
                    "expired_qty": expired,
                    "discount_loss": sold_old
                    * float(row.unit_price)
                    * config.yesterday_discount,
                    "revenue": revenue,
                    "production_cost": production_cost,
                    "gross_profit": revenue - production_cost,
                }
            )
            carry = ending_carry
            previous_date = current_date
    return pd.DataFrame(outputs, columns=SIMULATION_COLUMNS)


def simulate_partner_economics(
    detail: pd.DataFrame,
    config: EconomicConfig = EconomicConfig(),
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Simulate actual and published-plan economics on comparable SKU-days."""
    eligible, coverage = eligible_economic_rows(detail, config=config)
    if eligible.empty:
        return pd.DataFrame(), coverage
    actual = _simulate(eligible, "produced_qty", "actual", config)
    ai_plan = _simulate(eligible, "forecast_net_need", "ai_plan", config)
    return pd.concat([actual, ai_plan], ignore_index=True), coverage


def summarize_partner_economics(rows: pd.DataFrame) -> dict[str, float | None]:
    if rows.empty:
        return {
            "actual_gross_profit": None,
            "ai_gross_profit": None,
            "profit_delta": None,
            "profit_delta_pct": None,
            "actual_discount_loss": None,
            "actual_expired_qty": None,
            "actual_underproduction_qty": None,
        }
    totals = rows.groupby("scenario", as_index=True).agg(
        gross_profit=("gross_profit", "sum"),
        discount_loss=("discount_loss", "sum"),
        expired_qty=("expired_qty", "sum"),
        underproduction_qty=("underproduction_qty", "sum"),
    )
    if not {"actual", "ai_plan"}.issubset(totals.index):
        return summarize_partner_economics(pd.DataFrame())
    actual = float(totals.at["actual", "gross_profit"])
    ai = float(totals.at["ai_plan", "gross_profit"])
    delta = ai - actual
    return {
        "actual_gross_profit": actual,
        "ai_gross_profit": ai,
        "profit_delta": delta,
        "profit_delta_pct": delta / actual if actual > 0 else None,
        "actual_discount_loss": float(totals.at["actual", "discount_loss"]),
        "actual_expired_qty": float(totals.at["actual", "expired_qty"]),
        "actual_underproduction_qty": float(totals.at["actual", "underproduction_qty"]),
    }


def rank_economic_actions(rows: pd.DataFrame, limit: int = 10) -> list[dict]:
    """Rank SKU opportunities by positive AI-vs-actual gross-profit delta."""
    if rows.empty:
        return []
    grouped = rows.groupby(
        ["scenario", "product_id", "product_name", "category_name"],
        dropna=False,
        as_index=False,
    ).agg(
        gross_profit=("gross_profit", "sum"),
        production_qty=("production_qty", "sum"),
        underproduction_qty=("underproduction_qty", "sum"),
        discount_loss=("discount_loss", "sum"),
        expired_qty=("expired_qty", "sum"),
    )
    pivot = grouped.pivot_table(
        index=["product_id", "product_name", "category_name"],
        columns="scenario",
        values=[
            "gross_profit",
            "production_qty",
            "underproduction_qty",
            "discount_loss",
            "expired_qty",
        ],
        aggfunc="sum",
    )
    if pivot.empty:
        return []
    pivot.columns = [f"{metric}_{scenario}" for metric, scenario in pivot.columns]
    pivot = pivot.reset_index()
    pivot["profit_delta"] = pivot.get("gross_profit_ai_plan", 0) - pivot.get(
        "gross_profit_actual", 0
    )
    pivot["production_delta"] = pivot.get("production_qty_ai_plan", 0) - pivot.get(
        "production_qty_actual", 0
    )
    pivot = (
        pivot[pivot["profit_delta"] > 0]
        .sort_values("profit_delta", ascending=False)
        .head(limit)
    )
    result = []
    for row in pivot.itertuples(index=False):
        production_delta = float(row.production_delta)
        reason = (
            "Недостаточный выпуск"
            if production_delta > 0
            else "Избыточный выпуск / остаток"
        )
        recommendation = (
            f"Следовать плану: увеличить выпуск примерно на {production_delta:.0f} шт."
            if production_delta > 0
            else (
                "Следовать плану: сократить выпуск примерно на "
                f"{abs(production_delta):.0f} шт."
            )
        )
        result.append(
            {
                "product_id": int(row.product_id),
                "product_name": row.product_name
                if pd.notna(row.product_name)
                else f"SKU {int(row.product_id)}",
                "category_name": row.category_name
                if pd.notna(row.category_name)
                else None,
                "reason": reason,
                "profit_delta": float(row.profit_delta),
                "production_delta": production_delta,
                "recommendation": recommendation,
            }
        )
    return result
