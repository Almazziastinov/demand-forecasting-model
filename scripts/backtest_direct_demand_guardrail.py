"""Causal backtest of robust demand-based guardrails around Direct forecasts.

The experiment is read-only.  It reconstructs the accepted end-of-day demand
target, estimates a workday/weekend level and a shrunk weekday profile from
strictly prior observations, and clips Direct only when the historical anchor
has enough support.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_operational_direct_old_loss_avg2_sep import (  # noqa: E402
    KEYS,
    build_actual_state,
    simulate_variant,
)


OUTPUT = ROOT / "reports/direct_demand_guardrail_20260914"
TARGET_FILES = [
    ROOT / "reports/operational_direct_old_loss_avg2_aug24_31_20260914/rows.parquet",
    ROOT / "reports/operational_direct_old_loss_avg2_sep01_09_20260914/rows.parquet",
]
DETAIL_FILES = [
    ROOT / ".codex_tmp/pilot_recalc_report/detail.csv",
    ROOT / ".codex_tmp/pilot_management_sep01_09/detail.csv",
]
SALE_TIME_FILES = [
    ROOT / ".codex_tmp/august_sale_times.csv",
    ROOT / ".codex_tmp/sep01_09_sale_times.csv",
]
CAUSAL_PANEL = ROOT / ".codex_tmp/historical_clickhouse_panel/causal_panel.parquet"
REGINA_BAKERIES = {89, 107, 222}


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights))


def _robust_weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) >= 6:
        lower, upper = np.quantile(values, [0.10, 0.90])
        values = np.clip(values, lower, upper)
    return _weighted_mean(values, weights)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cutoff = weights.sum() / 2.0
    return float(values[np.searchsorted(np.cumsum(weights), cutoff, side="left")])


def _recency_weights(age_days: np.ndarray) -> np.ndarray:
    return np.where(age_days <= 7, 3.0, np.where(age_days <= 14, 2.0, 1.0))


def load_historical_demand() -> pd.DataFrame:
    details = []
    for path in DETAIL_FILES:
        part = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        part["date"] = pd.to_datetime(part["business_date"]).dt.normalize()
        details.append(part)
    detail = pd.concat(details, ignore_index=True).drop_duplicates(KEYS, keep="last")
    for column in ["sold_qty", "produced_qty", "received_qty", "sent_qty"]:
        detail[column] = pd.to_numeric(detail[column], errors="coerce").fillna(0.0)

    times = []
    for path in SALE_TIME_FILES:
        part = pd.read_csv(path)
        part["date"] = pd.to_datetime(part["date"]).dt.normalize()
        times.append(part)
    sale_times = pd.concat(times, ignore_index=True).drop_duplicates(KEYS, keep="last")
    detail = detail.merge(sale_times[KEYS + ["last_sale_time", "bakery_last_sale_time"]], on=KEYS, how="left")

    # July rows are needed by the first August folds.  Fill their hour fields
    # from the historical fact panel; the demand eligibility still comes from
    # the operational report and is therefore identical across the period.
    panel = pd.read_parquet(CAUSAL_PANEL, columns=KEYS + ["last_sale_hour"])
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["last_sale_hour"] = pd.to_numeric(panel["last_sale_hour"], errors="coerce")
    panel["bakery_last_sale_hour"] = panel.groupby(["date", "bakery_id"])["last_sale_hour"].transform("max")
    detail = detail.merge(
        panel[KEYS + ["last_sale_hour", "bakery_last_sale_hour"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )

    exact_last = pd.to_datetime(detail["last_sale_time"], errors="coerce", utc=True)
    exact_bakery_end = pd.to_datetime(detail["bakery_last_sale_time"], errors="coerce", utc=True)
    opening = detail["date"].dt.tz_localize("Europe/Moscow").dt.tz_convert("UTC") + pd.Timedelta(hours=7.5)
    exact_elapsed = (exact_last - opening).dt.total_seconds() / 3600.0
    exact_remaining = (exact_bakery_end - exact_last).dt.total_seconds().clip(lower=0.0) / 3600.0
    elapsed = exact_elapsed.fillna(detail["last_sale_hour"] - 7.5)
    remaining = exact_remaining.fillna((detail["bakery_last_sale_hour"] - detail["last_sale_hour"]).clip(lower=0.0))

    eligible = (
        detail["eligible_lost_demand"].fillna(False).astype(bool)
        & elapsed.ge(2.0)
        & remaining.gt(0.0)
        & detail["sold_qty"].gt(0.0)
    )
    raw_lost = detail["sold_qty"] / elapsed * remaining
    cap = np.maximum(detail["sold_qty"] * 1.5, 15.0)
    detail["restored_lost"] = raw_lost.where(eligible, 0.0).clip(lower=0.0).clip(upper=cap)
    detail["demand"] = detail["sold_qty"] + detail["restored_lost"]
    detail["dow"] = detail["date"].dt.dayofweek
    detail["is_weekend"] = detail["dow"].ge(5)
    return detail[
        KEYS + ["sold_qty", "restored_lost", "demand", "dow", "is_weekend"]
    ].sort_values(KEYS)


def load_targets() -> pd.DataFrame:
    parts = []
    for path in TARGET_FILES:
        part = pd.read_parquet(path)
        part = part[part["variant"].eq("actual_state")].copy()
        parts.append(part)
    target = pd.concat(parts, ignore_index=True).drop_duplicates(KEYS, keep="last")
    target["date"] = pd.to_datetime(target["date"]).dt.normalize()
    return target.sort_values(KEYS).reset_index(drop=True)


def estimate_anchor(
    history: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    forecast_date: pd.Timestamp,
    window_days: int,
    *,
    robust: bool = True,
) -> dict[str, float]:
    dates, demands, dows, weekends = history
    forecast_day = np.datetime64(forecast_date, "D")
    age_days = (forecast_day - dates).astype("timedelta64[D]").astype(int)
    mask = (age_days >= 1) & (age_days <= window_days)
    demands = demands[mask]
    dows = dows[mask]
    weekends = weekends[mask]
    age_days = age_days[mask]
    target_dow = int(forecast_date.dayofweek)
    target_weekend = target_dow >= 5
    typed_mask = weekends == target_weekend
    typed_values = demands[typed_mask]
    typed_dows = dows[typed_mask]
    typed_age = age_days[typed_mask]
    min_type = 4 if target_weekend else 8
    same_dow_mask = typed_dows == target_dow
    same_dow_values = typed_values[same_dow_mask]
    same_dow_age = typed_age[same_dow_mask]
    if len(typed_values) < min_type or len(same_dow_values) < 2:
        return {
            "anchor": np.nan,
            "rel_mad": np.nan,
            "type_n": len(typed_values),
            "dow_n": len(same_dow_values),
        }

    typed_weights = _recency_weights(typed_age)
    reducer = _robust_weighted_mean if robust else _weighted_mean
    level = reducer(typed_values, typed_weights)
    if level <= 0:
        return {
            "anchor": np.nan,
            "rel_mad": np.nan,
            "type_n": len(typed_values),
            "dow_n": len(same_dow_values),
        }

    dow_weights = _recency_weights(same_dow_age)
    dow_mean = reducer(same_dow_values, dow_weights)
    shrink = len(same_dow_values) / (len(same_dow_values) + 4.0)
    weekday_factor = np.clip(1.0 + shrink * (dow_mean / level - 1.0), 0.60, 1.40)
    anchor = level * weekday_factor

    expected = np.empty(len(typed_values), dtype=float)
    for dow in np.unique(typed_dows):
        dow_mask = typed_dows == dow
        values = typed_values[dow_mask]
        weights = typed_weights[dow_mask]
        local_mean = reducer(values, weights)
        local_shrink = len(values) / (len(values) + 4.0)
        factor = np.clip(1.0 + local_shrink * (local_mean / level - 1.0), 0.60, 1.40)
        expected[dow_mask] = level * factor
    ratios = typed_values / np.maximum(expected, 1e-9)
    ratio_median = _weighted_median(ratios, typed_weights)
    rel_mad = 1.4826 * _weighted_median(np.abs(ratios - ratio_median), typed_weights)
    return {
        "anchor": anchor,
        "rel_mad": rel_mad,
        "type_n": len(typed_values),
        "dow_n": len(same_dow_values),
    }


def add_guardrails(target: pd.DataFrame, demand: pd.DataFrame) -> pd.DataFrame:
    histories = {}
    for key, group in demand.groupby(["bakery_id", "product_id"], sort=False):
        group = group.sort_values("date")
        histories[key] = (
            group["date"].to_numpy(dtype="datetime64[D]"),
            group["demand"].to_numpy(dtype=float),
            group["dow"].to_numpy(dtype=int),
            group["is_weekend"].to_numpy(dtype=bool),
        )
    records = []
    for row in target[["date", "bakery_id", "product_id", "forecast_qty"]].itertuples(index=False):
        history = histories.get((row.bakery_id, row.product_id))
        record = {"date": row.date, "bakery_id": row.bakery_id, "product_id": row.product_id}
        for window in [28, 35]:
            estimate = (
                estimate_anchor(history, row.date, window)
                if history is not None
                else {"anchor": np.nan, "rel_mad": np.nan, "type_n": 0, "dow_n": 0}
            )
            anchor = estimate["anchor"]
            fixed_width = max(0.15 * anchor, 2.0) if pd.notna(anchor) else np.nan
            relative_width = np.clip(1.5 * estimate["rel_mad"], 0.15, 0.50)
            dynamic_width = max(relative_width * anchor, 2.0) if pd.notna(anchor) else np.nan
            record.update(
                {
                    f"anchor_{window}": anchor,
                    f"relative_mad_{window}": estimate["rel_mad"],
                    f"type_n_{window}": estimate["type_n"],
                    f"dow_n_{window}": estimate["dow_n"],
                    f"fixed_lower_{window}": max(anchor - fixed_width, 0.0) if pd.notna(anchor) else np.nan,
                    f"fixed_upper_{window}": anchor + fixed_width if pd.notna(anchor) else np.nan,
                    f"dynamic_lower_{window}": max(anchor - dynamic_width, 0.0) if pd.notna(anchor) else np.nan,
                    f"dynamic_upper_{window}": anchor + dynamic_width if pd.notna(anchor) else np.nan,
                }
            )
            plain = (
                estimate_anchor(history, row.date, window, robust=False)
                if history is not None
                else {"anchor": np.nan}
            )
            plain_anchor = plain["anchor"]
            plain_width = max(0.15 * plain_anchor, 2.0) if pd.notna(plain_anchor) else np.nan
            record.update(
                {
                    f"plain_anchor_{window}": plain_anchor,
                    f"plain_fixed_lower_{window}": (
                        max(plain_anchor - plain_width, 0.0) if pd.notna(plain_anchor) else np.nan
                    ),
                    f"plain_fixed_upper_{window}": (
                        plain_anchor + plain_width if pd.notna(plain_anchor) else np.nan
                    ),
                }
            )
        records.append(record)
    anchors = pd.DataFrame(records)
    result = target.merge(anchors, on=KEYS, how="left", validate="one_to_one")
    for window in [28, 35]:
        result[f"guard_fixed_{window}"] = result["forecast_qty"].clip(
            lower=result[f"fixed_lower_{window}"], upper=result[f"fixed_upper_{window}"]
        ).fillna(result["forecast_qty"])
        result[f"guard_dynamic_{window}"] = result["forecast_qty"].clip(
            lower=result[f"dynamic_lower_{window}"], upper=result[f"dynamic_upper_{window}"]
        ).fillna(result["forecast_qty"])
        result[f"guard_upper_dynamic_{window}"] = np.minimum(
            result["forecast_qty"], result[f"dynamic_upper_{window}"].fillna(result["forecast_qty"])
        )
        result[f"guard_plain_fixed_{window}"] = result["forecast_qty"].clip(
            lower=result[f"plain_fixed_lower_{window}"], upper=result[f"plain_fixed_upper_{window}"]
        ).fillna(result["forecast_qty"])
        result[f"guard_upper_fixed_{window}"] = np.minimum(
            result["forecast_qty"], result[f"fixed_upper_{window}"].fillna(result["forecast_qty"])
        )
        result[f"guard_lower_fixed_{window}"] = np.maximum(
            result["forecast_qty"], result[f"fixed_lower_{window}"].fillna(result["forecast_qty"])
        )
    return result


def forecast_summary(rows: pd.DataFrame, scope: str, period: str) -> dict[str, object]:
    demand = rows["demand"].to_numpy(dtype=float)
    forecast = rows["target"].to_numpy(dtype=float)
    error = forecast - demand
    return {
        "scope": scope,
        "period": period,
        "variant": rows["variant"].iloc[0],
        "rows": len(rows),
        "forecast": forecast.sum(),
        "demand": demand.sum(),
        "wape_pct": 100 * np.abs(error).sum() / max(demand.sum(), 1e-9),
        "bias_pct": 100 * error.sum() / max(demand.sum(), 1e-9),
        "under_qty": np.maximum(-error, 0.0).sum(),
        "over_qty": np.maximum(error, 0.0).sum(),
    }


def main() -> None:
    historical_demand = load_historical_demand()
    target = add_guardrails(load_targets(), historical_demand)
    variants = {
        "direct": "forecast_qty",
        "guard_fixed_28": "guard_fixed_28",
        "guard_plain_fixed_28": "guard_plain_fixed_28",
        "guard_upper_fixed_28": "guard_upper_fixed_28",
        "guard_lower_fixed_28": "guard_lower_fixed_28",
        "guard_dynamic_28": "guard_dynamic_28",
        "guard_upper_dynamic_28": "guard_upper_dynamic_28",
        "guard_fixed_35": "guard_fixed_35",
        "guard_plain_fixed_35": "guard_plain_fixed_35",
        "guard_dynamic_35": "guard_dynamic_35",
        "guard_upper_dynamic_35": "guard_upper_dynamic_35",
    }

    simulations = [build_actual_state(target)]
    for variant, column in variants.items():
        part = simulate_variant(target, variant, column)
        part["target"] = part[column]
        simulations.append(part)
    simulated = pd.concat(simulations, ignore_index=True)

    summaries = []
    forecast_summaries = []
    scopes = {"all55": set(target["bakery_id"].unique()), "regina3": REGINA_BAKERIES}
    periods = {
        "combined": (pd.Timestamp("2026-08-24"), pd.Timestamp("2026-09-09")),
        "aug24_31": (pd.Timestamp("2026-08-24"), pd.Timestamp("2026-08-31")),
        "sep01_09": (pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-09")),
    }
    for scope, bakery_ids in scopes.items():
        for period, (date_from, date_to) in periods.items():
            selected = simulated[
                simulated["bakery_id"].isin(bakery_ids)
                & simulated["date"].between(date_from, date_to)
            ]
            for variant, group in selected.groupby("variant", sort=False):
                summaries.append(
                    {
                        "scope": scope,
                        "period": period,
                        "variant": variant,
                        "production": group["production"].sum(),
                        "demand": group["demand"].sum(),
                        "served": group["served"].sum(),
                        "lost": group["lost"].sum(),
                        "lost_cases": int(group["lost_case"].sum()),
                        "ending_stock": group["ending_stock"].sum(),
                        "expired_old": group["expired_old"].sum(),
                        "gross_profit": group["gross_profit"].sum(),
                        "service_level_pct": 100 * group["served"].sum() / max(group["demand"].sum(), 1e-9),
                    }
                )
                if variant != "actual_state":
                    forecast_summaries.append(forecast_summary(group, scope, period))

    summary = pd.DataFrame(summaries)
    for (scope, period), indices in summary.groupby(["scope", "period"]).groups.items():
        block = summary.loc[indices]
        direct = block[block["variant"].eq("direct")].iloc[0]
        actual = block[block["variant"].eq("actual_state")].iloc[0]
        summary.loc[indices, "gross_profit_delta_vs_direct"] = block["gross_profit"] - direct["gross_profit"]
        summary.loc[indices, "lost_delta_vs_direct"] = block["lost"] - direct["lost"]
        summary.loc[indices, "lost_cases_delta_vs_direct"] = block["lost_cases"] - direct["lost_cases"]
        summary.loc[indices, "ending_stock_delta_vs_direct"] = block["ending_stock"] - direct["ending_stock"]
        summary.loc[indices, "gross_profit_delta_vs_actual"] = block["gross_profit"] - actual["gross_profit"]

    forecast_metrics = pd.DataFrame(forecast_summaries)
    coverage = []
    for window in [28, 35]:
        available = target[f"anchor_{window}"].notna()
        changed_dynamic = available & ~np.isclose(target["forecast_qty"], target[f"guard_dynamic_{window}"])
        changed_fixed = available & ~np.isclose(target["forecast_qty"], target[f"guard_fixed_{window}"])
        coverage.append(
            {
                "window_days": window,
                "rows": len(target),
                "anchor_available": int(available.sum()),
                "anchor_available_pct": 100 * available.mean(),
                "fixed_changed": int(changed_fixed.sum()),
                "fixed_changed_pct": 100 * changed_fixed.mean(),
                "dynamic_changed": int(changed_dynamic.sum()),
                "dynamic_changed_pct": 100 * changed_dynamic.mean(),
            }
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    target.to_parquet(OUTPUT / "forecast_rows.parquet", index=False)
    summary.to_csv(OUTPUT / "economics_summary.csv", index=False, encoding="utf-8-sig")
    forecast_metrics.to_csv(OUTPUT / "forecast_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(coverage).to_csv(OUTPUT / "coverage.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(coverage).to_string(index=False))
    print("\nAll 55, combined")
    columns = [
        "variant", "production", "lost", "lost_cases", "ending_stock", "gross_profit",
        "gross_profit_delta_vs_direct", "lost_delta_vs_direct", "lost_cases_delta_vs_direct",
    ]
    print(summary[(summary["scope"] == "all55") & (summary["period"] == "combined")][columns].to_string(index=False))
    print("\nForecast metrics, all 55, combined")
    combined_metrics = forecast_metrics[
        (forecast_metrics["scope"] == "all55") & (forecast_metrics["period"] == "combined")
    ]
    print(combined_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
