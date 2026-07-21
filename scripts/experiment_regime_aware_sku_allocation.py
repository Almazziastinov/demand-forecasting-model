"""Leakage-safe allocation experiment on the complete forecast universe.

The experiment corrects smoothed SKU-share residuals for historically labelled
products, but renormalizes over every forecasted SKU in the bakery-day. It is
read only with respect to ClickHouse and writes local reports only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_stockout_allocation_failures import SALE_EVENT_HEX  # noqa: E402
from scripts.export_clickhouse_checks import create_client  # noqa: E402

DEFAULT_INPUT = ROOT / "reports/pilot_stockout_forecast_bias/sku_day_comparison.csv"
DEFAULT_STABILITY = ROOT / "reports/stockout_historical_shadow/bakery_sku_stability.csv"
DEFAULT_OUTPUT = ROOT / "reports/regime_aware_sku_allocation_experiment"
KEYS = ["date", "bakery_id", "product_id"]


def choose_dominant_runs(labels: pd.DataFrame) -> pd.DataFrame:
    """Choose the run supporting most labelled SKU for each bakery-day."""
    work = labels.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    work = work[work["source_run_id"].notna()].copy()
    if work.empty:
        raise ValueError("no source_run_id values are available")
    work["generated_order"] = pd.to_datetime(
        work["latest_generated_at"], errors="coerce", utc=True
    )
    choices = work.groupby(["date", "bakery_id", "source_run_id"], as_index=False).agg(
        labelled_rows=("product_id", "size"),
        generated_order=("generated_order", "max"),
    )
    choices = choices.sort_values(
        ["date", "bakery_id", "labelled_rows", "generated_order", "source_run_id"],
        ascending=[True, True, False, False, False],
    )
    return choices.drop_duplicates(["date", "bakery_id"])[
        ["date", "bakery_id", "source_run_id", "labelled_rows"]
    ].reset_index(drop=True)


def load_forecast_universe(client, choices: pd.DataFrame) -> pd.DataFrame:
    """Load every forecasted SKU for the selected historical bakery-day runs."""
    run_ids = sorted(choices["source_run_id"].unique().tolist())
    bakery_ids = sorted(choices["bakery_id"].astype(int).unique().tolist())
    forecasts = client.query_df(
        """
        select
            source_run_id,
            forecast_date as date,
            bakery_id,
            product_id,
            product_name,
            category_name,
            forecast_qty,
            generated_at
        from sku_forecast_day_snapshots final
        where source_run_id in %(run_ids)s
          and bakery_id in %(bakery_ids)s
          and forecast_date between toDate(%(date_from)s) and toDate(%(date_to)s)
        """,
        parameters={
            "run_ids": run_ids,
            "bakery_ids": bakery_ids,
            "date_from": str(choices["date"].min().date()),
            "date_to": str(choices["date"].max().date()),
        },
    )
    forecasts["date"] = pd.to_datetime(forecasts["date"]).dt.normalize()
    selected = choices.merge(
        forecasts,
        on=["date", "bakery_id", "source_run_id"],
        how="left",
        validate="one_to_many",
    )
    selected = selected.sort_values("generated_at").drop_duplicates(KEYS, keep="last")
    if selected["forecast_qty"].isna().any():
        missing = selected.loc[selected["forecast_qty"].isna(), ["date", "bakery_id"]]
        sample = missing.head().to_dict("records")
        raise ValueError(f"selected runs have missing forecast rows: {sample}")
    return selected


def load_actual_sales(client, choices: pd.DataFrame) -> pd.DataFrame:
    """Load complete actual SKU sales and bakery totals for the replay dates."""
    bakery_ids = sorted(choices["bakery_id"].astype(int).unique().tolist())
    return client.query_df(
        f"""
        select
            check_date as date,
            toInt64(bakery_id) as bakery_id,
            toInt64(product_id) as product_id,
            sum(quantity) as daily_sold
        from mart_sales_60d as m
        where m.check_date between toDate(%(date_from)s) and toDate(%(date_to)s)
          and toInt64(m.bakery_id) in %(bakery_ids)s
          and m.quantity > 0
          and hex(m.cash_event_type) = '{SALE_EVENT_HEX}'
        group by date, bakery_id, product_id
        """,
        parameters={
            "date_from": str(choices["date"].min().date()),
            "date_to": str(choices["date"].max().date()),
            "bakery_ids": bakery_ids,
        },
    )


def prepare_universe(
    forecasts: pd.DataFrame,
    actual: pd.DataFrame,
    labels: pd.DataFrame,
    stability: pd.DataFrame,
) -> pd.DataFrame:
    """Create comparable forecast and observed shares on the full universe."""
    work = forecasts.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    actual_work = actual.copy()
    actual_work["date"] = pd.to_datetime(actual_work["date"]).dt.normalize()
    bakery_actual = actual_work.groupby(["date", "bakery_id"], as_index=False).agg(
        bakery_actual_qty=("daily_sold", "sum")
    )
    work = work.merge(actual_work, on=KEYS, how="left", validate="one_to_one")
    work["daily_sold"] = work["daily_sold"].fillna(0.0)
    work = work.merge(bakery_actual, on=["date", "bakery_id"], how="left")

    label_columns = [*KEYS, "stockout_group", "has_forecast"]
    label_work = labels[label_columns].copy()
    label_work["date"] = pd.to_datetime(label_work["date"]).dt.normalize()
    work = work.merge(label_work, on=KEYS, how="left", validate="one_to_one")
    work["stockout_group"] = work["stockout_group"].fillna("unlabelled")
    work["is_labelled_candidate"] = work["stockout_group"].ne("unlabelled")
    screened_pairs = (
        labels[["bakery_id", "product_id"]]
        .drop_duplicates()
        .assign(is_screened_pair=True)
    )
    work = work.merge(
        screened_pairs,
        on=["bakery_id", "product_id"],
        how="left",
        validate="many_to_one",
    )
    work["is_screened_pair"] = work["is_screened_pair"].eq(True)

    segment_columns = [
        "bakery_id",
        "product_id",
        "recurrent_allocation",
        "is_bakery_top5_by_sales",
        "is_potentially_problematic",
    ]
    work = work.merge(
        stability[segment_columns].drop_duplicates(["bakery_id", "product_id"]),
        on=["bakery_id", "product_id"],
        how="left",
        validate="many_to_one",
    )
    for column in [
        "recurrent_allocation",
        "is_bakery_top5_by_sales",
        "is_potentially_problematic",
    ]:
        work[column] = work[column].eq(True)

    work["baseline_total"] = work.groupby(["date", "bakery_id"])[
        "forecast_qty"
    ].transform("sum")
    work["baseline_share"] = work["forecast_qty"] / work["baseline_total"].replace(
        0.0, np.nan
    )
    work["observed_share"] = work["daily_sold"] / work["bakery_actual_qty"].replace(
        0.0, np.nan
    )
    ratio = work["observed_share"] / work["baseline_share"].replace(0.0, np.nan)
    work["observed_log_residual"] = np.log(ratio.clip(0.25, 4.0))
    work.loc[~work["is_labelled_candidate"], "observed_log_residual"] = np.nan
    return work.sort_values(KEYS).reset_index(drop=True)


def _median_absolute_deviation(values: pd.Series) -> float:
    median = float(values.median())
    return float((values - median).abs().median())


def build_walk_forward_signals(
    frame: pd.DataFrame,
    *,
    lookback_days: int = 42,
    recent_days: int = 14,
    min_history_days: int = 4,
    min_regime_days: int = 3,
    min_direction_consistency: float = 0.65,
    max_log_mad: float = 0.70,
    regime_shift_threshold: float = 0.20,
) -> pd.DataFrame:
    """Build smoothed residual signals using strictly earlier normal days."""
    work = frame.copy()
    pair_group = work.groupby(["bakery_id", "product_id"], sort=False)
    work["prior_sales_q75_28"] = pair_group["daily_sold"].transform(
        lambda values: values.shift(1).rolling(28, min_periods=7).quantile(0.75)
    )
    work["prior_sales_q90_28"] = pair_group["daily_sold"].transform(
        lambda values: values.shift(1).rolling(28, min_periods=7).quantile(0.90)
    )
    work["prior_sales_q95_28"] = pair_group["daily_sold"].transform(
        lambda values: values.shift(1).rolling(28, min_periods=7).quantile(0.95)
    )
    work["prior_sales_days_28"] = pair_group["daily_sold"].transform(
        lambda values: values.shift(1).rolling(28, min_periods=1).count()
    )
    signal_columns = {
        "history_days": 0,
        "recent_history_days": 0,
        "older_history_days": 0,
        "stable_log_residual": 0.0,
        "upper_log_residual": 0.0,
        "risk_log_residual": 0.0,
        "recent_log_residual": 0.0,
        "older_log_residual": 0.0,
        "residual_log_mad": np.nan,
        "direction_consistency": 0.0,
        "prior_stockout_rate_14": 0.0,
        "regime_shift": 0.0,
        "regime_confirmed": False,
        "signal_reliability": 0.0,
        "allocation_eligible": False,
        "risk_eligible": False,
    }
    for column, default in signal_columns.items():
        work[column] = default

    candidate_indexes = work.index[work["is_labelled_candidate"]]
    grouped = (
        work.loc[candidate_indexes]
        .groupby(["bakery_id", "product_id"], sort=False)
        .groups
    )
    for indexes in grouped.values():
        pair = work.loc[indexes].sort_values("date")
        for index, row in pair.iterrows():
            cutoff = row["date"]
            start = cutoff - pd.Timedelta(days=lookback_days)
            history_all = pair[pair["date"].lt(cutoff) & pair["date"].ge(start)]
            history = history_all[
                history_all["stockout_group"].eq("confirmed_non_stockout")
                & history_all["observed_log_residual"].notna()
            ]
            values = history["observed_log_residual"]
            history_days = int(history["date"].nunique())
            work.at[index, "history_days"] = history_days
            if history_days == 0:
                continue

            stable = float(values.median())
            upper = float(values.quantile(0.75))
            mad = _median_absolute_deviation(values)
            if stable > 0:
                consistency = float(values.gt(0).mean())
            elif stable < 0:
                consistency = float(values.lt(0).mean())
            else:
                consistency = 0.5
            recent = history[
                history["date"].ge(cutoff - pd.Timedelta(days=recent_days))
            ]
            older = history[history["date"].lt(cutoff - pd.Timedelta(days=recent_days))]
            recent_value = (
                float(recent["observed_log_residual"].median())
                if len(recent)
                else stable
            )
            older_value = (
                float(older["observed_log_residual"].median()) if len(older) else stable
            )
            shift = recent_value - older_value
            regime_confirmed = (
                recent["date"].nunique() >= min_regime_days
                and older["date"].nunique() >= min_regime_days
                and abs(shift) >= regime_shift_threshold
            )
            reliability = (
                history_days
                / (history_days + 7.0)
                * max(0.0, (consistency - 0.5) / 0.5)
                * np.exp(-mad)
            )
            eligible = (
                history_days >= min_history_days
                and consistency >= min_direction_consistency
                and mad <= max_log_mad
                and abs(stable) >= 0.05
            )
            prior_stockout = history_all[
                history_all["date"].ge(cutoff - pd.Timedelta(days=14))
            ]["stockout_group"].eq("clear_stockout")
            prior_stockout_rate = (
                float(prior_stockout.mean()) if len(prior_stockout) else 0.0
            )
            risk_signal = max(0.0, upper) * min(1.0, prior_stockout_rate / 0.25)
            risk_eligible = (
                history_days >= min_history_days
                and upper >= 0.05
                and prior_stockout_rate >= 0.10
                and mad <= max_log_mad
            )
            work.at[index, "stable_log_residual"] = stable
            work.at[index, "upper_log_residual"] = upper
            work.at[index, "risk_log_residual"] = risk_signal
            work.at[index, "recent_log_residual"] = recent_value
            work.at[index, "older_log_residual"] = older_value
            work.at[index, "residual_log_mad"] = mad
            work.at[index, "direction_consistency"] = consistency
            work.at[index, "recent_history_days"] = int(recent["date"].nunique())
            work.at[index, "older_history_days"] = int(older["date"].nunique())
            work.at[index, "prior_stockout_rate_14"] = prior_stockout_rate
            work.at[index, "regime_shift"] = shift
            work.at[index, "regime_confirmed"] = regime_confirmed
            work.at[index, "signal_reliability"] = reliability
            work.at[index, "allocation_eligible"] = eligible
            work.at[index, "risk_eligible"] = risk_eligible
    return work


def apply_guarded_allocation(
    frame: pd.DataFrame,
    *,
    signal_mode: str,
    strength: float,
    max_shift_fraction: float,
    max_sku_change: float = 0.20,
) -> pd.DataFrame:
    """Apply residual corrections and preserve the complete bakery-day total."""
    if signal_mode not in {"stable", "regime", "risk"}:
        raise ValueError("signal_mode must be stable, regime, or risk")
    work = frame.copy()
    signal = work["stable_log_residual"].copy()
    if signal_mode == "regime":
        signal = signal.where(~work["regime_confirmed"], work["recent_log_residual"])
    elif signal_mode == "risk":
        signal = work["risk_log_residual"]
    effective = signal * work["signal_reliability"] * strength
    eligibility = (
        work["risk_eligible"] if signal_mode == "risk" else work["allocation_eligible"]
    )
    effective = effective.where(eligibility, 0.0)
    work["scenario_eligible"] = eligibility
    correction = np.exp(effective).clip(1.0 - max_sku_change, 1.0 + max_sku_change)
    work["raw_weight"] = work["baseline_share"] * correction
    raw_total = work.groupby(["date", "bakery_id"])["raw_weight"].transform("sum")
    work["raw_adjusted_share"] = work["raw_weight"] / raw_total.replace(0.0, np.nan)
    work["raw_shift_share"] = (
        work["raw_adjusted_share"] - work["baseline_share"]
    ).abs()
    shift_fraction = (
        work.groupby(["date", "bakery_id"])["raw_shift_share"].transform("sum") / 2.0
    )
    blend = (max_shift_fraction / shift_fraction.replace(0.0, np.nan)).clip(upper=1.0)
    blend = blend.fillna(1.0)
    work["adjusted_share"] = work["baseline_share"] + blend * (
        work["raw_adjusted_share"] - work["baseline_share"]
    )
    work["adjusted_forecast_qty"] = work["baseline_total"] * work["adjusted_share"]
    work["shifted_qty"] = (work["adjusted_forecast_qty"] - work["forecast_qty"]).abs()
    work["scenario_signal"] = signal
    return work


def apply_positive_capacity_allocation(
    frame: pd.DataFrame,
    *,
    signal_mode: str,
    strength: float,
    max_shift_fraction: float,
    max_sku_uplift: float = 0.20,
    donor_quantile_column: str = "prior_sales_q75_28",
    donor_floor_margin: float = 0.0,
) -> pd.DataFrame:
    """Fund positive residuals only from forecast headroom above prior q75 sales."""
    if signal_mode not in {"stable", "regime", "risk"}:
        raise ValueError("signal_mode must be stable, regime, or risk")
    work = frame.copy()
    signal = work["stable_log_residual"].copy()
    if signal_mode == "regime":
        signal = signal.where(~work["regime_confirmed"], work["recent_log_residual"])
    elif signal_mode == "risk":
        signal = work["risk_log_residual"]
    positive_signal = signal.clip(lower=0.0)
    effective = positive_signal * work["signal_reliability"] * strength
    eligibility = (
        work["risk_eligible"] if signal_mode == "risk" else work["allocation_eligible"]
    )
    effective = effective.where(eligibility, 0.0)
    work["scenario_eligible"] = eligibility
    uplift_ratio = (np.exp(effective) - 1.0).clip(0.0, max_sku_uplift)
    work["requested_uplift"] = work["forecast_qty"] * uplift_ratio

    has_donor_history = work["prior_sales_days_28"].ge(7)
    donor_floor = work[donor_quantile_column] + donor_floor_margin
    work["donor_capacity"] = (work["forecast_qty"] - donor_floor).clip(lower=0.0)
    work.loc[
        ~has_donor_history | ~work["is_screened_pair"] | work["requested_uplift"].gt(0),
        "donor_capacity",
    ] = 0.0
    group_keys = ["date", "bakery_id"]
    requested_total = work.groupby(group_keys)["requested_uplift"].transform("sum")
    capacity_total = work.groupby(group_keys)["donor_capacity"].transform("sum")
    budget_total = work["baseline_total"] * max_shift_fraction
    available = pd.concat([capacity_total, budget_total], axis=1).min(axis=1)
    request_scale = (available / requested_total.replace(0.0, np.nan)).clip(upper=1.0)
    request_scale = request_scale.fillna(0.0)
    work["applied_uplift"] = work["requested_uplift"] * request_scale
    applied_total = work.groupby(group_keys)["applied_uplift"].transform("sum")
    donor_share = work["donor_capacity"] / capacity_total.replace(0.0, np.nan)
    work["donor_deduction"] = donor_share.fillna(0.0) * applied_total
    work["adjusted_forecast_qty"] = (
        work["forecast_qty"] + work["applied_uplift"] - work["donor_deduction"]
    )
    work["adjusted_share"] = work["adjusted_forecast_qty"] / work[
        "baseline_total"
    ].replace(0.0, np.nan)
    work["shifted_qty"] = (work["adjusted_forecast_qty"] - work["forecast_qty"]).abs()
    work["scenario_signal"] = signal
    return work


def _segment_mask(frame: pd.DataFrame, segment: str) -> pd.Series:
    if segment == "all_labelled":
        return frame["is_labelled_candidate"]
    if segment == "recurrent_allocation":
        return frame["recurrent_allocation"]
    if segment == "recurrent_top5":
        return frame["recurrent_allocation"] & frame["is_bakery_top5_by_sales"]
    if segment == "recurrent_other":
        return frame["recurrent_allocation"] & ~frame["is_bakery_top5_by_sales"]
    if segment == "other_labelled":
        return frame["is_labelled_candidate"] & ~frame["recurrent_allocation"]
    raise ValueError(f"unknown segment: {segment}")


def evaluate_segment(frame: pd.DataFrame, *, segment: str) -> dict[str, object]:
    subset = frame[_segment_mask(frame, segment)]
    stockout = subset[subset["stockout_group"].eq("clear_stockout")]
    normal = subset[subset["stockout_group"].eq("confirmed_non_stockout")]
    stockout_before = (stockout["daily_sold"] - stockout["forecast_qty"]).clip(
        lower=0.0
    )
    stockout_after = (stockout["daily_sold"] - stockout["adjusted_forecast_qty"]).clip(
        lower=0.0
    )
    normal_before = normal["daily_sold"] - normal["forecast_qty"]
    normal_after = normal["daily_sold"] - normal["adjusted_forecast_qty"]
    return {
        "segment": segment,
        "rows": int(len(subset)),
        "stockout_rows": int(len(stockout)),
        "normal_rows": int(len(normal)),
        "stockout_shortfall_before": float(stockout_before.sum()),
        "stockout_shortfall_after": float(stockout_after.sum()),
        "stockout_shortfall_reduction": float(
            stockout_before.sum() - stockout_after.sum()
        ),
        "stockout_cases_fixed": int(
            ((stockout_before > 0.5) & (stockout_after <= 0.5)).sum()
        ),
        "stockout_new_underforecast": int(
            ((stockout_before <= 0.5) & (stockout_after > 0.5)).sum()
        ),
        "normal_mae_before": float(normal_before.abs().mean()),
        "normal_mae_after": float(normal_after.abs().mean()),
        "normal_bias_before": float(normal_before.mean()),
        "normal_bias_after": float(normal_after.mean()),
        "normal_new_underforecast": int(
            ((normal_before <= 0.5) & (normal_after > 0.5)).sum()
        ),
    }


def evaluate_scenario(
    frame: pd.DataFrame, *, scenario: str
) -> tuple[dict[str, object], pd.DataFrame]:
    segments = [
        "all_labelled",
        "recurrent_allocation",
        "recurrent_top5",
        "recurrent_other",
        "other_labelled",
    ]
    rows = pd.DataFrame(
        [evaluate_segment(frame, segment=segment) for segment in segments]
    )
    rows.insert(0, "scenario", scenario)
    overall = rows[rows["segment"].eq("all_labelled")].iloc[0]
    recurrent = rows[rows["segment"].eq("recurrent_allocation")].iloc[0]
    universe_before = frame["daily_sold"] - frame["forecast_qty"]
    universe_after = frame["daily_sold"] - frame["adjusted_forecast_qty"]
    original_totals = frame.groupby(["date", "bakery_id"])["forecast_qty"].sum()
    adjusted_totals = frame.groupby(["date", "bakery_id"])[
        "adjusted_forecast_qty"
    ].sum()
    summary = {
        "scenario": scenario,
        "stockout_shortfall_after": float(overall["stockout_shortfall_after"]),
        "stockout_shortfall_reduction": float(overall["stockout_shortfall_reduction"]),
        "stockout_cases_fixed": int(overall["stockout_cases_fixed"]),
        "stockout_new_underforecast": int(overall["stockout_new_underforecast"]),
        "normal_mae_after": float(overall["normal_mae_after"]),
        "normal_mae_delta": float(
            overall["normal_mae_after"] - overall["normal_mae_before"]
        ),
        "normal_new_underforecast": int(overall["normal_new_underforecast"]),
        "recurrent_shortfall_reduction": float(
            recurrent["stockout_shortfall_reduction"]
        ),
        "universe_mae_after": float(universe_after.abs().mean()),
        "universe_mae_delta": float(
            universe_after.abs().mean() - universe_before.abs().mean()
        ),
        "universe_shortfall_delta": float(
            universe_after.clip(lower=0.0).sum() - universe_before.clip(lower=0.0).sum()
        ),
        "universe_new_underforecast": int(
            ((universe_before <= 0.5) & (universe_after > 0.5)).sum()
        ),
        "eligible_rows": int(frame["scenario_eligible"].sum()),
        "shifted_units": float(frame["shifted_qty"].sum() / 2.0),
        "max_bakery_total_delta": float(
            (original_totals - adjusted_totals).abs().max()
        ),
    }
    return summary, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--stability", default=str(DEFAULT_STABILITY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = pd.read_csv(args.input, encoding="utf-8-sig")
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    stability = pd.read_csv(args.stability, encoding="utf-8-sig")
    choices = choose_dominant_runs(labels)
    client = create_client(args.env_file)
    forecasts = load_forecast_universe(client, choices)
    actual = load_actual_sales(client, choices)
    universe = prepare_universe(forecasts, actual, labels, stability)
    featured = build_walk_forward_signals(universe)

    scenario_rows: list[dict[str, object]] = []
    segment_rows: list[pd.DataFrame] = []
    details: dict[str, pd.DataFrame] = {}
    for signal_mode in ["stable", "regime"]:
        for strength in [0.25, 0.50, 0.75, 1.00]:
            for budget in [0.01, 0.025, 0.05]:
                name = f"{signal_mode}_strength_{strength:.2f}_budget_{budget:.3f}"
                adjusted = apply_guarded_allocation(
                    featured,
                    signal_mode=signal_mode,
                    strength=strength,
                    max_shift_fraction=budget,
                )
                summary, segments = evaluate_scenario(adjusted, scenario=name)
                summary.update(
                    {
                        "allocation_method": "renormalized_weights",
                        "signal_mode": signal_mode,
                        "strength": strength,
                        "budget": budget,
                    }
                )
                scenario_rows.append(summary)
                segment_rows.append(segments)
                details[name] = adjusted
    donor_specs = [
        ("q75", "prior_sales_q75_28", 0.0),
        ("q90m05", "prior_sales_q90_28", 0.5),
        ("q90m10", "prior_sales_q90_28", 1.0),
        ("q90m20", "prior_sales_q90_28", 2.0),
        ("q90m30", "prior_sales_q90_28", 3.0),
        ("q90m50", "prior_sales_q90_28", 5.0),
        ("q95m20", "prior_sales_q95_28", 2.0),
        ("q95m50", "prior_sales_q95_28", 5.0),
    ]
    for signal_mode in ["stable", "regime"]:
        for strength in [0.25, 0.50, 0.75, 1.00]:
            for budget in [0.0025, 0.005, 0.01]:
                for donor_name, donor_column, donor_margin in donor_specs:
                    name = (
                        f"positive_capacity_{signal_mode}_{donor_name}"
                        f"_strength_{strength:.2f}_budget_{budget:.4f}"
                    )
                    adjusted = apply_positive_capacity_allocation(
                        featured,
                        signal_mode=signal_mode,
                        strength=strength,
                        max_shift_fraction=budget,
                        donor_quantile_column=donor_column,
                        donor_floor_margin=donor_margin,
                    )
                    summary, segment_result = evaluate_scenario(adjusted, scenario=name)
                    summary.update(
                        {
                            "allocation_method": "positive_capacity",
                            "signal_mode": signal_mode,
                            "strength": strength,
                            "budget": budget,
                            "donor_rule": donor_name,
                        }
                    )
                    scenario_rows.append(summary)
                    segment_rows.append(segment_result)
                    details[name] = adjusted
    risk_donor_specs = [
        ("q90m05", "prior_sales_q90_28", 0.5),
        ("q90m10", "prior_sales_q90_28", 1.0),
        ("q90m20", "prior_sales_q90_28", 2.0),
    ]
    for strength in [0.50, 1.00]:
        for budget in [0.0025, 0.005]:
            for donor_name, donor_column, donor_margin in risk_donor_specs:
                name = (
                    f"positive_capacity_risk_{donor_name}"
                    f"_strength_{strength:.2f}_budget_{budget:.4f}"
                )
                adjusted = apply_positive_capacity_allocation(
                    featured,
                    signal_mode="risk",
                    strength=strength,
                    max_shift_fraction=budget,
                    donor_quantile_column=donor_column,
                    donor_floor_margin=donor_margin,
                )
                summary, segment_result = evaluate_scenario(adjusted, scenario=name)
                summary.update(
                    {
                        "allocation_method": "positive_capacity",
                        "signal_mode": "risk",
                        "strength": strength,
                        "budget": budget,
                        "donor_rule": donor_name,
                    }
                )
                scenario_rows.append(summary)
                segment_rows.append(segment_result)
                details[name] = adjusted

    comparison = pd.DataFrame(scenario_rows)
    baseline_mae = evaluate_segment(
        featured.assign(adjusted_forecast_qty=featured["forecast_qty"]),
        segment="all_labelled",
    )["normal_mae_before"]
    comparison["passes_gates"] = (
        comparison["stockout_shortfall_reduction"].ge(0.0)
        & comparison["recurrent_shortfall_reduction"].gt(0.0)
        & comparison["normal_mae_after"].le(float(baseline_mae))
        & comparison["universe_mae_delta"].le(0.0)
        & comparison["universe_shortfall_delta"].le(0.0)
        & comparison["stockout_new_underforecast"].eq(0)
        & comparison["normal_new_underforecast"].eq(0)
        & comparison["universe_new_underforecast"].eq(0)
    )
    comparison = comparison.sort_values(
        [
            "passes_gates",
            "universe_new_underforecast",
            "recurrent_shortfall_reduction",
            "stockout_shortfall_reduction",
            "normal_mae_after",
        ],
        ascending=[False, True, False, False, True],
    ).reset_index(drop=True)
    best = str(comparison.iloc[0]["scenario"])
    review_candidates = comparison[
        comparison["stockout_shortfall_reduction"].gt(0.0)
        & comparison["recurrent_shortfall_reduction"].gt(0.0)
        & comparison["normal_mae_delta"].le(0.0)
        & comparison["universe_mae_delta"].le(0.0)
        & comparison["universe_shortfall_delta"].le(0.0)
        & comparison["stockout_new_underforecast"].eq(0)
        & comparison["normal_new_underforecast"].eq(0)
    ].sort_values(
        ["universe_new_underforecast", "recurrent_shortfall_reduction"],
        ascending=[True, False],
    )
    review_candidate = (
        str(review_candidates.iloc[0]["scenario"])
        if not review_candidates.empty
        else best
    )
    segments = pd.concat(segment_rows, ignore_index=True)
    best_segments = segments[segments["scenario"].eq(best)].to_dict(orient="records")
    risk_comparison = comparison[comparison["signal_mode"].eq("risk")]
    best_risk = (
        risk_comparison.sort_values(
            ["universe_new_underforecast", "stockout_shortfall_reduction"],
            ascending=[True, False],
        ).iloc[0]
        if not risk_comparison.empty
        else None
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    choices.to_csv(output / "selected_runs.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output / "scenario_comparison.csv", index=False)
    segments.to_csv(output / "segment_comparison.csv", index=False)
    detail_columns = [
        *KEYS,
        "product_name",
        "stockout_group",
        "forecast_qty",
        "adjusted_forecast_qty",
        "daily_sold",
        "baseline_share",
        "adjusted_share",
        "history_days",
        "stable_log_residual",
        "recent_log_residual",
        "regime_shift",
        "regime_confirmed",
        "signal_reliability",
        "allocation_eligible",
        "recurrent_allocation",
        "is_bakery_top5_by_sales",
        "prior_sales_q75_28",
        "prior_sales_q90_28",
        "prior_sales_q95_28",
        "requested_uplift",
        "applied_uplift",
        "donor_capacity",
        "donor_deduction",
    ]
    available_detail_columns = [
        column for column in detail_columns if column in details[best].columns
    ]
    details[best][available_detail_columns].to_csv(
        output / "best_scenario_rows.csv", index=False, encoding="utf-8-sig"
    )
    review_detail_columns = [
        column
        for column in detail_columns
        if column in details[review_candidate].columns
    ]
    details[review_candidate][review_detail_columns].to_csv(
        output / "review_candidate_rows.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "best_scenario": best,
        "best_metrics": comparison.iloc[0].to_dict(),
        "review_candidate": review_candidate,
        "review_candidate_metrics": (
            review_candidates.iloc[0].to_dict()
            if not review_candidates.empty
            else comparison.iloc[0].to_dict()
        ),
        "best_segment_metrics": best_segments,
        "best_risk_metrics": best_risk.to_dict() if best_risk is not None else None,
        "selected_bakery_days": int(len(choices)),
        "full_universe_rows": int(len(universe)),
        "labelled_rows": int(universe["is_labelled_candidate"].sum()),
        "labelled_forecast_coverage": float(
            universe["is_labelled_candidate"].sum() / len(labels)
        ),
        "zero_or_missing_forecast_label_rows": int((~labels["has_forecast"]).sum()),
        "zero_or_missing_forecast_stockout_rows": int(
            (
                ~labels["has_forecast"] & labels["stockout_group"].eq("clear_stockout")
            ).sum()
        ),
        "share_denominator": "complete_selected_run_bakery_day_forecast",
        "training": "walk_forward_prior_confirmed_non_stockout_only",
        "bakery_total_constrained": True,
        "production_write": False,
        "decision": (
            "eligible_for_shadow"
            if bool(comparison.iloc[0]["passes_gates"])
            else "rejected"
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
