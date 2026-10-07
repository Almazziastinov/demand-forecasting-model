"""Causal replay of the production Direct architecture with demand guardrails.

The production artifact itself was trained through 2026-08-23 and therefore
cannot be backcast to earlier months without leakage.  This script retrains the
same SKU architecture at each monthly cutoff, then evaluates the unchanged
guardrail policies with the canonical stateful economics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.backtest_corrected_direct_june_july as direct_replay  # noqa: E402
import scripts.backtest_regime_aware_guardrail_aug24_sep09 as seasonal_helpers  # noqa: E402
from scripts.backtest_direct_demand_guardrail_aug24_sep09 import (  # noqa: E402
    actual_rows,
    simulate_stateful,
)
from scripts.backtest_regime_aware_guardrail_full_walkforward import (  # noqa: E402
    add_variants,
    case_summary,
    monthly_summary,
    prior_year_history,
)
from scripts.build_direct_uplift_floor_candidates import add_floor_reference  # noqa: E402
from scripts.evaluate_three_model_august_holdout import (  # noqa: E402
    add_causal_freshness_priors,
    add_reconciliation,
)
from src.experiments_v2.direct_alpha_allocation import (  # noqa: E402
    DirectAlphaAllocationConfig,
    build_selected_direct_plan,
)


KEYS = ["date", "bakery_id", "product_id"]
DAY_KEYS = ["date", "bakery_id"]
PANEL = ROOT / ".codex_tmp/historical_clickhouse_panel/causal_panel.parquet"
UNIVERSE = ROOT / "reports/historical_daily_retraining_gate_loss075_full_20260910/predictions.parquet"
LABELS = ROOT / "reports/weighted_lost_demand_labels_20260914/floor_history.parquet"
MAPPING = ROOT / "reports/markup_price_mapping_20260826/mapped_products.csv"
FEATURE_CACHE = ROOT / ".codex_tmp/prod_direct_architecture_guardrails/features_may_aug.parquet"
OUTPUT = ROOT / "reports/prod_direct_architecture_guardrails_20260915"
FOLDS = {
    "2026-06": (pd.Timestamp("2026-05-01"), pd.Timestamp("2026-05-31")),
    "2026-07": (pd.Timestamp("2026-05-22"), pd.Timestamp("2026-06-30")),
    "2026-08": (pd.Timestamp("2026-06-22"), pd.Timestamp("2026-07-31")),
}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = pd.read_parquet(PANEL)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel = panel[panel["date"].between("2025-02-01", "2026-08-31")].copy()

    universe = pd.read_parquet(UNIVERSE)
    universe["date"] = pd.to_datetime(universe["date"]).dt.normalize()
    universe = universe[universe["date"].between("2026-05-01", "2026-08-31")].copy()
    universe = universe.rename(
        columns={"category_name": "category", "direct_plan": "incumbent_sku_forecast"}
    )

    labels = pd.read_parquet(LABELS)
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    labels = labels[labels["date"].between("2026-05-01", "2026-08-31")].copy()
    for frame in (panel, universe, labels):
        frame["product_id"] = pd.to_numeric(frame["product_id"], errors="coerce").astype("int64")
    return panel, universe, labels


def build_features(panel: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    direct_replay.FEATURE_CACHE = FEATURE_CACHE
    return direct_replay.get_features(panel, universe)


def causal_p50_factors(
    universe: pd.DataFrame, train_start: pd.Timestamp, train_end: pd.Timestamp
) -> tuple[pd.Series, float]:
    daily = universe.loc[
        universe["date"].between(train_start, train_end),
        ["date", "bakery_id", "direct_total", "p50_total"],
    ].drop_duplicates(["date", "bakery_id"])
    daily["factor"] = daily["p50_total"] / daily["direct_total"].replace(0.0, np.nan)
    daily = daily.replace([np.inf, -np.inf], np.nan).dropna(subset=["factor"])
    daily = daily[daily["factor"].between(0.5, 2.0)]
    factors = daily.groupby("bakery_id")["factor"].median()
    return factors, float(daily["factor"].median())


def train_fold_plan(
    enriched: pd.DataFrame,
    universe: pd.DataFrame,
    old_labels: pd.DataFrame,
    period: str,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, object]]:
    test_start = pd.Period(period).start_time
    test_end = pd.Period(period).end_time.normalize()
    train = enriched[enriched["date"].between(train_start, train_end)].copy()
    test = enriched[enriched["date"].between(test_start, test_end)].copy()
    base = direct_replay.train_base(train, "actual_sold")
    classifier, severity = direct_replay.train_old_loss(train)

    rows = test.copy()
    rows["direct_raw_demand"] = np.maximum(base.predict(rows[direct_replay.FEATURES]), 1e-9)
    groups = [rows[key] for key in DAY_KEYS]
    raw_total = rows["direct_raw_demand"].groupby(groups).transform("sum")
    bakery_total = rows["incumbent_sku_forecast"].groupby(groups).transform("sum")
    rows["direct_forecast"] = rows["direct_raw_demand"] / raw_total.replace(0.0, np.nan) * bakery_total

    factors, fallback = causal_p50_factors(universe, train_start, train_end)
    rows["p50_factor"] = rows["bakery_id"].map(factors).fillna(fallback)
    rows["direct_p50"] = rows["direct_forecast"] * rows["p50_factor"]
    rows["predicted_stockout_probability"] = classifier.predict_proba(
        rows[direct_replay.FEATURES]
    )[:, 1]
    rows["predicted_lost_if_stockout"] = np.expm1(
        severity.predict(rows[direct_replay.FEATURES])
    ).clip(min=0.0)
    rows["predictive_uplift"] = (
        rows["predicted_stockout_probability"] * rows["predicted_lost_if_stockout"]
    )
    rows["loss_scale"] = 1.0

    frozen_floor_history = old_labels[old_labels["date"].le(train_end)].copy()
    rows = add_floor_reference(rows, frozen_floor_history)
    selected = build_selected_direct_plan(rows, DirectAlphaAllocationConfig())
    plan = selected[KEYS + ["selected_sku_forecast"]].rename(
        columns={"selected_sku_forecast": "forecast_qty"}
    )
    metadata = {
        "period": period,
        "train_start": str(train_start.date()),
        "train_end": str(train_end.date()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "p50_bakeries": int(len(factors)),
        "p50_fallback": fallback,
        "plan_qty": float(plan["forecast_qty"].sum()),
    }
    return plan, metadata


def economics_rows(panel: pd.DataFrame, universe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    columns = KEYS + [
        "demand",
        "observed_sales_qty",
        "observed_sales_amount",
    ]
    rows = universe[universe["date"].between("2026-06-01", "2026-08-31")][columns].copy()
    flow_columns = KEYS + [
        "release_qty",
        "incoming_move_qty",
        "outgoing_move_qty",
        "written_off_qty",
    ]
    rows = rows.merge(
        panel[panel["date"].between("2026-06-01", "2026-08-31")][flow_columns],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    numeric = [column for column in rows.columns if column not in KEYS]
    rows[numeric] = rows[numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)

    mapping = pd.read_csv(MAPPING, encoding="utf-8-sig")
    mapping = mapping[mapping["valid_economics"].astype(bool)].copy()
    mapping["product_id"] = pd.to_numeric(mapping["product_id"], errors="coerce")
    mapping = mapping.dropna(subset=["product_id"])
    mapping["product_id"] = mapping["product_id"].astype(int)
    mapping = mapping.sort_values("unit_price").drop_duplicates("product_id", keep="last")
    source_rows = len(rows)
    source_demand = float(rows["demand"].sum())
    rows = rows.merge(
        mapping[["product_id", "unit_cost", "unit_price"]],
        on="product_id",
        how="inner",
        validate="many_to_one",
    )
    coverage = {
        "source_rows": float(source_rows),
        "priced_rows": float(len(rows)),
        "priced_rows_pct": 100.0 * len(rows) / source_rows,
        "priced_demand_pct": 100.0 * float(rows["demand"].sum()) / source_demand,
    }
    rows["source_period"] = rows["date"].dt.to_period("M").astype(str)
    return rows.sort_values(KEYS).reset_index(drop=True), coverage


def refresh_seasonal_variants(
    work: pd.DataFrame,
    demand_history: pd.DataFrame,
    prior_history: pd.DataFrame,
) -> pd.DataFrame:
    """Rebuild only seasonal columns on already-computed causal fixed guards."""
    seasonal_columns = [
        "prior_year_seasonal_ratio",
        "seasonal_weight",
        "effective_seasonal_ratio",
        "prior_year_pair_days",
        "current_growth_ratio",
        "current_growth_comparable",
        "regime_confirmed",
        "seasonal_upper",
        "seasonal_guard",
        "regime_upper",
        "regime_aware_guard",
    ]
    work = work.drop(columns=[column for column in seasonal_columns if column in work], errors="ignore")
    early = work[work["date"].dt.day.le(seasonal_helpers.SEASONAL_DECAY_DAYS)].copy()
    seasonal = seasonal_helpers.build_seasonal_priors(early, prior_history)
    current = seasonal_helpers.build_current_growth(early, demand_history)
    signals = seasonal.merge(current, on=KEYS, validate="one_to_one")
    work = work.merge(signals, on=KEYS, how="left", validate="one_to_one")
    defaults = {
        "prior_year_seasonal_ratio": 1.0,
        "seasonal_weight": 0.0,
        "effective_seasonal_ratio": 1.0,
        "prior_year_pair_days": 0.0,
        "current_growth_ratio": 1.0,
        "current_growth_comparable": False,
    }
    for column, value in defaults.items():
        work[column] = work[column].fillna(value)
    anchor = work["plain_anchor_28"].fillna(work["forecast_qty"])
    seasonal_anchor = anchor * work["effective_seasonal_ratio"]
    work["seasonal_upper"] = seasonal_anchor + np.maximum(0.15 * seasonal_anchor, 2.0)
    work["seasonal_guard"] = work["floor_only"].clip(upper=work["seasonal_upper"])
    work["regime_confirmed"] = (
        work["prior_year_seasonal_ratio"].ge(seasonal_helpers.SEASONAL_MIN_RATIO)
        & work["current_growth_comparable"].astype(bool)
        & work["current_growth_ratio"].ge(seasonal_helpers.CURRENT_MIN_RATIO)
    )
    work["regime_upper"] = work["plain_fixed_upper_28"].fillna(work["forecast_qty"])
    confirmed = work["regime_confirmed"]
    work.loc[confirmed, "regime_upper"] = work.loc[confirmed, "seasonal_upper"]
    work["regime_aware_guard"] = work["floor_only"].clip(upper=work["regime_upper"])
    return work


def main() -> None:
    panel, universe, labels = load_inputs()
    demand_history = universe[KEYS + ["demand"]].copy()
    demand_history["dow"] = demand_history["date"].dt.dayofweek
    demand_history["is_weekend"] = demand_history["dow"].ge(5)
    seasonal_helpers.TARGET_MONTHS = (6, 7, 8)
    cached_rows = OUTPUT / "decision_rows.parquet"
    if cached_rows.exists() and (OUTPUT / "direct_plans.parquet").exists():
        print("Reusing trained Direct plans and fixed guard rows...", flush=True)
        rows = pd.read_parquet(cached_rows)
        plans = pd.read_parquet(OUTPUT / "direct_plans.parquet")
        existing = json.loads((OUTPUT / "metadata.json").read_text(encoding="utf-8"))
        fold_metadata = existing["folds"]
        coverage = existing["coverage"]
        prior_history = prior_year_history(panel, rows)
        rows = refresh_seasonal_variants(rows, demand_history, prior_history)
    else:
        features = build_features(panel, universe)
        features["date"] = pd.to_datetime(features["date"]).dt.normalize()
        old_labels, _ = direct_replay.prepare_labels(labels)
        enriched = features.merge(
            old_labels[KEYS + ["is_clear_stockout", "imputed_demand"]],
            on=KEYS,
            how="left",
            validate="one_to_one",
        )
        enriched["is_clear_stockout"] = enriched["is_clear_stockout"].fillna(False)
        enriched["imputed_demand"] = enriched["imputed_demand"].fillna(0.0)
        plans = []
        fold_metadata = []
        for period, (train_start, train_end) in FOLDS.items():
            plan, metadata = train_fold_plan(
                enriched, universe, old_labels, period, train_start, train_end
            )
            plans.append(plan)
            fold_metadata.append(metadata)
            print(f"Completed Direct replay {period}: {metadata}", flush=True)
        plans = pd.concat(plans, ignore_index=True)
        rows, coverage = economics_rows(panel, universe)
        rows = rows.merge(plans, on=KEYS, how="inner", validate="one_to_one")
        rows = add_causal_freshness_priors(rows)
        rows = add_reconciliation(rows)
        prior_history = prior_year_history(panel, rows)
        rows = add_variants(rows, demand_history, prior_history)

    simulations: dict[str, pd.DataFrame] = {"actual_state": actual_rows(rows)}
    columns = {
        "direct": "forecast_qty",
        "floor_only": "floor_only",
        "symmetric_guard_28": "symmetric_guard_28",
        "seasonal_guard": "seasonal_guard",
        "regime_aware_guard": "regime_aware_guard",
    }
    for variant, column in columns.items():
        print(f"Simulating {variant}...", flush=True)
        simulations[variant] = simulate_stateful(rows, column)

    monthly = monthly_summary(simulations)
    cases = case_summary(simulations)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUTPUT / "monthly_summary.csv", index=False, encoding="utf-8-sig")
    cases.to_csv(OUTPUT / "case_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([coverage]).to_csv(OUTPUT / "coverage.csv", index=False, encoding="utf-8-sig")
    plans.to_parquet(OUTPUT / "direct_plans.parquet", index=False)
    rows.to_parquet(OUTPUT / "decision_rows.parquet", index=False)
    (OUTPUT / "metadata.json").write_text(
        json.dumps(
            {
                "production_write": False,
                "method": "monthly causal retraining of the production Direct SKU architecture",
                "artifact_backcast": False,
                "folds": fold_metadata,
                "coverage": coverage,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nMONTHLY", flush=True)
    print(monthly.to_string(index=False), flush=True)
    print("\nCASES", flush=True)
    print(cases.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
