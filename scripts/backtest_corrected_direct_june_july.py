"""Causal monthly backtest of corrected-demand Direct for June and July 2026."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.backtest_direct_bakery_sku_allocation import (  # noqa: E402
    CATEGORICAL,
    FEATURES,
)
from scripts.simulate_two_day_economics import simulate_group  # noqa: E402
from src.experiments_v2.direct_alpha_allocation import (  # noqa: E402
    DirectAlphaAllocationConfig,
    build_selected_direct_plan,
)


PANEL = ROOT / ".codex_tmp/historical_clickhouse_panel/causal_panel.parquet"
UNIVERSE = ROOT / "reports/historical_daily_retraining_gate_loss075_full_20260910/predictions.parquet"
LABELS = ROOT / "reports/weighted_lost_demand_labels_20260914/floor_history.parquet"
MAPPING = ROOT / "reports/markup_price_mapping_20260826/mapped_products.csv"
FEATURE_CACHE = ROOT / ".codex_tmp/corrected_direct_june_july/features.parquet"
OUTPUT = ROOT / "reports/corrected_direct_june_july_20260914"
KEYS = ["date", "bakery_id", "product_id"]
DAY = ["date", "bakery_id"]
CONTINUATION_GATE = 0.30
FOLDS = {
    "2026-06": (pd.Timestamp("2026-05-01"), pd.Timestamp("2026-05-31")),
    "2026-07": (pd.Timestamp("2026-05-22"), pd.Timestamp("2026-06-30")),
}
GUARDED_CONFIGS = [
    (0.10, 0.95),
    (0.25, 0.95),
    (0.50, 0.95),
    (0.10, 0.90),
    (0.25, 0.90),
    (0.50, 0.90),
]
ADDITIVE_BUDGETS = (0.005, 0.01, 0.02)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = pd.read_parquet(PANEL)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel = panel[panel["date"].between("2026-03-01", "2026-07-31")].copy()
    universe = pd.read_parquet(UNIVERSE)
    universe["date"] = pd.to_datetime(universe["date"]).dt.normalize()
    universe = universe[universe["date"].between("2026-05-01", "2026-07-31")].copy()
    universe = universe.rename(
        columns={"category_name": "category", "direct_plan": "incumbent_sku_forecast"}
    )
    labels = pd.read_parquet(LABELS)
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    labels = labels[labels["date"].between("2026-05-01", "2026-07-31")].copy()
    for frame in (panel, universe, labels):
        frame["product_id"] = pd.to_numeric(frame["product_id"]).astype("int64")
    labels = labels[labels["bakery_id"].isin(universe["bakery_id"].unique())].copy()
    return panel, universe, labels


def get_features(panel: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    if FEATURE_CACHE.exists():
        return pd.read_parquet(FEATURE_CACHE)
    features = universe[KEYS + ["category", "incumbent_sku_forecast"]].copy()
    dates = pd.date_range(panel["date"].min(), features["date"].max())
    pairs = pd.MultiIndex.from_frame(
        pd.concat(
            [panel[["bakery_id", "product_id"]], features[["bakery_id", "product_id"]]],
            ignore_index=True,
        ).drop_duplicates()
    )
    sales = (
        panel.groupby(KEYS)["observed_sales_qty"]
        .sum()
        .unstack("date", fill_value=0.0)
        .reindex(index=pairs, columns=dates, fill_value=0.0)
        .fillna(0.0)
        .to_numpy(dtype="float64")
    )
    cumulative = np.concatenate(
        [np.zeros((sales.shape[0], 1)), np.cumsum(sales, axis=1)], axis=1
    )
    presence_cumulative = np.concatenate(
        [np.zeros((sales.shape[0], 1)), np.cumsum(sales > 0.0, axis=1)], axis=1
    )
    row_pairs = pd.MultiIndex.from_frame(features[["bakery_id", "product_id"]])
    pair_code = pairs.get_indexer(row_pairs)
    date_code = dates.get_indexer(features["date"])

    def trailing_sum(days: int) -> np.ndarray:
        left = np.maximum(date_code - days, 0)
        return cumulative[pair_code, date_code] - cumulative[pair_code, left]

    recent = trailing_sum(7)
    prior = trailing_sum(14) - recent
    broad = trailing_sum(56)
    presence = (
        presence_cumulative[pair_code, date_code]
        - presence_cumulative[pair_code, np.maximum(date_code - 28, 0)]
    )
    weekday = np.zeros(len(features), dtype="float64")
    for lag in (7, 14, 21, 28):
        valid = date_code >= lag
        weekday[valid] += sales[pair_code[valid], date_code[valid] - lag]

    features["recent_7_mean"] = recent / 7.0
    features["prior_7_mean"] = prior / 7.0
    features["broad_56_mean"] = broad / 56.0
    features["same_weekday_4_mean"] = weekday / 4.0
    features["presence_28"] = presence / 28.0
    groups = [features["date"], features["bakery_id"]]

    def normalized(values: np.ndarray) -> pd.Series:
        series = pd.Series(values, index=features.index)
        total = series.groupby(groups).transform("sum")
        return (series / total.replace(0.0, np.nan)).fillna(0.0)

    features["recent_7_share"] = normalized(recent)
    features["broad_56_share"] = normalized(broad)
    features["same_weekday_4_share"] = normalized(weekday)
    broad_series = pd.Series(broad, index=features.index)
    category_total = broad_series.groupby(
        [features["date"], features["bakery_id"], features["category"]]
    ).transform("sum")
    bakery_total = broad_series.groupby(groups).transform("sum")
    features["historical_category_share"] = (
        category_total / bakery_total.replace(0.0, np.nan)
    ).fillna(0.0)
    features["recent_trend"] = ((recent + 1.0) / (prior + 1.0)).clip(0.25, 4.0)
    forecast_total = features["incumbent_sku_forecast"].groupby(groups).transform("sum")
    features["log_bakery_total"] = np.log1p(forecast_total)
    features["dow"] = features["date"].dt.dayofweek
    features["actual_sold"] = sales[pair_code, date_code]
    for source, target in [
        ("bakery_id", "bakery_code"),
        ("product_id", "product_code"),
        ("category", "category_code"),
    ]:
        features[target] = pd.Categorical(features[source]).codes
    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURE_CACHE, index=False)
    return features


def prepare_labels(labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    old = labels.copy()
    old["imputed_demand"] = old["original_imputed_demand"].fillna(0.0).clip(lower=0.0)
    old["demand_point_estimate"] = old["demand_lower_bound"] + old["imputed_demand"]
    corrected = labels.copy()
    corrected["economic_lost"] = corrected["hierarchical_imputed_demand"].where(
        corrected["is_clear_stockout"].fillna(False)
        & corrected["continuation_probability"].ge(CONTINUATION_GATE),
        0.0,
    )
    return old, corrected


def train_base(train: pd.DataFrame, target: str) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="poisson",
        n_estimators=240,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=120,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        random_state=42,
        verbosity=-1,
    )
    model.fit(train[FEATURES], train[target], categorical_feature=CATEGORICAL)
    return model


def train_old_loss(train: pd.DataFrame) -> tuple[lgb.LGBMClassifier, lgb.LGBMRegressor]:
    classifier = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=180,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=150,
        reg_lambda=4.0,
        random_state=43,
        verbosity=-1,
    )
    classifier.fit(
        train[FEATURES], train["is_clear_stockout"].astype(int), categorical_feature=CATEGORICAL
    )
    positive = train[train["imputed_demand"].gt(0.0)]
    severity = lgb.LGBMRegressor(
        objective="huber",
        n_estimators=180,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=100,
        reg_lambda=4.0,
        random_state=44,
        verbosity=-1,
    )
    severity.fit(
        positive[FEATURES],
        np.log1p(positive["imputed_demand"]),
        categorical_feature=CATEGORICAL,
    )
    return classifier, severity


def make_plan(
    test: pd.DataFrame,
    base: lgb.LGBMRegressor,
    classifier: lgb.LGBMClassifier,
    severity: lgb.LGBMRegressor,
    *,
    use_uplift: bool = True,
) -> pd.DataFrame:
    rows = test.copy()
    rows["direct_raw_demand"] = np.maximum(base.predict(rows[FEATURES]), 1e-9)
    groups = [rows[key] for key in DAY]
    raw_total = rows["direct_raw_demand"].groupby(groups).transform("sum")
    bakery_total = rows["incumbent_sku_forecast"].groupby(groups).transform("sum")
    rows["direct_forecast"] = rows["direct_raw_demand"] / raw_total * bakery_total
    rows["direct_p50"] = rows["direct_forecast"]
    rows["predicted_stockout_probability"] = classifier.predict_proba(rows[FEATURES])[:, 1]
    rows["predicted_lost_if_stockout"] = np.expm1(severity.predict(rows[FEATURES])).clip(
        min=0.0
    )
    rows["predictive_uplift"] = (
        rows["predicted_stockout_probability"] * rows["predicted_lost_if_stockout"]
    )
    if not use_uplift:
        rows["predictive_uplift"] = 0.0
    rows["loss_scale"] = 1.0
    rows["floor_history_n"] = 0
    rows["floor_demand_p67"] = 0.0
    rows["historical_stockout_rate"] = 0.0
    rows["historical_lost_mean"] = 0.0
    rows["historical_volume"] = 0.0
    result = build_selected_direct_plan(rows, DirectAlphaAllocationConfig())
    return result[KEYS + ["selected_sku_forecast"]]


def economics_input(
    panel: pd.DataFrame, labels: pd.DataFrame, test_dates: pd.DatetimeIndex
) -> pd.DataFrame:
    rows = panel[panel["date"].isin(test_dates)].copy()
    evaluation = labels[KEYS + ["demand_lower_bound", "original_imputed_demand"]].copy()
    rows = rows.merge(evaluation, on=KEYS, how="left", validate="one_to_one")
    rows["demand"] = rows["demand_lower_bound"].fillna(rows["observed_sales_qty"]) + rows[
        "original_imputed_demand"
    ].fillna(0.0)
    rows["opening_stock"] = 0.0
    rows["received"] = rows["incoming_move_qty"].fillna(0.0).clip(lower=0.0)
    rows["sent"] = rows["outgoing_move_qty"].fillna(0.0).clip(lower=0.0)
    rows["produced"] = rows["release_qty"].fillna(0.0).clip(lower=0.0)
    mapping = pd.read_csv(MAPPING, encoding="utf-8-sig")
    mapping = mapping[mapping["valid_economics"].astype(bool)].copy()
    mapping["product_id"] = pd.to_numeric(mapping["product_id"], errors="coerce")
    mapping = mapping.dropna(subset=["product_id"])
    mapping["product_id"] = mapping["product_id"].astype(int)
    mapping = mapping.sort_values("unit_price").drop_duplicates("product_id", keep="last")
    rows = rows.merge(
        mapping[["product_id", "unit_cost", "unit_price"]],
        on="product_id",
        how="inner",
        validate="many_to_one",
    )
    rows["sale_price"] = rows["avg_sales_price"].where(
        rows["avg_sales_price"].gt(0.0), rows["unit_price"]
    )
    return rows


def simulate(rows: pd.DataFrame, plan_column: str, variant: str) -> pd.DataFrame:
    result = pd.concat(
        [
            simulate_group(group, plan_column)
            for _, group in rows.groupby(["bakery_id", "product_id"], sort=False)
        ],
        ignore_index=True,
    )
    prices = rows[KEYS + ["sale_price", "unit_cost"]]
    result = result.merge(prices, on=KEYS, how="left", validate="one_to_one")
    result["revenue"] = result["sold_fresh"] * result["sale_price"] + result[
        "sold_yesterday"
    ] * result["sale_price"] * 0.70
    result["production_cost"] = result["production"] * result["unit_cost"]
    result["gross_profit"] = result["revenue"] - result["production_cost"]
    result["variant"] = variant
    return result


def add_guarded_plan(
    rows: pd.DataFrame, weight: float, protection: float
) -> tuple[pd.DataFrame, str]:
    result = rows.copy()
    groups = [result["date"], result["bakery_id"]]
    current_total = result["current_plan"].groupby(groups).transform("sum")
    corrected_total = result["candidate_plan"].groupby(groups).transform("sum")
    corrected_neutral = result["candidate_plan"] * (
        current_total / corrected_total.replace(0.0, np.nan)
    ).fillna(1.0)
    desired = result["current_plan"] + weight * (
        corrected_neutral - result["current_plan"]
    )
    lower = protection * result["current_plan"]
    protected = np.maximum(desired, lower)
    excess = protected - lower
    excess_total = excess.groupby(groups).transform("sum")
    lower_total = lower.groupby(groups).transform("sum")
    budget = (current_total - lower_total).clip(lower=0.0)
    projected = lower + excess * (budget / excess_total.replace(0.0, np.nan)).fillna(0.0)
    label = f"guard_w{int(weight * 100):02d}_p{int(protection * 100):02d}"
    result[label] = projected.clip(lower=0.0)
    return result, label


def add_additive_plan(rows: pd.DataFrame, budget_share: float) -> tuple[pd.DataFrame, str]:
    result = rows.copy()
    groups = [result["date"], result["bakery_id"]]
    current_total = result["current_plan"].groupby(groups).transform("sum")
    corrected_total = result["candidate_plan"].groupby(groups).transform("sum")
    corrected_neutral = result["candidate_plan"] * (
        current_total / corrected_total.replace(0.0, np.nan)
    ).fillna(1.0)
    positive_residual = (corrected_neutral - result["current_plan"]).clip(lower=0.0)
    residual_total = positive_residual.groupby(groups).transform("sum")
    budget = budget_share * current_total
    scale = (budget / residual_total.replace(0.0, np.nan)).clip(upper=1.0).fillna(0.0)
    label = f"additive_{budget_share * 100:.1f}pct".replace(".", "p")
    result[label] = result["current_plan"] + positive_residual * scale
    return result, label


def add_signal_overlay(
    rows: pd.DataFrame,
    signal: pd.Series,
    budget_share: float,
    prefix: str,
) -> tuple[pd.DataFrame, str]:
    result = rows.copy()
    groups = [result["date"], result["bakery_id"]]
    current_total = result["current_plan"].groupby(groups).transform("sum")
    positive_signal = signal.clip(lower=0.0)
    signal_total = positive_signal.groupby(groups).transform("sum")
    budget = budget_share * current_total
    scale = (budget / signal_total.replace(0.0, np.nan)).clip(upper=1.0).fillna(0.0)
    label = f"{prefix}_{budget_share * 100:.1f}pct".replace(".", "p")
    result[label] = result["current_plan"] + positive_signal * scale
    return result, label


def add_uniform_overlay(
    rows: pd.DataFrame, budget_share: float
) -> tuple[pd.DataFrame, str]:
    result = rows.copy()
    label = f"uniform_{budget_share * 100:.1f}pct".replace(".", "p")
    result[label] = result["current_plan"] * (1.0 + budget_share)
    return result, label


def main() -> None:
    panel, universe, labels = load_inputs()
    features = get_features(panel, universe)
    features["date"] = pd.to_datetime(features["date"]).dt.normalize()
    old_labels, corrected_labels = prepare_labels(labels)
    enriched = features.merge(
        old_labels[KEYS + ["is_clear_stockout", "imputed_demand"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    ).merge(
        corrected_labels[KEYS + ["economic_lost"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    enriched["is_clear_stockout"] = enriched["is_clear_stockout"].fillna(False)
    enriched["imputed_demand"] = enriched["imputed_demand"].fillna(0.0)
    enriched["economic_lost"] = enriched["economic_lost"].fillna(0.0)
    enriched["corrected_demand"] = enriched["actual_sold"] + enriched["economic_lost"]

    simulations = []
    fold_metadata = []
    for period, (train_start, train_end) in FOLDS.items():
        test_dates = pd.date_range(f"{period}-01", periods=1).to_period("M")[0].to_timestamp(
            how="start"
        )
        test_dates = pd.date_range(test_dates, test_dates + pd.offsets.MonthEnd(0))
        train = enriched[enriched["date"].between(train_start, train_end)].copy()
        test = enriched[enriched["date"].isin(test_dates)].copy()
        current_base = train_base(train, "actual_sold")
        corrected_base = train_base(train, "corrected_demand")
        classifier, severity = train_old_loss(train)
        current_plan = make_plan(test, current_base, classifier, severity).rename(
            columns={"selected_sku_forecast": "current_plan"}
        )
        current_no_loss_plan = make_plan(
            test,
            current_base,
            classifier,
            severity,
            use_uplift=False,
        ).rename(columns={"selected_sku_forecast": "current_no_loss_plan"})
        candidate_plan = make_plan(test, corrected_base, classifier, severity).rename(
            columns={"selected_sku_forecast": "candidate_plan"}
        )
        econ = economics_input(panel, labels, test_dates)
        econ = econ.merge(current_plan, on=KEYS, how="inner", validate="one_to_one").merge(
            candidate_plan, on=KEYS, how="inner", validate="one_to_one"
        ).merge(
            current_no_loss_plan, on=KEYS, how="inner", validate="one_to_one"
        )
        simulations.append(simulate(econ, "current_plan", "current_direct"))
        groups = [econ["date"], econ["bakery_id"]]
        current_total = econ["current_plan"].groupby(groups).transform("sum")
        no_loss_total = econ["current_no_loss_plan"].groupby(groups).transform("sum")
        no_loss_neutral = econ["current_no_loss_plan"] * (
            current_total / no_loss_total.replace(0.0, np.nan)
        ).fillna(1.0)
        old_loss_signal = (econ["current_plan"] - no_loss_neutral).clip(lower=0.0)
        for budget_share in ADDITIVE_BUDGETS:
            additive, label = add_additive_plan(econ, budget_share)
            simulations.append(simulate(additive, label, label))
            uniform, uniform_label = add_uniform_overlay(econ, budget_share)
            simulations.append(simulate(uniform, uniform_label, uniform_label))
            old_loss, old_loss_label = add_signal_overlay(
                econ, old_loss_signal, budget_share, "old_loss"
            )
            simulations.append(simulate(old_loss, old_loss_label, old_loss_label))
        fold_metadata.append(
            {
                "period": period,
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "train_rows": len(train),
                "test_rows": len(test),
            }
        )
        print(f"completed {period}: train={len(train):,} test={len(test):,}", flush=True)

    result = pd.concat(simulations, ignore_index=True)
    result["period"] = result["date"].dt.to_period("M").astype(str)
    summary = result.groupby(["period", "variant"], as_index=False).agg(
        demand=("demand", "sum"),
        production=("production", "sum"),
        served=("served", "sum"),
        lost=("lost", "sum"),
        writeoff=("expired_strategy_stock", "sum"),
        revenue=("revenue", "sum"),
        production_cost=("production_cost", "sum"),
        gross_profit=("gross_profit", "sum"),
        lost_cases=("lost", lambda values: int(values.ge(1.0).sum())),
    )
    baseline = summary[summary["variant"].eq("current_direct")][
        ["period", "gross_profit"]
    ].rename(columns={"gross_profit": "current_gp"})
    summary = summary.merge(baseline, on="period", validate="many_to_one")
    summary["gp_delta_vs_current"] = summary["gross_profit"] - summary["current_gp"]
    summary["service_pct"] = 100 * summary["served"] / summary["demand"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT / "economics.parquet", index=False)
    summary.to_csv(OUTPUT / "monthly_summary.csv", index=False, encoding="utf-8-sig")
    (OUTPUT / "metadata.json").write_text(
        json.dumps(
            {
                "production_write": False,
                "continuation_gate": CONTINUATION_GATE,
                "folds": fold_metadata,
                "evaluation_demand": "demand lower bound + original reconstructed lost",
                "floor": "disabled symmetrically to isolate corrected-base effect",
                "volume": "candidate normalized to current Direct per bakery-day",
                "experiment": "positive corrected residual added above current Direct",
                "additive_budgets": list(ADDITIVE_BUDGETS),
                "controls": ["uniform", "old_loss_positive_residual"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
