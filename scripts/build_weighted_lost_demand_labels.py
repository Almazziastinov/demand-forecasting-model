"""Build research-only probability-weighted hierarchical lost-demand labels."""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.calibrate_post_last_sale_demand import build_cases  # noqa: E402
from scripts.experiment_lost_case_probability import (  # noqa: E402
    FEATURES,
    INPUT as INVENTORY_INPUT,
    TRAIN_END,
    add_priors,
    build_panel,
)


SOURCE_LABELS = ROOT / "reports/calibrated_stockout_network_20260826/sku_day_demand.csv"
NETWORK_HOURLY = ROOT / ".codex_tmp/rolling_hourly_sales_20260601_20260823.parquet"
OUTPUT = ROOT / "reports/weighted_lost_demand_labels_20260914"
CUTOFFS = [12, 15, 18]
OBSERVED_BINS = [-np.inf, 5, 15, 40, np.inf]
OBSERVED_LABELS = ["<=5", "6-15", "16-40", ">40"]


def train_continuation_model() -> tuple[lgb.LGBMClassifier, pd.DataFrame]:
    panel = build_panel(pd.read_csv(INVENTORY_INPUT))
    complete = panel[
        panel["balance_consistent"]
        & panel["production_observed"]
        & ~panel["inventory_stockout"]
    ].copy()
    train_base = complete[complete["date"].le(TRAIN_END)].copy()
    train, _ = add_priors(train_base, train_base.iloc[0:0].copy())
    train[FEATURES] = train[FEATURES].fillna(-1.0)
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=24,
        min_child_samples=40,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        random_state=42,
        verbosity=-1,
    )
    model.fit(train[FEATURES], train["future_sale"])
    return model, train_base


def network_candidate_features(hourly: pd.DataFrame) -> pd.DataFrame:
    work = hourly.copy()
    work["date"] = pd.to_datetime(work["date"])
    work["dow"] = work["date"].dt.dayofweek
    work["balance_is_consistent"] = True
    work["is_inventory_stockout"] = False
    work["is_production_observed"] = True
    return build_panel(work)


def choose_cutoff(last_sale: pd.Series) -> pd.Series:
    return np.select(
        [last_sale.le(12), last_sale.le(15)], [12, 15], default=18
    ).astype(int)


def score_probabilities(labels: pd.DataFrame) -> pd.DataFrame:
    model, prior_train = train_continuation_model()
    hourly = pd.read_parquet(NETWORK_HOURLY)
    candidates = network_candidate_features(hourly)
    selected = labels["is_clear_stockout"].fillna(False).astype(bool)
    scoring = labels.loc[
        selected,
        ["date", "bakery_id", "product_id", "last_sale_hour", "demand_lower_bound"],
    ].copy()
    scoring["cutoff"] = choose_cutoff(scoring["last_sale_hour"])
    scoring = scoring.merge(
        candidates,
        on=["date", "bakery_id", "product_id", "cutoff"],
        how="left",
        suffixes=("", "_features"),
        validate="one_to_one",
    )
    scoring["observed_qty"] = scoring["observed_qty"].fillna(
        scoring["demand_lower_bound"]
    )
    scoring["dow"] = scoring["dow"].fillna(scoring["date"].dt.dayofweek)
    _, scoring = add_priors(prior_train, scoring)
    scoring[FEATURES] = scoring[FEATURES].fillna(-1.0)
    scoring["continuation_probability"] = model.predict_proba(scoring[FEATURES])[:, 1]
    result = labels.merge(
        scoring[["date", "bakery_id", "product_id", "continuation_probability"]],
        on=["date", "bakery_id", "product_id"],
        how="left",
        validate="one_to_one",
    )
    result["continuation_probability"] = result["continuation_probability"].fillna(0.0)
    return result


def add_hierarchical_severity(labels: pd.DataFrame) -> pd.DataFrame:
    hourly = pd.read_parquet(NETWORK_HOURLY)
    cases = build_cases(hourly, CUTOFFS)
    cases["observed_band"] = pd.cut(
        cases["observed"], bins=OBSERVED_BINS, labels=OBSERVED_LABELS
    ).astype(str)
    selected = labels["is_clear_stockout"].fillna(False).astype(bool)
    result = labels.copy()
    result["cutoff"] = choose_cutoff(result["last_sale_hour"].fillna(18.0))
    result["observed_band"] = pd.cut(
        result["demand_lower_bound"], bins=OBSERVED_BINS, labels=OBSERVED_LABELS
    ).astype(str)
    result["severity_ratio"] = 1.0

    for date in sorted(result.loc[selected, "date"].unique()):
        date = pd.Timestamp(date)
        calibration = cases[cases["date"].between(date - pd.Timedelta(days=21), date - pd.Timedelta(days=1))]
        global_stats = calibration.groupby("cutoff", as_index=False).agg(
            global_true=("true_hidden", "sum"), global_raw=("raw_prediction", "sum")
        )
        global_stats["global_multiplier"] = global_stats["global_true"] / global_stats["global_raw"]
        grouped = calibration.groupby(
            ["cutoff", "product_id", "observed_band"], as_index=False
        ).agg(group_true=("true_hidden", "sum"), group_raw=("raw_prediction", "sum"))
        grouped = grouped.merge(
            global_stats[["cutoff", "global_multiplier"]], on="cutoff", how="left"
        )
        grouped["group_multiplier"] = (
            grouped["group_true"] + 100.0 * grouped["global_multiplier"]
        ) / (grouped["group_raw"] + 100.0)
        grouped["severity_ratio"] = grouped["group_multiplier"] / grouped["global_multiplier"]
        mask = selected & result["date"].eq(date)
        day = result.loc[mask, ["cutoff", "product_id", "observed_band"]].merge(
            grouped[["cutoff", "product_id", "observed_band", "severity_ratio"]],
            on=["cutoff", "product_id", "observed_band"],
            how="left",
            validate="many_to_one",
        )
        result.loc[mask, "severity_ratio"] = day["severity_ratio"].fillna(1.0).to_numpy()

    result["hierarchical_imputed_demand"] = result["imputed_demand"] * result["severity_ratio"]
    # Preserve each date/cutoff total from the current causal label before the
    # probability weight is applied. Only within-group placement changes here.
    selected_rows = result[selected].copy()
    keys = [selected_rows["date"], selected_rows["cutoff"]]
    old_total = selected_rows["imputed_demand"].groupby(keys).transform("sum")
    new_total = selected_rows["hierarchical_imputed_demand"].groupby(keys).transform("sum")
    scale = (old_total / new_total.replace(0.0, np.nan)).fillna(1.0)
    result.loc[selected, "hierarchical_imputed_demand"] *= scale.to_numpy()
    return result


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(SOURCE_LABELS, encoding="utf-8-sig")
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    labels = score_probabilities(labels)
    labels = add_hierarchical_severity(labels)
    labels["original_imputed_demand"] = labels["imputed_demand"]
    labels["weighted_imputed_demand"] = (
        labels["continuation_probability"] * labels["hierarchical_imputed_demand"]
    )
    labels["imputed_demand"] = labels["weighted_imputed_demand"]
    labels["demand_point_estimate"] = labels["demand_lower_bound"] + labels["imputed_demand"]
    labels.to_parquet(OUTPUT / "floor_history.parquet", index=False)

    selected = labels["is_clear_stockout"].fillna(False).astype(bool)
    summary = pd.DataFrame(
        [
            {
                "rows": len(labels),
                "stockout_candidates": int(selected.sum()),
                "original_imputed": float(labels["original_imputed_demand"].sum()),
                "hierarchical_imputed_before_probability": float(
                    labels["hierarchical_imputed_demand"].sum()
                ),
                "weighted_imputed": float(labels["weighted_imputed_demand"].sum()),
                "weighted_to_original_pct": 100
                * float(labels["weighted_imputed_demand"].sum())
                / float(labels["original_imputed_demand"].sum()),
                "mean_candidate_probability": float(
                    labels.loc[selected, "continuation_probability"].mean()
                ),
            }
        ]
    )
    summary.to_csv(OUTPUT / "summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
