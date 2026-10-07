"""Estimate demand-continuation probability on inventory-complete days."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports/inventory_stockout_hourly_10/hourly_frame.csv"
OUTPUT = ROOT / "reports/lost_case_probability_20260914"
CUTOFFS = [12, 15, 18]
TRAIN_END = pd.Timestamp("2026-03-21")
TEST_START = pd.Timestamp("2026-03-22")
FEATURES = [
    "cutoff",
    "dow",
    "observed_qty",
    "active_hours",
    "first_sale_hour",
    "last_sale_hour_observed",
    "hours_since_last_sale",
    "sold_last_1h",
    "sold_last_2h",
    "sold_last_3h",
    "bakery_observed_qty",
    "bakery_last_2h",
    "sku_bakery_share",
    "product_prior",
    "bakery_prior",
    "bakery_product_prior",
]


def build_panel(hourly: pd.DataFrame) -> pd.DataFrame:
    hourly = hourly.copy()
    hourly["date"] = pd.to_datetime(hourly["date"])
    for column in [
        "balance_is_consistent",
        "is_inventory_stockout",
        "is_production_observed",
    ]:
        hourly[column] = hourly[column].astype(bool)
    panels = []
    keys = ["date", "bakery_id", "product_id"]
    for cutoff in CUTOFFS:
        observed = hourly[hourly["hour"].le(cutoff)].copy()
        observed["positive_hour"] = observed["hour"].where(observed["sold"].gt(0.0))
        observed["last_1h"] = observed["sold"].where(observed["hour"].eq(cutoff), 0.0)
        observed["last_2h"] = observed["sold"].where(observed["hour"].ge(cutoff - 1), 0.0)
        observed["last_3h"] = observed["sold"].where(observed["hour"].ge(cutoff - 2), 0.0)
        daily = observed.groupby(keys, as_index=False).agg(
            observed_qty=("sold", "sum"),
            active_hours=("sold", lambda value: int(value.gt(0.0).sum())),
            first_sale_hour=("positive_hour", "min"),
            last_sale_hour_observed=("positive_hour", "max"),
            sold_last_1h=("last_1h", "sum"),
            sold_last_2h=("last_2h", "sum"),
            sold_last_3h=("last_3h", "sum"),
            dow=("dow", "first"),
            balance_consistent=("balance_is_consistent", "first"),
            inventory_stockout=("is_inventory_stockout", "first"),
            production_observed=("is_production_observed", "first"),
        )
        future = (
            hourly[hourly["hour"].gt(cutoff)]
            .groupby(keys, as_index=False)["sold"]
            .sum()
            .rename(columns={"sold": "future_qty"})
        )
        daily = daily.merge(future, on=keys, how="left", validate="one_to_one")
        daily["future_qty"] = daily["future_qty"].fillna(0.0)
        daily["future_sale"] = daily["future_qty"].gt(0.0).astype(int)
        bakery = daily.groupby(["date", "bakery_id"], as_index=False).agg(
            bakery_observed_qty=("observed_qty", "sum"),
            bakery_last_2h=("sold_last_2h", "sum"),
        )
        daily = daily.merge(bakery, on=["date", "bakery_id"], validate="many_to_one")
        daily["sku_bakery_share"] = daily["observed_qty"] / daily[
            "bakery_observed_qty"
        ].replace(0.0, np.nan)
        daily["cutoff"] = cutoff
        daily["hours_since_last_sale"] = cutoff - daily["last_sale_hour_observed"]
        panels.append(daily)
    panel = pd.concat(panels, ignore_index=True)
    return panel[panel["observed_qty"].gt(0.0)].copy()


def add_priors(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_rate = train.groupby("cutoff")["future_sale"].mean().rename("global_prior")
    train = train.merge(global_rate, on="cutoff", how="left")
    test = test.merge(global_rate, on="cutoff", how="left")
    for keys, name, strength in [
        (["cutoff", "product_id"], "product_prior", 20.0),
        (["cutoff", "bakery_id"], "bakery_prior", 30.0),
        (["cutoff", "bakery_id", "product_id"], "bakery_product_prior", 10.0),
    ]:
        stats = train.groupby(keys, as_index=False).agg(
            positives=("future_sale", "sum"), observations=("future_sale", "size")
        )
        stats = stats.merge(global_rate, on="cutoff", how="left")
        stats[name] = (stats["positives"] + strength * stats["global_prior"]) / (
            stats["observations"] + strength
        )
        lookup = stats[keys + [name]]
        train = train.merge(lookup, on=keys, how="left", validate="many_to_one")
        test = test.merge(lookup, on=keys, how="left", validate="many_to_one")
        train[name] = train[name].fillna(train["global_prior"])
        test[name] = test[name].fillna(test["global_prior"])
    return train, test


def metric_row(label: str, y_true: pd.Series, probability: np.ndarray) -> dict[str, float | str]:
    prediction = probability >= 0.5
    return {
        "variant": label,
        "base_rate_pct": 100 * float(y_true.mean()),
        "roc_auc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability),
        "brier": brier_score_loss(y_true, probability),
        "log_loss": log_loss(y_true, probability),
        "precision_at_050": precision_score(y_true, prediction, zero_division=0),
        "recall_at_050": recall_score(y_true, prediction, zero_division=0),
        "predicted_positive_pct": 100 * float(prediction.mean()),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel = build_panel(pd.read_csv(INPUT))
    complete = panel[
        panel["balance_consistent"]
        & panel["production_observed"]
        & ~panel["inventory_stockout"]
    ].copy()
    train = complete[complete["date"].le(TRAIN_END)].copy()
    test = complete[complete["date"].ge(TEST_START)].copy()
    train_base = train.copy()
    train, test = add_priors(train, test)
    train[FEATURES] = train[FEATURES].fillna(-1.0)
    test[FEATURES] = test[FEATURES].fillna(-1.0)

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
    probability = model.predict_proba(test[FEATURES])[:, 1]
    test["probability"] = probability
    test["product_prior_probability"] = test["product_prior"]

    rows = [
        metric_row("network_prior", test["future_sale"], test["global_prior"].to_numpy()),
        metric_row("product_prior", test["future_sale"], test["product_prior"].to_numpy()),
        metric_row("continuation_model", test["future_sale"], probability),
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT / "summary.csv", index=False, encoding="utf-8-sig")

    by_cutoff = []
    for cutoff, group in test.groupby("cutoff"):
        for label, column in [
            ("product_prior", "product_prior_probability"),
            ("continuation_model", "probability"),
        ]:
            row = metric_row(label, group["future_sale"], group[column].to_numpy())
            row["cutoff"] = cutoff
            by_cutoff.append(row)
    pd.DataFrame(by_cutoff).to_csv(
        OUTPUT / "by_cutoff.csv", index=False, encoding="utf-8-sig"
    )

    bins = pd.qcut(test["probability"], 10, duplicates="drop")
    calibration = test.assign(probability_bin=bins).groupby(
        "probability_bin", observed=True, as_index=False
    ).agg(
        cases=("future_sale", "size"),
        predicted_probability=("probability", "mean"),
        actual_rate=("future_sale", "mean"),
        future_qty=("future_qty", "sum"),
    )
    calibration.to_csv(OUTPUT / "calibration.csv", index=False, encoding="utf-8-sig")

    candidates = panel[
        panel["date"].ge(TEST_START)
        & panel["balance_consistent"]
        & panel["production_observed"]
        & panel["inventory_stockout"]
        & panel["last_sale_hour_observed"].le(panel["cutoff"])
    ].copy()
    _, candidates = add_priors(train_base, candidates)
    candidates[FEATURES] = candidates[FEATURES].fillna(-1.0)
    candidates["probability"] = model.predict_proba(candidates[FEATURES])[:, 1]
    candidate_summary = candidates.groupby("cutoff", as_index=False).agg(
        candidates=("future_sale", "size"),
        mean_probability=("probability", "mean"),
        median_probability=("probability", "median"),
        recognized_at_050=("probability", lambda value: int(value.ge(0.5).sum())),
        recognized_at_070=("probability", lambda value: int(value.ge(0.7).sum())),
    )
    candidate_summary.to_csv(
        OUTPUT / "stockout_candidate_scores.csv", index=False, encoding="utf-8-sig"
    )
    test.to_parquet(OUTPUT / "test_predictions.parquet", index=False)
    print(summary.to_string(index=False))
    print("\nBy cutoff")
    print(pd.DataFrame(by_cutoff).to_string(index=False))
    print("\nCandidate scores")
    print(candidate_summary.to_string(index=False))


if __name__ == "__main__":
    main()
