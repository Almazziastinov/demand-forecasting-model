"""Test volume-weighted Direct with soft uplift normalization and Core protection."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from backtest_direct_bakery_sku_allocation import CATEGORICAL, DAY, FEATURES, KEYS
from rolling_validate_direct_uplift_floor import FEATURE_CACHE, FOLDS
from select_direct_adaptive_floor import make_plan


ROOT = Path(__file__).resolve().parents[1]
ROLLING = ROOT / "reports/rolling_direct_uplift_floor_20260827/rows.parquet"
OUTPUT = ROOT / "reports/weighted_direct_soft_normalization_20260827"
ALPHAS = (0.0, 0.25, 0.50, 0.75, 1.0)
SOFT_GROUP = ["scenario", "date", "bakery_id"]
FLOOR_PARAMS = {
    "min_n": 8.0,
    "min_stockout_rate": 0.75,
    "min_lost_mean": 4.0,
    "scale": 0.8,
    "unit_cap": 5.0,
    "relative_cap": 0.1,
}


def fit_weighted_direct(
    rows: pd.DataFrame, train_end: str, test_dates: pd.DatetimeIndex
) -> pd.DataFrame:
    train = rows[rows["date"].le(pd.Timestamp(train_end))].copy()
    test = rows[rows["date"].isin(test_dates)].copy()
    sample_weight = np.sqrt(1.0 + train["broad_56_mean"].clip(lower=0.0)).clip(
        upper=10.0
    )
    sample_weight /= sample_weight.mean()
    model = lgb.LGBMRegressor(
        objective="poisson",
        n_estimators=240,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=120,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        random_state=52,
        verbosity=-1,
    )
    model.fit(
        train[FEATURES],
        train["actual_sold"],
        sample_weight=sample_weight,
        categorical_feature=CATEGORICAL,
    )
    test["weighted_direct_raw"] = np.maximum(model.predict(test[FEATURES]), 1e-9)
    denominator = test.groupby(DAY)["weighted_direct_raw"].transform("sum")
    share = test["weighted_direct_raw"] / denominator.replace(0, np.nan)
    bakery_total = test.groupby(DAY)["incumbent_sku_forecast"].transform("sum")
    test["weighted_direct_forecast"] = share.fillna(0.0) * bakery_total
    return test[KEYS + ["weighted_direct_raw", "weighted_direct_forecast"]]


def add_core_flag(rows: pd.DataFrame) -> pd.DataFrame:
    result = rows.copy()
    result["historical_volume"] = result["broad_56_mean"].clip(lower=0.0)
    result = result.sort_values(
        [*SOFT_GROUP, "historical_volume"], ascending=[True, True, True, False]
    )
    total = result.groupby(SOFT_GROUP)["historical_volume"].transform("sum")
    share = result["historical_volume"] / total.replace(0, np.nan)
    cumulative_before = (
        share.groupby([result[key] for key in SOFT_GROUP]).cumsum() - share
    )
    result["is_core_sku"] = cumulative_before.lt(0.70) & share.gt(0)
    return result.sort_index()


def protect_core(
    candidate: pd.Series,
    base: pd.Series,
    target_volume: pd.Series,
    core: pd.Series,
    rows: pd.DataFrame,
) -> pd.Series:
    protected = candidate.where(~core, np.maximum(candidate, base))
    groups = [rows[key] for key in SOFT_GROUP]
    core_value = protected.where(core, 0.0).groupby(groups).transform("sum")
    noncore_value = protected.where(~core, 0.0).groupby(groups).transform("sum")
    noncore_factor = (
        (target_volume - core_value) / noncore_value.replace(0, np.nan)
    ).clip(lower=0.0)
    return protected.where(core, protected * noncore_factor.fillna(0.0))


def add_alpha_candidates(rows: pd.DataFrame) -> pd.DataFrame:
    result = rows.copy()
    uplift = result["loss_scale"] * result["predictive_uplift"]
    groups = [result[key] for key in SOFT_GROUP]
    uplift_total = uplift.groupby(groups).transform("sum")
    for prefix, base_column in [
        ("weighted", "weighted_direct_p50"),
        ("original", "direct_p50"),
    ]:
        base = result[base_column]
        pre_normalized = base + uplift
        p50_volume = result.groupby(SOFT_GROUP)[base_column].transform("sum")
        pre_total = pre_normalized.groupby(groups).transform("sum")
        for alpha in ALPHAS:
            suffix = int(alpha * 100)
            target = p50_volume + alpha * uplift_total
            normalized = pre_normalized / pre_total.replace(0, np.nan) * target
            protected = protect_core(
                normalized.fillna(base), base, target, result["is_core_sku"], result
            )
            column = f"{prefix}_alpha_{suffix:03d}"
            result[column] = protected
            floor_source = result.copy()
            floor_source["direct_uplift_p50"] = result[column]
            result[f"{column}_floor"] = make_plan(floor_source, FLOOR_PARAMS)
    return result


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    evaluation = rows[~rows["rolling_fold"].eq(FOLDS[0]["name"])]
    variants = ["direct_p50", "weighted_direct_p50"] + [
        f"{prefix}_alpha_{int(alpha * 100):03d}{floor}"
        for prefix in ("weighted", "original")
        for alpha in ALPHAS
        for floor in ("", "_floor")
    ]
    records = []
    for scenario, part in evaluation.groupby("scenario"):
        demand = part["scenario_demand"]
        for variant in variants:
            error = part[variant] - demand
            total = part.groupby(DAY)[variant].transform("sum")
            top = (
                (part[variant] / total.replace(0, np.nan))
                .groupby([part["date"], part["bakery_id"]])
                .max()
            )
            sku = part[part["product_id"].eq(1071)]
            sku_error = sku[variant] - sku["scenario_demand"]
            records.append(
                {
                    "scenario": scenario,
                    "variant": variant,
                    "volume": float(part[variant].sum()),
                    "wape_pct": float(100 * error.abs().sum() / demand.sum()),
                    "bias_pct": float(100 * error.sum() / demand.sum()),
                    "surplus": float(error.clip(lower=0).sum()),
                    "underbake": float((-error).clip(lower=0).sum()),
                    "top_share_max": float(top.max()),
                    "bakery_days_ge20": int(top.ge(0.20).sum()),
                    "sku_1071_wape_pct": float(
                        100 * sku_error.abs().sum() / sku["scenario_demand"].sum()
                    ),
                    "sku_1071_bias_pct": float(
                        100 * sku_error.sum() / sku["scenario_demand"].sum()
                    ),
                    "sku_1071_forecast": float(sku[variant].sum()),
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    features = pd.read_parquet(FEATURE_CACHE)
    rolling = pd.read_parquet(ROLLING)
    for frame in (features, rolling):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["product_id"] = pd.to_numeric(frame["product_id"]).astype("int64")
    weighted_parts = []
    for fold in FOLDS:
        prediction = fit_weighted_direct(features, fold["train_end"], fold["dates"])
        prediction["rolling_fold"] = fold["name"]
        weighted_parts.append(prediction)
    weighted = pd.concat(weighted_parts, ignore_index=True)
    rolling = rolling.merge(
        weighted, on=[*KEYS, "rolling_fold"], how="left", validate="many_to_one"
    )
    rolling["weighted_direct_p50"] = (
        rolling["weighted_direct_forecast"] * rolling["p50_factor"]
    )
    rolling["loss_scale"] = rolling["scenario"].map(
        {"conservative": 0.5, "calibrated": 1.0, "upper": 1.5}
    )
    rolling = add_core_flag(rolling)
    rolling = add_alpha_candidates(rolling)
    metrics = summarize(rolling)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rolling.to_parquet(OUTPUT / "rows.parquet", index=False)
    metrics.to_csv(OUTPUT / "metrics.csv", index=False, encoding="utf-8-sig")
    summary = {
        "weight": "sqrt(1 + broad_56_mean), normalized to mean 1, capped at 10",
        "core": "causal top 70% of bakery historical volume; no decrease below weighted Direct P50",
        "alphas": ALPHAS,
        "floor_params": FLOOR_PARAMS,
        "production_write": False,
    }
    (OUTPUT / "design.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        metrics[metrics["scenario"].eq("calibrated")]
        .sort_values(["wape_pct", "underbake"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
