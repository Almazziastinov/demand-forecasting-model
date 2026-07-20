"""Detect pre-day signals of stockout regime shifts for selected SKUs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "reports" / "sku_share_calibration" / "sku_day_share_comparison.csv"
)
DEFAULT_DIAGNOSIS = (
    ROOT / "reports" / "sku_share_calibration" / "sku_calibration_diagnosis.csv"
)
DEFAULT_OUTPUT = ROOT / "reports" / "stockout_regime_signals"


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    work = work.sort_values(["bakery_id", "product_id", "date"]).reset_index(drop=True)
    work["is_stockout"] = work["stockout_group"].eq("clear_stockout").astype(int)
    group = work.groupby(["bakery_id", "product_id"], sort=False)
    for window in [3, 7, 14]:
        work[f"prior_sold_mean_{window}"] = group["daily_sold"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        work[f"prior_stockout_rate_{window}"] = group["is_stockout"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        work[f"prior_observed_share_mean_{window}"] = group["observed_share"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
    work["forecast_vs_prior_sold_7"] = work["forecast_qty"] / work[
        "prior_sold_mean_7"
    ].replace(0.0, np.nan)
    work["forecast_share_vs_prior_share_7"] = work["forecast_share"] / work[
        "prior_observed_share_mean_7"
    ].replace(0.0, np.nan)
    work["forecast_share_change_vs_7"] = (
        work["forecast_share"] - work["prior_observed_share_mean_7"]
    )
    return work


def univariate_signals(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        sample = frame[[feature, "is_stockout"]].dropna()
        if sample["is_stockout"].nunique() < 2:
            continue
        auc = roc_auc_score(sample["is_stockout"], sample[feature])
        direction = "higher_in_stockout" if auc >= 0.5 else "lower_in_stockout"
        rows.append(
            {
                "feature": feature,
                "rows": len(sample),
                "stockout_median": sample.loc[
                    sample["is_stockout"].eq(1), feature
                ].median(),
                "normal_median": sample.loc[
                    sample["is_stockout"].eq(0), feature
                ].median(),
                "auc_oriented": max(auc, 1.0 - auc),
                "direction": direction,
            }
        )
    return pd.DataFrame(rows).sort_values("auc_oriented", ascending=False)


def fit_temporal_model(
    frame: pd.DataFrame, numeric: list[str], categorical: list[str]
) -> tuple[dict[str, float | int], pd.DataFrame]:
    train = frame[frame["date"] < pd.Timestamp("2026-07-01")].copy()
    test = frame[frame["date"] >= pd.Timestamp("2026-07-01")].copy()
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    model = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(max_iter=2000, class_weight="balanced", C=0.3),
            ),
        ]
    )
    model.fit(train[numeric + categorical], train["is_stockout"])
    probability = model.predict_proba(test[numeric + categorical])[:, 1]
    baseline = float(train["is_stockout"].mean())
    metrics = {
        "train_rows": len(train),
        "test_rows": len(test),
        "test_stockouts": int(test["is_stockout"].sum()),
        "test_stockout_rate": float(test["is_stockout"].mean()),
        "roc_auc": float(roc_auc_score(test["is_stockout"], probability)),
        "average_precision": float(
            average_precision_score(test["is_stockout"], probability)
        ),
        "train_prevalence_baseline": baseline,
    }
    names = model.named_steps["preprocess"].get_feature_names_out()
    coefficients = model.named_steps["model"].coef_[0]
    importance = pd.DataFrame({"feature": names, "coefficient": coefficients})
    importance["abs_coefficient"] = importance["coefficient"].abs()
    importance = importance.sort_values("abs_coefficient", ascending=False)
    return metrics, importance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze pre-day stockout regime signals"
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--diagnosis", default=str(DEFAULT_DIAGNOSIS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input, encoding="utf-8-sig")
    diagnosis = pd.read_csv(args.diagnosis, encoding="utf-8-sig")
    regime_ids = diagnosis.loc[
        diagnosis["calibration_diagnosis"].eq("regime_shift_not_captured"), "product_id"
    ]
    frame = frame[frame["product_id"].isin(regime_ids)].copy()
    features = build_features(frame)
    numeric = [
        "forecast_qty",
        "forecast_share",
        "bakery_forecast_qty",
        "prior_sold_mean_3",
        "prior_sold_mean_7",
        "prior_sold_mean_14",
        "prior_stockout_rate_3",
        "prior_stockout_rate_7",
        "prior_stockout_rate_14",
        "prior_observed_share_mean_7",
        "forecast_vs_prior_sold_7",
        "forecast_share_vs_prior_share_7",
        "forecast_share_change_vs_7",
    ]
    categorical = ["bakery_id", "product_id", "dow"]
    signals = univariate_signals(features, numeric)
    metrics, importance = fit_temporal_model(features, numeric, categorical)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    features.to_csv(
        output / "regime_feature_rows.csv", index=False, encoding="utf-8-sig"
    )
    signals.to_csv(output / "univariate_signals.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(
        output / "logistic_coefficients.csv", index=False, encoding="utf-8-sig"
    )
    payload = {
        "sku_count": int(features["product_id"].nunique()),
        "rows": int(len(features)),
        "stockouts": int(features["is_stockout"].sum()),
        "temporal_test": metrics,
        "top_univariate": signals.head(10).to_dict(orient="records"),
    }
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
