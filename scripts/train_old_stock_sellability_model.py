"""Train a causal model of the share of sales fulfilled by yesterday stock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipelines.forecast_publish.load_forecast_run import create_client  # noqa: E402


SCOPE = ROOT / "reports/comparable_produced_scope_p50_loss075_20260911/predictions.parquet"
POISSON = ROOT / "reports/direct_poisson_allocation_full_20260910/predictions.parquet"
OUTPUT = ROOT / "reports/old_stock_sellability_model_20260912"
CACHE = ROOT / ".codex_tmp/pilot_freshness_daily_20260912.parquet"
KEYS = ["date", "bakery_id", "product_id"]
FEATURES = [
    "plan",
    "direct_total",
    "recent_7_mean",
    "broad_56_mean",
    "same_weekday_4_mean",
    "pair_old_share_prior",
    "product_old_share_56",
    "bakery_old_share_56",
    "network_old_share_56",
    "dow",
    "month",
    "bakery_code",
    "product_code",
    "category_code",
]
CATEGORICAL = ["dow", "month", "bakery_code", "product_code", "category_code"]
OLD_HEX = "D092D187D0B5D180D0B0D188D0BDD0B8D0B9"
FRESH_HEX = "D0A1D0B2D0B5D0B6D0B8D0B9"


def extract_freshness(bakery_ids: tuple[int, ...]) -> pd.DataFrame:
    client = create_client(ROOT / ".env")
    result = client.query_df(
        f"""
        select
            check_date as date,
            toInt64OrZero(toString(bakery_id)) as bakery_id,
            toInt64OrZero(toString(product_id)) as product_id,
            sumIf(toFloat64(quantity), hex(freshness) = '{OLD_HEX}') as old_qty,
            sumIf(toFloat64(quantity), hex(freshness) = '{FRESH_HEX}') as fresh_qty
        from (
            select distinct
                check_datetime,
                check_date,
                bakery_id,
                product_id,
                quantity,
                price,
                line_amount,
                cash_event_type,
                freshness
            from Svezhar.fct_check_lines
            where check_date between toDate('2026-01-01') and toDate('2026-08-31')
              and toInt64OrZero(toString(bakery_id)) in %(bakery_ids)s
              and hex(cash_event_type) = 'D09FD180D0BED0B4D0B0D0B6D0B0'
              and quantity > 0
        )
        group by date, bakery_id, product_id
        """,
        parameters={"bakery_ids": bakery_ids},
    )
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(CACHE, index=False)
    return result


def _rolling_share(
    rows: pd.DataFrame,
    group_columns: list[str],
    output_column: str,
) -> pd.DataFrame:
    daily = rows.groupby(["date", *group_columns], as_index=False)[
        ["old_qty", "label_qty"]
    ].sum()
    dates = pd.DataFrame({"date": pd.date_range(rows["date"].min(), rows["date"].max())})
    if group_columns:
        groups = rows[group_columns].drop_duplicates().assign(_join=1)
        grid = groups.merge(dates.assign(_join=1), on="_join").drop(columns="_join")
    else:
        grid = dates
    daily = grid.merge(daily, on=["date", *group_columns], how="left")
    daily[["old_qty", "label_qty"]] = daily[["old_qty", "label_qty"]].fillna(0.0)
    daily = daily.sort_values([*group_columns, "date"])
    if group_columns:
        grouped = daily.groupby(group_columns, sort=False)
        prior_old = grouped["old_qty"].transform(
            lambda value: value.shift(1).rolling(56, min_periods=7).sum()
        )
        prior_total = grouped["label_qty"].transform(
            lambda value: value.shift(1).rolling(56, min_periods=7).sum()
        )
    else:
        prior_old = daily["old_qty"].shift(1).rolling(56, min_periods=7).sum()
        prior_total = daily["label_qty"].shift(1).rolling(56, min_periods=7).sum()
    daily[output_column] = ((prior_old + 5.0) / (prior_total + 100.0)).clip(0.0, 0.5)
    return daily[["date", *group_columns, output_column]]


def build_training_frame(freshness: pd.DataFrame) -> pd.DataFrame:
    rows = pd.read_parquet(
        SCOPE,
        columns=KEYS
        + ["category_name", "direct_total", "s7", "s56", "sw"],
    )
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    poisson = pd.read_parquet(POISSON, columns=KEYS + ["direct_raw_demand"])
    poisson["date"] = pd.to_datetime(poisson["date"]).dt.normalize()
    rows = rows.merge(poisson, on=KEYS, how="left", validate="one_to_one")
    raw = rows["direct_raw_demand"].fillna(0.0).clip(lower=0.0)
    groups = [rows["date"], rows["bakery_id"]]
    raw_total = raw.groupby(groups).transform("sum")
    fallback = 1.0 / rows.groupby(["date", "bakery_id"])["product_id"].transform("count")
    rows["plan"] = rows["direct_total"] * (raw / raw_total.replace(0.0, np.nan)).fillna(fallback)
    rows = rows.merge(freshness, on=KEYS, how="left", validate="one_to_one")
    rows[["old_qty", "fresh_qty"]] = rows[["old_qty", "fresh_qty"]].fillna(0.0)
    rows["label_qty"] = rows["old_qty"] + rows["fresh_qty"]
    rows["old_share"] = rows["old_qty"] / rows["label_qty"].replace(0.0, np.nan)

    ordered = rows.sort_values(["bakery_id", "product_id", "date"])
    grouped = ordered.groupby(["bakery_id", "product_id"], sort=False)
    prior_old = grouped["old_qty"].cumsum() - ordered["old_qty"]
    prior_total = grouped["label_qty"].cumsum() - ordered["label_qty"]
    ordered["pair_old_share_prior"] = ((prior_old + 5.0) / (prior_total + 100.0)).clip(0.0, 0.5)
    rows = ordered.sort_index()

    for group_columns, column in [
        (["product_id"], "product_old_share_56"),
        (["bakery_id"], "bakery_old_share_56"),
        ([], "network_old_share_56"),
    ]:
        rows = rows.merge(
            _rolling_share(rows, group_columns, column),
            on=["date", *group_columns],
            how="left",
            validate="many_to_one",
        )
    for column in [
        "pair_old_share_prior",
        "product_old_share_56",
        "bakery_old_share_56",
        "network_old_share_56",
    ]:
        rows[column] = rows[column].fillna(0.05).clip(0.0, 0.5)
    rows["recent_7_mean"] = rows["s7"] / 7.0
    rows["broad_56_mean"] = rows["s56"] / 56.0
    rows["same_weekday_4_mean"] = rows["sw"] / 4.0
    rows["dow"] = rows["date"].dt.dayofweek.astype("category")
    rows["month"] = rows["date"].dt.month.astype("category")
    for source, target in [
        ("bakery_id", "bakery_code"),
        ("product_id", "product_code"),
        ("category_name", "category_code"),
    ]:
        rows[target] = pd.Categorical(rows[source]).codes.astype("int32")
        rows[target] = rows[target].astype("category")
    return rows


def weighted_mae(actual: pd.Series, predicted: np.ndarray, weight: pd.Series) -> float:
    return float(np.average(np.abs(actual.to_numpy() - predicted), weights=weight.to_numpy()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    scope = pd.read_parquet(SCOPE, columns=["bakery_id"])
    bakery_ids = tuple(sorted(scope["bakery_id"].astype(int).unique()))
    freshness = extract_freshness(bakery_ids) if args.refresh or not CACHE.exists() else pd.read_parquet(CACHE)
    rows = build_training_frame(freshness)
    eligible = rows[rows["label_qty"].gt(0.0) & rows["old_share"].notna()].copy()
    train = eligible[eligible["date"].le("2026-06-30")]
    validation = eligible[eligible["date"].between("2026-07-01", "2026-07-31")]
    test = eligible[eligible["date"].between("2026-08-01", "2026-08-31")]
    validation_all = rows[
        rows["date"].between("2026-07-01", "2026-07-31")
    ].copy()
    test_all = rows[rows["date"].between("2026-08-01", "2026-08-31")].copy()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    test_predictions = test_all[
        KEYS
        + [
            "old_share",
            "label_qty",
            "old_qty",
            "plan",
            "product_old_share_56",
        ]
    ].copy()
    validation_predictions = validation_all[
        KEYS
        + [
            "old_share",
            "label_qty",
            "old_qty",
            "plan",
            "product_old_share_56",
        ]
    ].copy()

    baselines = {
        "product_prior": test["product_old_share_56"].to_numpy(),
        "product_prior_x2_5": (test["product_old_share_56"] * 2.5).clip(0.0, 1.0).to_numpy(),
    }
    for label, prediction in baselines.items():
        test_predictions[label] = (
            test_all["product_old_share_56"]
            if label == "product_prior"
            else (test_all["product_old_share_56"] * 2.5).clip(0.0, 1.0)
        ).to_numpy()
        validation_predictions[label] = (
            validation_all["product_old_share_56"]
            if label == "product_prior"
            else (validation_all["product_old_share_56"] * 2.5).clip(0.0, 1.0)
        ).to_numpy()
        records.append(
            {
                "variant": label,
                "split": "test",
                "weighted_share_mae": weighted_mae(test["old_share"], prediction, test["label_qty"]),
                "old_qty_mae": float(np.mean(np.abs(test["old_qty"].to_numpy() - test["plan"].to_numpy() * prediction))),
            }
        )

    for alpha in [0.50, 0.75, 0.90]:
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=300,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=120,
            reg_lambda=4.0,
            random_state=42,
            verbosity=-1,
        )
        model.fit(
            train[FEATURES],
            train["old_share"],
            sample_weight=train["label_qty"].clip(upper=100.0),
            categorical_feature=CATEGORICAL,
        )
        for split_name, frame in [("validation", validation), ("test", test)]:
            prediction = np.clip(model.predict(frame[FEATURES]), 0.0, 1.0)
            records.append(
                {
                    "variant": f"quantile_{alpha:.2f}",
                    "split": split_name,
                    "weighted_share_mae": weighted_mae(frame["old_share"], prediction, frame["label_qty"]),
                    "old_qty_mae": float(np.mean(np.abs(frame["old_qty"].to_numpy() - frame["plan"].to_numpy() * prediction))),
                }
            )
            if split_name == "test":
                test_predictions[f"quantile_{alpha:.2f}"] = np.clip(
                    model.predict(test_all[FEATURES]), 0.0, 1.0
                )
            elif split_name == "validation":
                validation_predictions[f"quantile_{alpha:.2f}"] = np.clip(
                    model.predict(validation_all[FEATURES]), 0.0, 1.0
                )
        joblib.dump(model, OUTPUT / f"old_share_quantile_{alpha:.2f}.joblib")

    metrics = pd.DataFrame(records)
    metrics.to_csv(OUTPUT / "metrics.csv", index=False, encoding="utf-8-sig")
    validation_predictions.to_parquet(OUTPUT / "july_predictions.parquet", index=False)
    test_predictions.to_parquet(OUTPUT / "august_predictions.parquet", index=False)
    metadata = {
        "train_end": "2026-06-30",
        "validation": ["2026-07-01", "2026-07-31"],
        "test": ["2026-08-01", "2026-08-31"],
        "features": FEATURES,
        "categorical": CATEGORICAL,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "production_write": False,
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
