"""Build strict daily walk-forward P50 forecasts for a stable bakery cohort."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment_demand_adjusted_bakery_target import (  # noqa: E402
    rebuild_target_features,
)
from src.experiments_v2.bakery_day_forecast import (  # noqa: E402
    BASE_FEATURES,
    TARGET_COL,
    build_model_frame,
    cast_category_columns,
)
from src.experiments_v2.common import predict_clipped, train_quantile  # noqa: E402


BASE_PATH = ROOT / ".codex_tmp/network_extension_20260826_bakery_daily_extended.csv.gz"
AUGUST_DETAIL = ROOT / ".codex_tmp/pilot_recalc_report/detail.csv"
SEPTEMBER_DETAIL = ROOT / ".codex_tmp/pilot_management_sep01_09/detail.csv"
JULY_TIMES = ROOT / ".codex_tmp/july23_31_sale_times.csv"
AUGUST_TIMES = ROOT / ".codex_tmp/august_sale_times.csv"
SEPTEMBER_TIMES = ROOT / ".codex_tmp/sep01_09_sale_times.csv"
OUTPUT = Path(
    os.environ.get(
        "P50_BACKTEST_OUTPUT",
        ROOT / "reports/retrained_p50_comparable_daily_100_20260910",
    )
)
KEYS = ["date", "bakery_id", "product_id"]
LOSS_WEIGHT = 1.0
START = pd.Timestamp(os.environ.get("P50_BACKTEST_START", "2026-08-01"))
END = pd.Timestamp(os.environ.get("P50_BACKTEST_END", "2026-09-09"))


def load_detail() -> pd.DataFrame:
    parts = []
    for path in [AUGUST_DETAIL, SEPTEMBER_DETAIL]:
        part = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        part["date"] = pd.to_datetime(part["business_date"]).dt.normalize()
        parts.append(part)
    detail = pd.concat(parts, ignore_index=True).drop_duplicates(KEYS, keep="last")
    detail = detail[detail["date"].between("2026-07-23", END)].copy()
    for column in ["sold_qty", "forecast_qty"]:
        detail[column] = pd.to_numeric(detail[column], errors="coerce").fillna(0.0)
    return detail


def add_end_day_lost(detail: pd.DataFrame) -> pd.DataFrame:
    times = pd.concat(
        [pd.read_csv(path) for path in [JULY_TIMES, AUGUST_TIMES, SEPTEMBER_TIMES]],
        ignore_index=True,
    )
    times["date"] = pd.to_datetime(times["date"]).dt.normalize()
    times = times.drop_duplicates(KEYS, keep="last")
    work = detail.merge(
        times[KEYS + ["last_sale_time", "bakery_last_sale_time"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    last = pd.to_datetime(work["last_sale_time"], errors="coerce", utc=True)
    bakery_end = pd.to_datetime(
        work["bakery_last_sale_time"], errors="coerce", utc=True
    )
    opening = work["date"].dt.tz_localize("Europe/Moscow").dt.tz_convert(
        "UTC"
    ) + pd.Timedelta(hours=7.5)
    elapsed = (last - opening).dt.total_seconds() / 3600
    remaining = (bakery_end - last).dt.total_seconds().clip(lower=0) / 3600
    eligible = (
        work["eligible_lost_demand"].fillna(False).astype(bool)
        & elapsed.ge(2)
        & remaining.gt(0)
        & work["sold_qty"].gt(0)
    )
    raw = (work["sold_qty"] / elapsed * remaining).where(eligible, 0.0).clip(lower=0)
    cap = pd.concat(
        [work["sold_qty"] * 1.5, pd.Series(15.0, index=work.index)], axis=1
    ).max(axis=1)
    work["lost"] = raw.clip(upper=cap)
    return work


def stable_cohort(detail: pd.DataFrame) -> set[int]:
    evaluation = detail[detail["date"].between(START, END)]
    required_days = evaluation["date"].nunique()
    coverage = evaluation.groupby("bakery_id")["date"].nunique()
    return set(coverage[coverage.eq(required_days)].index.astype(int))


def extend_base(base: pd.DataFrame, detail: pd.DataFrame) -> pd.DataFrame:
    daily = (
        detail.groupby(["date", "bakery_id"], as_index=False)["sold_qty"]
        .sum()
        .rename(columns={"sold_qty": TARGET_COL})
    )
    bakery_attributes = (
        base.dropna(subset=["bakery_id"])
        .sort_values("date")
        .groupby("bakery_id", as_index=False)
        .last()[["bakery_id", "bakery_name", "city"]]
    )
    daily = daily.merge(bakery_attributes, on="bakery_id", how="left")
    future = daily[daily["date"].gt(base["date"].max())].copy()
    for column in base.columns:
        if column not in future:
            future[column] = pd.NA
    return pd.concat([base, future[base.columns]], ignore_index=True)


def predict_daily(
    base: pd.DataFrame, daily_lost: pd.DataFrame, cohort: set[int]
) -> pd.DataFrame:
    predictions = []
    for forecast_date in pd.date_range(START, END):
        cutoff = forecast_date - pd.Timedelta(days=1)
        adjusted = base.merge(daily_lost, on=["date", "bakery_id"], how="left")
        adjusted["lost"] = adjusted["lost"].fillna(0.0)
        history = adjusted["date"].le(cutoff)
        adjusted.loc[history, TARGET_COL] += LOSS_WEIGHT * adjusted.loc[history, "lost"]
        frame = build_model_frame(
            rebuild_target_features(adjusted.drop(columns="lost"))
        )
        train = frame[frame["date"].le(cutoff)]
        test = frame[frame["date"].eq(forecast_date) & frame["bakery_id"].isin(cohort)]
        features = [column for column in BASE_FEATURES if column in frame]
        train_x, test_x = cast_category_columns(
            train[features].copy(), test[features].copy(), features
        )
        model = train_quantile(
            train_x,
            train[TARGET_COL],
            alpha=0.5,
            params={
                "n_estimators": 180,
                "learning_rate": 0.04,
                "num_leaves": 31,
                "min_child_samples": 100,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": 2.0,
                "random_state": 42,
                "verbosity": -1,
            },
        )
        result = test[["date", "bakery_id"]].copy()
        result["p50_bakery_new"] = predict_clipped(model, test_x)
        predictions.append(result)
        print(f"predicted {forecast_date.date()} rows={len(result)}")
    return pd.concat(predictions, ignore_index=True)


def allocate_to_skus(detail: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows = detail[detail["date"].between(START, END)].merge(
        predictions, on=["date", "bakery_id"], how="inner", validate="many_to_one"
    )
    rows["direct_forecast"] = rows["forecast_qty"].clip(lower=0.0)
    total = rows.groupby(["date", "bakery_id"])["direct_forecast"].transform("sum")
    rows["alpha25_tail_capped"] = (
        rows["direct_forecast"] / total.replace(0.0, pd.NA) * rows["p50_bakery_new"]
    )
    rows["scenario"] = "calibrated"
    return rows[KEYS + ["scenario", "direct_forecast", "alpha25_tail_capped"]]


def main() -> None:
    detail = add_end_day_lost(load_detail())
    cohort = stable_cohort(detail)
    base = pd.read_csv(BASE_PATH, encoding="utf-8-sig", low_memory=False)
    base["date"] = pd.to_datetime(base["date"]).dt.normalize()
    base = extend_base(base, detail)
    daily_lost = detail.groupby(["date", "bakery_id"], as_index=False)["lost"].sum()
    predictions = predict_daily(base, daily_lost, cohort)
    rows = allocate_to_skus(detail, predictions)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT / "bakery_day_predictions.csv", index=False)
    rows.to_parquet(OUTPUT / "rows.parquet", index=False)
    pd.DataFrame({"bakery_id": sorted(cohort)}).to_csv(
        OUTPUT / "stable_cohort.csv", index=False
    )
    print(
        f"cohort={len(cohort)} days={predictions['date'].nunique()} "
        f"prediction_rows={len(predictions)} sku_rows={len(rows)}"
    )


if __name__ == "__main__":
    main()
