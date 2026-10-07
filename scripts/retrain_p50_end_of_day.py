from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment_demand_adjusted_bakery_target import rebuild_target_features  # noqa: E402
from src.experiments_v2.bakery_day_forecast import (  # noqa: E402
    BASE_FEATURES,
    TARGET_COL,
    build_model_frame,
    cast_category_columns,
)
from src.experiments_v2.common import predict_clipped, train_quantile  # noqa: E402


BASE = ROOT / ".codex_tmp/network_extension_20260826_bakery_daily_extended.csv.gz"
DETAIL = ROOT / ".codex_tmp/pilot_recalc_report/detail.csv"
JULY_TIMES = ROOT / ".codex_tmp/july23_31_sale_times.csv"
AUGUST_TIMES = ROOT / ".codex_tmp/august_sale_times.csv"
AUGUST_ROWS = (
    ROOT / "reports/versioned_direct_scope_p50_loss_0.75_august_20260909/rows.parquet"
)
SEPTEMBER_ROWS = (
    ROOT
    / "reports"
    / "versioned_direct_scope_p50_loss_0.75_september_through_09_20260910"
    / "rows.parquet"
)
LOSS_WEIGHT = float(os.environ.get("P50_LOSS_WEIGHT", "0.75"))
WEIGHT_LABEL = f"{LOSS_WEIGHT:.2f}".replace(".", "")
OUTPUT = ROOT / f"reports/retrained_p50_end_of_day_{WEIGHT_LABEL}_20260910"
KEYS = ["date", "bakery_id", "product_id"]
FOLDS = {
    "2026-07-27": "2026-07-26",
    "2026-08-10": "2026-08-09",
    "2026-08-17": "2026-08-16",
}


def end_day_labels() -> tuple[pd.DataFrame, set[int]]:
    detail = pd.read_csv(DETAIL, encoding="utf-8-sig", low_memory=False)
    detail["date"] = pd.to_datetime(detail["business_date"]).dt.normalize()
    detail = detail[detail["date"].between("2026-07-23", "2026-08-31")].drop_duplicates(
        KEYS
    )
    times = pd.concat(
        [pd.read_csv(JULY_TIMES), pd.read_csv(AUGUST_TIMES)], ignore_index=True
    )
    times["date"] = pd.to_datetime(times["date"]).dt.normalize()
    detail = detail.merge(
        times[KEYS + ["last_sale_time", "bakery_last_sale_time"]], on=KEYS, how="left"
    )
    last = pd.to_datetime(detail["last_sale_time"], errors="coerce", utc=True)
    end = pd.to_datetime(detail["bakery_last_sale_time"], errors="coerce", utc=True)
    opening = detail["date"].dt.tz_localize("Europe/Moscow").dt.tz_convert(
        "UTC"
    ) + pd.Timedelta(hours=7.5)
    elapsed = (last - opening).dt.total_seconds() / 3600
    remaining = (end - last).dt.total_seconds().clip(lower=0) / 3600
    sold = pd.to_numeric(detail["sold_qty"], errors="coerce").fillna(0.0)
    eligible = (
        detail["eligible_lost_demand"].fillna(False).astype(bool)
        & elapsed.ge(2)
        & remaining.gt(0)
    )
    raw = (sold / elapsed * remaining).where(eligible, 0.0).clip(lower=0.0)
    cap = pd.concat([sold * 1.5, pd.Series(15.0, index=detail.index)], axis=1).max(
        axis=1
    )
    detail["lost"] = raw.clip(upper=cap)
    daily = detail.groupby(["date", "bakery_id"], as_index=False)["lost"].sum()
    return daily, set(detail["bakery_id"].astype(int))


def rescale_shape(rows: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    result = rows.merge(
        factors[["date", "bakery_id", "p50_bakery_new"]],
        on=["date", "bakery_id"],
        validate="many_to_one",
    )
    old_total = result.groupby(["date", "bakery_id"])["alpha25_tail_capped"].transform(
        "sum"
    )
    result["alpha25_tail_capped"] *= result["p50_bakery_new"] / old_total.replace(
        0.0, np.nan
    )
    return result


def main() -> None:
    daily_lost, pilot_ids = end_day_labels()
    base = pd.read_csv(BASE, encoding="utf-8-sig", low_memory=False)
    base["date"] = pd.to_datetime(base["date"]).dt.normalize()
    base = base[base["bakery_id"].isin(pilot_ids)].merge(
        daily_lost, on=["date", "bakery_id"], how="left"
    )
    base["lost"] = base["lost"].fillna(0.0)
    august = pd.read_parquet(AUGUST_ROWS)
    august["date"] = pd.to_datetime(august["date"]).dt.normalize()
    august = august[august["scenario"].eq("calibrated")].copy()
    factor_parts = []
    for fold, cutoff_value in FOLDS.items():
        cutoff = pd.Timestamp(cutoff_value)
        adjusted = base.copy()
        adjusted.loc[adjusted["date"].le(cutoff), TARGET_COL] += (
            LOSS_WEIGHT * adjusted.loc[adjusted["date"].le(cutoff), "lost"]
        )
        frame = build_model_frame(
            rebuild_target_features(adjusted.drop(columns="lost"))
        )
        fold_rows = august[august["rolling_fold"].eq(fold)]
        test = frame[
            frame["date"].isin(fold_rows["date"].unique())
            & frame["bakery_id"].isin(fold_rows["bakery_id"].unique())
        ]
        train = frame[frame["date"].le(cutoff)]
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
        prediction = test[["date", "bakery_id"]].copy()
        prediction["p50_bakery_new"] = predict_clipped(model, test_x)
        totals = (
            fold_rows.groupby(["date", "bakery_id"], as_index=False)["direct_forecast"]
            .sum()
            .rename(columns={"direct_forecast": "direct_total"})
        )
        factors = totals.merge(
            prediction, on=["date", "bakery_id"], validate="one_to_one"
        )
        factors["p50_factor_new"] = factors["p50_bakery_new"] / factors[
            "direct_total"
        ].replace(0.0, np.nan)
        factors["rolling_fold"] = fold
        factor_parts.append(factors)
    factors = pd.concat(factor_parts, ignore_index=True)
    august_candidate = rescale_shape(august, factors)
    median = factors.groupby("bakery_id")["p50_factor_new"].median()
    fallback = float(factors["p50_factor_new"].median())
    september = pd.read_parquet(SEPTEMBER_ROWS)
    september["date"] = pd.to_datetime(september["date"]).dt.normalize()
    september = september[september["scenario"].eq("calibrated")].copy()
    september["factor_new"] = september["bakery_id"].map(median).fillna(fallback)
    sept_totals = september.groupby(["date", "bakery_id"])["direct_forecast"].transform(
        "sum"
    )
    sept_target = sept_totals * september["factor_new"]
    old_total = september.groupby(["date", "bakery_id"])[
        "alpha25_tail_capped"
    ].transform("sum")
    september["alpha25_tail_capped"] *= sept_target / old_total.replace(0.0, np.nan)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    factors.to_csv(
        OUTPUT / "walk_forward_factors.csv", index=False, encoding="utf-8-sig"
    )
    august_candidate.to_parquet(OUTPUT / "august_rows.parquet", index=False)
    september.to_parquet(OUTPUT / "september_rows.parquet", index=False)
    pd.DataFrame({"bakery_id": median.index, "p50_factor": median.values}).to_csv(
        OUTPUT / "deployed_factor_candidate.csv", index=False, encoding="utf-8-sig"
    )
    print(
        f"weight={LOSS_WEIGHT:.2f} factors={len(factors)} "
        f"bakeries={len(median)} fallback={fallback:.6f}"
    )


if __name__ == "__main__":
    main()
