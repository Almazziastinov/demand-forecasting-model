"""
Shared helpers for experiment 72 bakery regime-shift forecasting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.experiments_v2.bakery_day_forecast import BASE_FEATURES  # noqa: E402
from src.experiments_v2.bakery_day_forecast import BAKERY_ID_COL  # noqa: E402
from src.experiments_v2.bakery_day_forecast import BAKERY_NAME_COL  # noqa: E402
from src.experiments_v2.bakery_day_forecast import CITY_COL  # noqa: E402
from src.experiments_v2.bakery_day_forecast import CATEGORICAL_COLS  # noqa: E402
from src.experiments_v2.bakery_day_forecast import DATE_COL  # noqa: E402
from src.experiments_v2.bakery_day_forecast import TARGET_COL  # noqa: E402
from src.experiments_v2.bakery_day_forecast import build_model_frame  # noqa: E402


def load_bakery_frame(dataset_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_path, encoding="utf-8-sig")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, BAKERY_ID_COL, TARGET_COL]).copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0.0).clip(lower=0.0)
    if CITY_COL not in df.columns:
        df[CITY_COL] = "unknown"
    if BAKERY_NAME_COL not in df.columns:
        df[BAKERY_NAME_COL] = df[BAKERY_ID_COL].astype(str)
    return build_model_frame(df).sort_values([BAKERY_ID_COL, DATE_COL]).reset_index(drop=True)


def make_train_test_split(df: pd.DataFrame, test_days: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    max_date = df[DATE_COL].max()
    test_start = max_date - pd.Timedelta(days=test_days - 1)
    train_df = df[df[DATE_COL] < test_start].copy()
    test_df = df[df[DATE_COL] >= test_start].copy()
    return train_df, test_df, test_start


def cast_category_columns(train_x: pd.DataFrame, test_x: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    for col in [c for c in CATEGORICAL_COLS if c in feature_cols]:
        train_x[col] = train_x[col].astype("category")
        test_x[col] = pd.Categorical(test_x[col], categories=train_x[col].cat.categories)
    return train_x, test_x


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if len(y_true_arr) == 0:
        return {"mae": np.nan, "mse": np.nan, "wmape": np.nan, "bias": np.nan}
    return {
        "mae": float(np.mean(np.abs(y_true_arr - y_pred_arr))),
        "mse": float(np.mean((y_true_arr - y_pred_arr) ** 2)),
        "wmape": float(np.sum(np.abs(y_true_arr - y_pred_arr)) / (np.sum(y_true_arr) + 1e-8) * 100.0),
        "bias": float(np.mean(y_true_arr - y_pred_arr)),
    }


def add_fast_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.sort_values([BAKERY_ID_COL, DATE_COL]).copy()
    grp = work.groupby(BAKERY_ID_COL, sort=False)
    dow_grp = work.groupby([BAKERY_ID_COL, "dow"], sort=False)

    work["same_dow_mean_2w"] = dow_grp[TARGET_COL].transform(lambda x: x.shift(1).rolling(window=2, min_periods=1).mean())
    work["same_dow_mean_4w"] = dow_grp[TARGET_COL].transform(lambda x: x.shift(1).rolling(window=4, min_periods=1).mean())
    work["same_dow_std_4w"] = dow_grp[TARGET_COL].transform(lambda x: x.shift(1).rolling(window=4, min_periods=2).std())
    work["week_over_week_ratio"] = work["bakery_sales_lag7"] / (work["bakery_sales_lag14"] + 1e-8)
    work["recent_level_ratio"] = work["bakery_sales_roll_mean7"] / (work["bakery_sales_roll_mean14"] + 1e-8)
    work["acceleration_ratio"] = work["bakery_sales_roll_mean7"] / (work["bakery_sales_roll_mean30"] + 1e-8)
    work["peak_ratio_7d"] = grp[TARGET_COL].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).max()
    ) / (work["bakery_sales_roll_mean7"] + 1e-8)
    work["weekday_share_recent"] = work["same_dow_mean_4w"] / (
        work.groupby(BAKERY_ID_COL, sort=False)["same_dow_mean_4w"].transform(lambda x: x.rolling(7, min_periods=1).sum())
        + 1e-8
    )

    fill_cols = [
        "same_dow_mean_2w",
        "same_dow_mean_4w",
        "same_dow_std_4w",
        "week_over_week_ratio",
        "recent_level_ratio",
        "acceleration_ratio",
        "peak_ratio_7d",
        "weekday_share_recent",
    ]
    for col in fill_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return work


FAST_SEASONAL_FEATURES = BASE_FEATURES + [
    "same_dow_mean_2w",
    "same_dow_mean_4w",
    "same_dow_std_4w",
    "week_over_week_ratio",
    "recent_level_ratio",
    "acceleration_ratio",
    "peak_ratio_7d",
    "weekday_share_recent",
]


def add_normalized_target(df: pd.DataFrame, anchor_col: str = "bakery_sales_roll_mean7") -> pd.DataFrame:
    work = df.copy()
    anchor = pd.to_numeric(work[anchor_col], errors="coerce").fillna(0.0).clip(lower=1.0)
    work["target_norm_roll7"] = work[TARGET_COL] / anchor
    return work


def build_bakery_weekly_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.sort_values([BAKERY_ID_COL, DATE_COL]).copy()
    work["week_start"] = work[DATE_COL] - pd.to_timedelta(work[DATE_COL].dt.dayofweek, unit="D")
    weekly = (
        work.groupby([BAKERY_ID_COL, BAKERY_NAME_COL, CITY_COL, "week_start"], as_index=False)
        .agg(
            week_sales=(TARGET_COL, "sum"),
            week_mean_day_sales=(TARGET_COL, "mean"),
            week_max_day_sales=(TARGET_COL, "max"),
            dow0_sales=(TARGET_COL, lambda s: float(s[work.loc[s.index, "dow"] == 0].sum())),
            dow1_sales=(TARGET_COL, lambda s: float(s[work.loc[s.index, "dow"] == 1].sum())),
            dow2_sales=(TARGET_COL, lambda s: float(s[work.loc[s.index, "dow"] == 2].sum())),
            dow3_sales=(TARGET_COL, lambda s: float(s[work.loc[s.index, "dow"] == 3].sum())),
            dow4_sales=(TARGET_COL, lambda s: float(s[work.loc[s.index, "dow"] == 4].sum())),
            dow5_sales=(TARGET_COL, lambda s: float(s[work.loc[s.index, "dow"] == 5].sum())),
            dow6_sales=(TARGET_COL, lambda s: float(s[work.loc[s.index, "dow"] == 6].sum())),
            avg_price=("avg_price", "mean"),
        )
        .sort_values([BAKERY_ID_COL, "week_start"])
        .reset_index(drop=True)
    )
    weekly["month"] = weekly["week_start"].dt.month
    weekly["iso_week"] = weekly["week_start"].dt.isocalendar().week.astype(int)
    weekly["week_of_month"] = ((weekly["week_start"].dt.day - 1) // 7 + 1).astype(int)
    grp = weekly.groupby(BAKERY_ID_COL, sort=False)["week_sales"]
    for lag in [1, 2, 4]:
        weekly[f"week_sales_lag{lag}"] = grp.shift(lag)
    for window, min_periods in [(2, 1), (4, 2), (8, 4)]:
        weekly[f"week_sales_roll_mean{window}"] = grp.transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=min_periods).mean()
        )
    weekly["week_sales_trend"] = weekly["week_sales_roll_mean2"] / (weekly["week_sales_roll_mean8"] + 1e-8)
    fill_cols = [c for c in weekly.columns if c.startswith("week_sales_") or c in ["avg_price"]]
    for col in fill_cols:
        weekly[col] = pd.to_numeric(weekly[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return weekly


WEEKLY_FEATURES = [
    BAKERY_ID_COL,
    CITY_COL,
    "month",
    "iso_week",
    "week_of_month",
    "avg_price",
    "week_sales_lag1",
    "week_sales_lag2",
    "week_sales_lag4",
    "week_sales_roll_mean2",
    "week_sales_roll_mean4",
    "week_sales_roll_mean8",
    "week_sales_trend",
]


def compute_recent_weekday_share_lookup(
    history_df: pd.DataFrame,
    *,
    recent_weeks: int = 4,
) -> pd.DataFrame:
    work = history_df.sort_values([BAKERY_ID_COL, DATE_COL]).copy()
    work["week_start"] = work[DATE_COL] - pd.to_timedelta(work[DATE_COL].dt.dayofweek, unit="D")
    recent_cutoff = work["week_start"].max() - pd.Timedelta(days=7 * (recent_weeks - 1))
    recent = work[work["week_start"] >= recent_cutoff].copy()

    day_sales = recent.groupby([BAKERY_ID_COL, "dow"], as_index=False)[TARGET_COL].sum()
    totals = day_sales.groupby(BAKERY_ID_COL, as_index=False)[TARGET_COL].sum().rename(columns={TARGET_COL: "bakery_total"})
    shares = day_sales.merge(totals, on=BAKERY_ID_COL, how="left")
    shares["weekday_share"] = np.where(shares["bakery_total"] > 0, shares[TARGET_COL] / shares["bakery_total"], 0.0)

    fallback = (
        work.groupby([BAKERY_ID_COL, "dow"], as_index=False)[TARGET_COL]
        .mean()
        .rename(columns={TARGET_COL: "fallback_mean"})
    )
    shares = shares.merge(fallback, on=[BAKERY_ID_COL, "dow"], how="outer")
    shares["weekday_share"] = shares["weekday_share"].fillna(0.0)

    if shares.groupby(BAKERY_ID_COL)["weekday_share"].sum().eq(0).any():
        fb = shares.groupby(BAKERY_ID_COL, as_index=False)["fallback_mean"].sum().rename(columns={"fallback_mean": "fb_total"})
        shares = shares.merge(fb, on=BAKERY_ID_COL, how="left")
        shares["weekday_share"] = np.where(
            shares.groupby(BAKERY_ID_COL)["weekday_share"].transform("sum") > 0,
            shares["weekday_share"],
            np.where(shares["fb_total"] > 0, shares["fallback_mean"] / shares["fb_total"], 1.0 / 7.0),
        )
        shares.drop(columns=["fb_total"], inplace=True)

    totals = shares.groupby(BAKERY_ID_COL, as_index=False)["weekday_share"].sum().rename(columns={"weekday_share": "share_sum"})
    shares = shares.merge(totals, on=BAKERY_ID_COL, how="left")
    shares["weekday_share"] = np.where(shares["share_sum"] > 0, shares["weekday_share"] / shares["share_sum"], 1.0 / 7.0)
    return shares[[BAKERY_ID_COL, "dow", "weekday_share"]].copy()


def compute_adaptive_weekday_share_lookup(
    history_df: pd.DataFrame,
    *,
    recent_weeks: int = 4,
    weight_last_week: float = 0.50,
    weight_recent_window: float = 0.35,
    weight_long_run: float = 0.15,
) -> pd.DataFrame:
    work = history_df.sort_values([BAKERY_ID_COL, DATE_COL]).copy()
    work["week_start"] = work[DATE_COL] - pd.to_timedelta(work[DATE_COL].dt.dayofweek, unit="D")
    coverage = (
        work.groupby([BAKERY_ID_COL, "week_start"], as_index=False)
        .agg(n_days=(DATE_COL, "nunique"), n_dow=("dow", "nunique"))
    )
    complete_weeks = coverage[(coverage["n_days"] == 7) & (coverage["n_dow"] == 7)][[BAKERY_ID_COL, "week_start"]]
    complete = work.merge(complete_weeks, on=[BAKERY_ID_COL, "week_start"], how="inner")

    if complete.empty:
        return compute_recent_weekday_share_lookup(work, recent_weeks=recent_weeks)

    long_run = complete.groupby([BAKERY_ID_COL, "dow"], as_index=False)[TARGET_COL].mean().rename(
        columns={TARGET_COL: "long_mean"}
    )

    last_complete_week = complete.groupby(BAKERY_ID_COL, as_index=False)["week_start"].max().rename(
        columns={"week_start": "last_week_start"}
    )
    last_week = complete.merge(last_complete_week, on=BAKERY_ID_COL, how="inner")
    last_week = last_week[last_week["week_start"] == last_week["last_week_start"]].copy()
    last_week = last_week.groupby([BAKERY_ID_COL, "dow"], as_index=False)[TARGET_COL].sum().rename(
        columns={TARGET_COL: "last_week_sales"}
    )

    recent_cutoff = complete["week_start"].max() - pd.Timedelta(days=7 * (recent_weeks - 1))
    recent = complete[complete["week_start"] >= recent_cutoff].copy()
    recent_window = recent.groupby([BAKERY_ID_COL, "dow"], as_index=False)[TARGET_COL].mean().rename(
        columns={TARGET_COL: "recent_mean"}
    )

    shares = long_run.merge(last_week, on=[BAKERY_ID_COL, "dow"], how="outer")
    shares = shares.merge(recent_window, on=[BAKERY_ID_COL, "dow"], how="outer")
    for col in ["long_mean", "last_week_sales", "recent_mean"]:
        shares[col] = pd.to_numeric(shares[col], errors="coerce").fillna(0.0)

    shares["weighted_signal"] = (
        shares["last_week_sales"] * weight_last_week
        + shares["recent_mean"] * weight_recent_window
        + shares["long_mean"] * weight_long_run
    )

    # Fallback for weak history: if blended signal collapses, revert to simpler recent share logic.
    signal_totals = shares.groupby(BAKERY_ID_COL, as_index=False)["weighted_signal"].sum().rename(
        columns={"weighted_signal": "signal_total"}
    )
    shares = shares.merge(signal_totals, on=BAKERY_ID_COL, how="left")
    shares["weekday_share"] = np.where(
        shares["signal_total"] > 0,
        shares["weighted_signal"] / shares["signal_total"],
        0.0,
    )

    weak_ids = set(shares.loc[shares["signal_total"] <= 0, BAKERY_ID_COL].astype(str))
    if weak_ids:
        fallback = compute_recent_weekday_share_lookup(work, recent_weeks=recent_weeks)
        fallback[BAKERY_ID_COL] = fallback[BAKERY_ID_COL].astype(str)
        shares[BAKERY_ID_COL] = shares[BAKERY_ID_COL].astype(str)
        strong = shares[~shares[BAKERY_ID_COL].isin(weak_ids)][[BAKERY_ID_COL, "dow", "weekday_share"]].copy()
        weak = fallback[fallback[BAKERY_ID_COL].isin(weak_ids)][[BAKERY_ID_COL, "dow", "weekday_share"]].copy()
        out = pd.concat([strong, weak], ignore_index=True)
    else:
        out = shares[[BAKERY_ID_COL, "dow", "weekday_share"]].copy()

    totals = out.groupby(BAKERY_ID_COL, as_index=False)["weekday_share"].sum().rename(columns={"weekday_share": "share_sum"})
    out = out.merge(totals, on=BAKERY_ID_COL, how="left")
    out["weekday_share"] = np.where(out["share_sum"] > 0, out["weekday_share"] / out["share_sum"], 1.0 / 7.0)
    return out[[BAKERY_ID_COL, "dow", "weekday_share"]].copy()


def local_seasonal_scaled_prediction(history_df: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    hist = history_df.sort_values([BAKERY_ID_COL, DATE_COL]).copy()
    dow = int(target_date.dayofweek)

    rows: list[dict] = []
    for bakery_id, group in hist.groupby(BAKERY_ID_COL, sort=False):
        sales = pd.to_numeric(group[TARGET_COL], errors="coerce").fillna(0.0)
        same_dow = group.loc[group["dow"] == dow, TARGET_COL]
        recent_same_dow = pd.to_numeric(same_dow, errors="coerce").dropna().tail(4)
        local_base = float(recent_same_dow.mean()) if len(recent_same_dow) else float(sales.tail(7).mean() if len(sales) else 0.0)
        roll7 = float(sales.tail(7).mean()) if len(sales) else 0.0
        roll30 = float(sales.tail(30).mean()) if len(sales) else roll7
        trend_scale = np.clip(roll7 / (roll30 + 1e-8), 0.7, 1.6) if roll30 > 0 else 1.0
        rows.append(
            {
                BAKERY_ID_COL: bakery_id,
                "local_scaled_pred": max(local_base * trend_scale, 0.0),
            }
        )
    return pd.DataFrame(rows)


def bakery_predictability_table(history_df: pd.DataFrame) -> pd.DataFrame:
    work = history_df.sort_values([BAKERY_ID_COL, DATE_COL]).copy()
    shares = work.groupby([BAKERY_ID_COL, "dow"], as_index=False)[TARGET_COL].mean()
    totals = shares.groupby(BAKERY_ID_COL, as_index=False)[TARGET_COL].sum().rename(columns={TARGET_COL: "dow_total"})
    shares = shares.merge(totals, on=BAKERY_ID_COL, how="left")
    shares["share"] = np.where(shares["dow_total"] > 0, shares[TARGET_COL] / shares["dow_total"], 0.0)
    summary = (
        shares.groupby(BAKERY_ID_COL, as_index=False)
        .agg(
            weekday_share_std=("share", "std"),
            weekday_share_max=("share", "max"),
        )
    )
    hist_summary = work.groupby(BAKERY_ID_COL, as_index=False).agg(
        train_days=(DATE_COL, "nunique"),
        roll7_mean=(TARGET_COL, lambda s: float(pd.to_numeric(s, errors="coerce").dropna().tail(7).mean()) if len(s) else 0.0),
        roll30_mean=(TARGET_COL, lambda s: float(pd.to_numeric(s, errors="coerce").dropna().tail(30).mean()) if len(s) else 0.0),
    )
    out = summary.merge(hist_summary, on=BAKERY_ID_COL, how="left")
    out["trend_ratio"] = np.where(out["roll30_mean"] > 0, out["roll7_mean"] / out["roll30_mean"], 1.0)
    out["use_local_override"] = (
        (out["train_days"] >= 84)
        & (out["weekday_share_std"] >= 0.03)
        & (out["trend_ratio"].between(0.85, 1.35))
    )
    return out[[BAKERY_ID_COL, "use_local_override", "trend_ratio", "weekday_share_std", "train_days"]]
