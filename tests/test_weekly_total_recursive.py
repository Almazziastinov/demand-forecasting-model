"""Tests for experiment 73 weekly total recursive helpers."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments_v2.bakery_day_forecast import BAKERY_ID_COL  # noqa: E402
from src.experiments_v2.bakery_day_forecast import DATE_COL  # noqa: E402
from src.experiments_v2.bakery_day_forecast import TARGET_COL  # noqa: E402
from src.experiments_v2.bakery_day_forecast import build_model_frame  # noqa: E402
from src.experiments_v2.bakery_regime_shift_common import compute_adaptive_weekday_share_lookup  # noqa: E402
from src.experiments_v2.bakery_regime_shift_common import compute_recent_weekday_share_lookup  # noqa: E402

RUN_PATH = Path(__file__).resolve().parents[1] / "src" / "experiments_v2" / "73_weekly_total_recursive" / "run.py"
SPEC = importlib.util.spec_from_file_location("exp73_run", RUN_PATH)
EXP73_RUN = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(EXP73_RUN)
build_complete_weekly_history = EXP73_RUN.build_complete_weekly_history
build_elapsed_week_sales_lookup = EXP73_RUN.build_elapsed_week_sales_lookup
heuristic_blend_recursive_backtest = EXP73_RUN.heuristic_blend_recursive_backtest
repeat_last_week_recursive_backtest = EXP73_RUN.repeat_last_week_recursive_backtest
seasonal_naive_lag7_recursive_backtest = EXP73_RUN.seasonal_naive_lag7_recursive_backtest


def _daily_frame() -> pd.DataFrame:
    rows = []
    for bakery_id, city, base in [("B1", "Kazan", 10.0), ("B2", "Moscow", 20.0)]:
        for dt in pd.date_range("2026-01-01", periods=28, freq="D"):
            rows.append(
                {
                    DATE_COL: dt,
                    BAKERY_ID_COL: bakery_id,
                    "bakery_name": bakery_id,
                    "city": city,
                    TARGET_COL: base + dt.dayofweek,
                    "avg_price": 100.0,
                    "dow": dt.dayofweek,
                    "day": dt.day,
                    "month": dt.month,
                    "iso_week": int(dt.isocalendar().week),
                    "is_weekend": int(dt.dayofweek >= 5),
                    "is_month_start": int(dt.day <= 5),
                    "is_month_end": int(dt.day >= 25),
                    "is_payday_week": int(dt.day in [4, 5, 6, 19, 20, 21]),
                    "bakery_sales_lag1": 0.0,
                    "bakery_sales_lag2": 0.0,
                    "bakery_sales_lag3": 0.0,
                    "bakery_sales_lag7": 0.0,
                    "bakery_sales_lag14": 0.0,
                    "bakery_sales_lag30": 0.0,
                    "bakery_sales_roll_mean3": 0.0,
                    "bakery_sales_roll_mean7": 0.0,
                    "bakery_sales_roll_mean14": 0.0,
                    "bakery_sales_roll_mean30": 0.0,
                    "bakery_sales_roll_std7": 0.0,
                }
            )
    return build_model_frame(pd.DataFrame(rows))


def test_recent_weekday_share_lookup_sums_to_one_per_bakery():
    shares = compute_recent_weekday_share_lookup(_daily_frame(), recent_weeks=4)
    sums = shares.groupby(BAKERY_ID_COL)["weekday_share"].sum().round(6)
    assert (sums == 1.0).all()


def test_adaptive_weekday_share_lookup_sums_to_one_per_bakery():
    shares = compute_adaptive_weekday_share_lookup(_daily_frame(), recent_weeks=4)
    sums = shares.groupby(BAKERY_ID_COL)["weekday_share"].sum().round(6)
    assert (sums == 1.0).all()


def test_build_complete_weekly_history_drops_partial_edge_weeks():
    df = _daily_frame()
    partial = df[df[DATE_COL] >= pd.Timestamp("2026-01-03")].copy()

    weekly = build_complete_weekly_history(partial)

    assert not weekly.empty
    assert weekly["week_start"].min() == pd.Timestamp("2026-01-05")
    assert pd.Timestamp("2025-12-29") not in set(weekly["week_start"])


def test_build_elapsed_week_sales_lookup_sums_known_days_in_current_week():
    df = _daily_frame()
    history = df[df[DATE_COL] < pd.Timestamp("2026-01-08")].copy()

    elapsed = build_elapsed_week_sales_lookup(history, pd.Timestamp("2026-01-05"))

    assert set(elapsed.columns) == {BAKERY_ID_COL, "elapsed_week_sales"}
    expected_b1 = history[(history[BAKERY_ID_COL] == "B1") & (history[DATE_COL] >= pd.Timestamp("2026-01-05"))][TARGET_COL].sum()
    expected_b2 = history[(history[BAKERY_ID_COL] == "B2") & (history[DATE_COL] >= pd.Timestamp("2026-01-05"))][TARGET_COL].sum()
    got = dict(zip(elapsed[BAKERY_ID_COL], elapsed["elapsed_week_sales"]))
    assert got["B1"] == expected_b1
    assert got["B2"] == expected_b2


def test_adaptive_weekday_share_lookup_tracks_last_complete_week_signal():
    rows = []
    for dt in pd.date_range("2026-01-05", periods=28, freq="D"):
        base = 100.0
        if dt >= pd.Timestamp("2026-01-26"):
            base = 300.0 if dt.dayofweek == 0 else 50.0
        rows.append(
            {
                DATE_COL: dt,
                BAKERY_ID_COL: "B1",
                "bakery_name": "B1",
                "city": "Kazan",
                TARGET_COL: base,
                "avg_price": 100.0,
                "dow": dt.dayofweek,
            }
        )
    df = pd.DataFrame(rows)

    adaptive = compute_adaptive_weekday_share_lookup(df, recent_weeks=4)
    monday_share = adaptive.loc[adaptive["dow"] == 0, "weekday_share"].iloc[0]
    tuesday_share = adaptive.loc[adaptive["dow"] == 1, "weekday_share"].iloc[0]

    assert monday_share > tuesday_share


def test_repeat_last_week_recursive_copies_previous_week_pattern():
    df = _daily_frame()
    train = df[df[DATE_COL] < pd.Timestamp("2026-01-22")].copy()
    test = df[(df[DATE_COL] >= pd.Timestamp("2026-01-22")) & (df[DATE_COL] < pd.Timestamp("2026-01-29"))].copy()

    preds, _ = repeat_last_week_recursive_backtest(train, test)
    merged = test[[DATE_COL, BAKERY_ID_COL, "dow"]].merge(preds, on=[DATE_COL, BAKERY_ID_COL], how="left")

    history = train[[DATE_COL, BAKERY_ID_COL, "dow", TARGET_COL]].copy()
    history["next_week_date"] = history[DATE_COL] + pd.Timedelta(days=7)
    expected = history[[BAKERY_ID_COL, "dow", "next_week_date", TARGET_COL]].rename(
        columns={"next_week_date": DATE_COL, TARGET_COL: "expected"}
    )
    merged = merged.merge(expected, on=[DATE_COL, BAKERY_ID_COL, "dow"], how="left")
    assert (merged["prediction"] == merged["expected"]).all()


def test_seasonal_naive_lag7_recursive_uses_lag7_column():
    df = _daily_frame()
    train = df[df[DATE_COL] < pd.Timestamp("2026-01-22")].copy()
    test = df[(df[DATE_COL] >= pd.Timestamp("2026-01-22")) & (df[DATE_COL] < pd.Timestamp("2026-01-24"))].copy()

    preds, _ = seasonal_naive_lag7_recursive_backtest(train, test)
    first_day = preds[preds[DATE_COL] == preds[DATE_COL].min()].copy()

    expected = train[train[DATE_COL] == pd.Timestamp("2026-01-15")][[BAKERY_ID_COL, TARGET_COL]].rename(
        columns={TARGET_COL: "expected"}
    )
    first_day = first_day.merge(expected, on=BAKERY_ID_COL, how="left")
    assert (first_day["prediction"] == first_day["expected"]).all()


def test_heuristic_blend_recursive_returns_non_negative_predictions():
    df = _daily_frame()
    train = df[df[DATE_COL] < pd.Timestamp("2026-01-22")].copy()
    test = df[(df[DATE_COL] >= pd.Timestamp("2026-01-22")) & (df[DATE_COL] < pd.Timestamp("2026-01-25"))].copy()

    preds, info = heuristic_blend_recursive_backtest(train, test, min_train_rows=10)

    assert len(preds) == len(test)
    assert (preds["prediction"] >= 0).all()
    assert "heuristic_blend" in info["status"]
