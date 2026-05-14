"""Tests for experiment 72 bakery regime-shift helpers."""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments_v2.bakery_regime_shift_common import BAKERY_ID_COL  # noqa: E402
from src.experiments_v2.bakery_regime_shift_common import DATE_COL  # noqa: E402
from src.experiments_v2.bakery_regime_shift_common import TARGET_COL  # noqa: E402
from src.experiments_v2.bakery_regime_shift_common import add_fast_seasonal_features  # noqa: E402
from src.experiments_v2.bakery_regime_shift_common import add_normalized_target  # noqa: E402
from src.experiments_v2.bakery_regime_shift_common import bakery_predictability_table  # noqa: E402
from src.experiments_v2.bakery_regime_shift_common import build_bakery_weekly_frame  # noqa: E402
from src.experiments_v2.bakery_regime_shift_common import build_model_frame  # noqa: E402
from src.experiments_v2.bakery_regime_shift_common import compute_recent_weekday_share_lookup  # noqa: E402


def _frame() -> pd.DataFrame:
    rows = []
    for bakery_id, city, base in [("B1", "Kazan", 10.0), ("B2", "Moscow", 20.0)]:
        for i, dt in enumerate(pd.date_range("2026-01-01", periods=35, freq="D")):
            rows.append(
                {
                    DATE_COL: dt,
                    BAKERY_ID_COL: bakery_id,
                    "bakery_name": bakery_id,
                    "city": city,
                    TARGET_COL: base + (dt.dayofweek * 2) + i * 0.5,
                    "avg_price": 100.0 + i,
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


def test_add_normalized_target_uses_roll_anchor():
    df = add_normalized_target(_frame())
    assert "target_norm_roll7" in df.columns
    assert df["target_norm_roll7"].notna().all()


def test_fast_seasonal_features_add_expected_columns():
    df = add_fast_seasonal_features(_frame())
    for col in ["same_dow_mean_2w", "same_dow_mean_4w", "week_over_week_ratio", "peak_ratio_7d"]:
        assert col in df.columns
        assert df[col].notna().all()


def test_recent_weekday_share_lookup_normalizes_per_bakery():
    shares = compute_recent_weekday_share_lookup(_frame(), recent_weeks=4)
    sums = shares.groupby(BAKERY_ID_COL)["weekday_share"].sum().round(6)
    assert (sums == 1.0).all()


def test_build_bakery_weekly_frame_creates_week_features():
    weekly = build_bakery_weekly_frame(_frame())
    assert "week_sales" in weekly.columns
    assert "week_sales_lag1" in weekly.columns
    assert weekly["week_start"].nunique() >= 4


def test_bakery_predictability_table_has_override_flag():
    table = bakery_predictability_table(_frame())
    assert "use_local_override" in table.columns
    assert table["use_local_override"].isin([True, False]).all()
