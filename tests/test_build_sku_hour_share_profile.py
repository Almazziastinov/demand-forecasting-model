"""Tests for SKU hour-share profile builder."""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments_v2.build_sku_hour_share_profile import aggregate_sku_hourly_chunk  # noqa: E402
from src.experiments_v2.build_sku_hour_share_profile import build_sku_hour_share_profile  # noqa: E402
from src.experiments_v2.build_sku_hour_share_profile import filter_hourly_by_assortment  # noqa: E402


def _hourly() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-05"),
                "dow": 0,
                "hour": 8,
                "bakery_id": "B1",
                "bakery_name": "Bakery 1",
                "product_id": "P1",
                "product_name": "Product 1",
                "category_name": "Cat",
                "sku_hour_sales": 2.0,
            },
            {
                "date": pd.Timestamp("2026-01-05"),
                "dow": 0,
                "hour": 8,
                "bakery_id": "B1",
                "bakery_name": "Bakery 1",
                "product_id": "P2",
                "product_name": "Product 2",
                "category_name": "Cat",
                "sku_hour_sales": 6.0,
            },
            {
                "date": pd.Timestamp("2026-01-12"),
                "dow": 0,
                "hour": 8,
                "bakery_id": "B1",
                "bakery_name": "Bakery 1",
                "product_id": "P1",
                "product_name": "Product 1",
                "category_name": "Cat",
                "sku_hour_sales": 1.0,
            },
            {
                "date": pd.Timestamp("2026-01-12"),
                "dow": 0,
                "hour": 8,
                "bakery_id": "B1",
                "bakery_name": "Bakery 1",
                "product_id": "P2",
                "product_name": "Product 2",
                "category_name": "Cat",
                "sku_hour_sales": 3.0,
            },
        ]
    )


def test_build_sku_hour_share_profile_normalizes_per_bakery_hour():
    profile, applied = build_sku_hour_share_profile(_hourly())
    sums = profile.groupby(["bakery_id", "dow", "hour"])[
        "mean_sku_share_in_hour_norm"
    ].sum()
    assert float(sums.iloc[0]) == 1.0
    p1 = profile[profile["product_id"] == "P1"].iloc[0]
    p2 = profile[profile["product_id"] == "P2"].iloc[0]
    assert round(float(p1["mean_sku_share_in_hour_norm"]), 4) == 0.25
    assert round(float(p2["mean_sku_share_in_hour_norm"]), 4) == 0.75


def test_assortment_filter_removes_sku_before_share_normalization():
    hourly = _hourly()
    hourly["city"] = "Kazan"
    hourly["product_id"] = hourly["product_id"].replace({"P1": "1", "P2": "2"})
    assortment = pd.DataFrame(
        [
            {
                "city": "Kazan",
                "product_id": "1",
            }
        ]
    )

    filtered, stats = filter_hourly_by_assortment(hourly, assortment)
    profile, applied = build_sku_hour_share_profile(filtered)

    assert stats["rows_removed"] == 2
    assert set(profile["product_id"]) == {"1"}
    assert round(float(profile["mean_sku_share_in_hour_norm"].iloc[0]), 4) == 1.0
    assert round(float(applied["sku_share_in_hour"].iloc[0]), 4) == 1.0


def test_build_sku_hour_share_profile_uses_daily_profile_weights():
    hourly = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-05"),
                "dow": 0,
                "hour": 8,
                "bakery_id": "B1",
                "bakery_name": "Bakery 1",
                "product_id": "P1",
                "product_name": "Product 1",
                "category_name": "Cat",
                "sku_hour_sales": 9.0,
            },
            {
                "date": pd.Timestamp("2026-01-05"),
                "dow": 0,
                "hour": 8,
                "bakery_id": "B1",
                "bakery_name": "Bakery 1",
                "product_id": "P2",
                "product_name": "Product 2",
                "category_name": "Cat",
                "sku_hour_sales": 1.0,
            },
            {
                "date": pd.Timestamp("2026-01-12"),
                "dow": 0,
                "hour": 8,
                "bakery_id": "B1",
                "bakery_name": "Bakery 1",
                "product_id": "P1",
                "product_name": "Product 1",
                "category_name": "Cat",
                "sku_hour_sales": 1.0,
            },
            {
                "date": pd.Timestamp("2026-01-12"),
                "dow": 0,
                "hour": 8,
                "bakery_id": "B1",
                "bakery_name": "Bakery 1",
                "product_id": "P2",
                "product_name": "Product 2",
                "category_name": "Cat",
                "sku_hour_sales": 9.0,
            },
        ]
    )
    weights = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-05"),
                "_bakery_id_norm": "B1",
                "profile_weight": 1.0,
            },
            {
                "date": pd.Timestamp("2026-01-12"),
                "_bakery_id_norm": "B1",
                "profile_weight": 0.1,
            },
        ]
    )

    profile, applied = build_sku_hour_share_profile(hourly, daily_weights=weights)

    p1 = profile[profile["product_id"] == "P1"].iloc[0]
    p2 = profile[profile["product_id"] == "P2"].iloc[0]
    assert round(float(p1["mean_sku_share_in_hour_norm"]), 4) == 0.8273
    assert round(float(p2["mean_sku_share_in_hour_norm"]), 4) == 0.1727
    assert round(float(applied["profile_weight"].mean()), 2) == 0.55


def test_build_sku_hour_share_profile_blends_recent_share():
    rows = []
    start = pd.Timestamp("2026-01-05")
    for i in range(6):
        date = start + pd.Timedelta(days=7 * i)
        if i < 2:
            p1_qty, p2_qty = 2.0, 8.0
        else:
            p1_qty, p2_qty = 8.0, 2.0
        rows.extend(
            [
                {
                    "date": date,
                    "dow": 0,
                    "hour": 8,
                    "bakery_id": "B1",
                    "bakery_name": "Bakery 1",
                    "product_id": "P1",
                    "product_name": "Product 1",
                    "category_name": "Cat",
                    "sku_hour_sales": p1_qty,
                },
                {
                    "date": date,
                    "dow": 0,
                    "hour": 8,
                    "bakery_id": "B1",
                    "bakery_name": "Bakery 1",
                    "product_id": "P2",
                    "product_name": "Product 2",
                    "category_name": "Cat",
                    "sku_hour_sales": p2_qty,
                },
            ]
        )
    hourly = pd.DataFrame(rows)

    profile, _ = build_sku_hour_share_profile(
        hourly,
        recent_days=28,
        recent_alpha=0.4,
    )

    p1 = profile[profile["product_id"] == "P1"].iloc[0]
    p2 = profile[profile["product_id"] == "P2"].iloc[0]
    assert round(float(p1["long_sku_share_in_hour"]), 4) == 0.6
    assert round(float(p1["recent_sku_share_in_hour"]), 4) == 0.8
    assert round(float(p1["mean_sku_share_in_hour_norm"]), 4) == 0.68
    assert round(float(p2["mean_sku_share_in_hour_norm"]), 4) == 0.32
    assert int(p1["recent_n_days"]) == 4
    assert float(p1["share_recent_alpha"]) == 0.4


def _hourly_for_reliability(
    *,
    n_days: int,
    sku_qty: list[float],
    other_qty: list[float],
) -> pd.DataFrame:
    assert len(sku_qty) == n_days and len(other_qty) == n_days
    rows = []
    base = pd.Timestamp("2026-01-05")
    for i in range(n_days):
        date = base + pd.Timedelta(days=7 * i)
        rows.append(
            {
                "date": date,
                "dow": 0,
                "hour": 8,
                "bakery_id": "B1",
                "bakery_name": "Bakery 1",
                "product_id": "P1",
                "product_name": "Product 1",
                "category_name": "Cat",
                "sku_hour_sales": sku_qty[i],
            }
        )
        rows.append(
            {
                "date": date,
                "dow": 0,
                "hour": 8,
                "bakery_id": "B1",
                "bakery_name": "Bakery 1",
                "product_id": "P2",
                "product_name": "Product 2",
                "category_name": "Cat",
                "sku_hour_sales": other_qty[i],
            }
        )
    return pd.DataFrame(rows)


def test_reliability_score_high_for_clean_long_history():
    n = 24
    hourly = _hourly_for_reliability(
        n_days=n,
        sku_qty=[5.0] * n,
        other_qty=[5.0] * n,
    )
    profile, _ = build_sku_hour_share_profile(hourly)

    p1 = profile[profile["product_id"] == "P1"].iloc[0]
    assert p1["n_days"] == n
    assert p1["zero_share_rate"] == 0.0
    assert p1["anomaly_share"] == 0.0
    assert p1["reliability_score"] > 0.95


def test_reliability_score_collapses_on_intermittent_sku():
    n = 24
    sku_qty = [5.0 if i % 4 == 0 else 0.0 for i in range(n)]
    hourly = _hourly_for_reliability(
        n_days=n,
        sku_qty=sku_qty,
        other_qty=[10.0] * n,
    )
    profile, _ = build_sku_hour_share_profile(hourly)

    p1 = profile[profile["product_id"] == "P1"].iloc[0]
    assert p1["zero_share_rate"] == 0.75
    assert p1["reliability_score"] < 0.2


def test_reliability_score_collapses_on_anomaly_days():
    n = 24
    hourly = _hourly_for_reliability(
        n_days=n,
        sku_qty=[5.0] * n,
        other_qty=[5.0] * n,
    )
    weights = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-05") + pd.Timedelta(days=7 * i),
                "_bakery_id_norm": "B1",
                "profile_weight": 0.35,
            }
            for i in range(n)
        ]
    )

    profile, _ = build_sku_hour_share_profile(hourly, daily_weights=weights)

    p1 = profile[profile["product_id"] == "P1"].iloc[0]
    assert p1["n_days"] == n
    assert p1["zero_share_rate"] == 0.0
    assert p1["anomaly_share"] == 1.0
    assert p1["reliability_score"] == 0.0


def test_aggregate_sku_hourly_chunk_supports_legacy_russian_snapshot_columns():
    raw = pd.DataFrame(
        [
            {
                "Дата продажи": "01.01.2026",
                "Дата время чека": "01.01.2026 14:21:01",
                "Вид события по кассе": "Продажа",
                "Касса.Торговая точка": "Bakery Legacy",
                "Номенклатура": "Product Legacy",
                "Категория": "Cat Legacy",
                "Кол-во": 2.0,
            }
        ]
    )
    hourly = aggregate_sku_hourly_chunk(raw)
    assert len(hourly) == 1
    row = hourly.iloc[0]
    assert row["bakery_id"] == "Bakery Legacy"
    assert row["product_id"] == "Product Legacy"
    assert row["hour"] == 14
    assert row["sku_hour_sales"] == 2.0
