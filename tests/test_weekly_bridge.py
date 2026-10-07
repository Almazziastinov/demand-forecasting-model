"""Exact-week and forecast-origin guards for retrospective weekly smoothing."""

from __future__ import annotations

import pandas as pd
import pytest

from src.model_tournament.weekly_bridge import weekly_bridge_dips


def _daily_panel() -> pd.DataFrame:
    rows = []
    for bakery_id in (1, 2):
        for product_id in (1, 2):
            for date in pd.date_range("2026-01-05", periods=21):
                sales = 30.0
                if product_id == 1 and date == pd.Timestamp("2026-01-05"):
                    sales = 120.0
                if product_id == 1 and date == pd.Timestamp("2026-01-12"):
                    sales = 80.0 if bakery_id == 1 else 125.0
                if product_id == 1 and date == pd.Timestamp("2026-01-19"):
                    sales = 130.0
                rows.append(
                    {
                        "date": date,
                        "bakery_id": bakery_id,
                        "product_id": product_id,
                        "observed_sales_qty": sales,
                    }
                )
    return pd.DataFrame(rows)


def test_bridge_matches_example_but_not_before_right_week_arrives() -> None:
    panel = _daily_panel()
    date = pd.Timestamp("2026-01-12")
    complete = weekly_bridge_dips(panel)
    chosen = complete.loc[
        complete["date"].eq(date)
        & complete["bakery_id"].eq(1)
        & complete["product_id"].eq(1)
    ].iloc[0]
    assert chosen["bridge_reference"] == pytest.approx(125.0)
    assert chosen["pure_weekly_bridge_uplift"] == pytest.approx(45.0)
    assert chosen["context_weekly_bridge_uplift"] == pytest.approx(0.0)

    as_of_second_monday = weekly_bridge_dips(panel[panel["date"].le(date)])
    same_day = as_of_second_monday.loc[
        as_of_second_monday["date"].eq(date)
        & as_of_second_monday["bakery_id"].eq(1)
        & as_of_second_monday["product_id"].eq(1)
    ].iloc[0]
    assert same_day["pure_weekly_bridge_uplift"] == 0.0


def test_sustained_weekly_drop_is_not_bridged() -> None:
    panel = _daily_panel()
    panel.loc[
        panel["date"].eq("2026-01-19") & panel["product_id"].eq(1),
        "observed_sales_qty",
    ] = 75.0
    result = weekly_bridge_dips(panel)
    row = result.loc[
        result["date"].eq("2026-01-12")
        & result["bakery_id"].eq(1)
        & result["product_id"].eq(1)
    ].iloc[0]
    assert row["pure_weekly_bridge_uplift"] == 0.0


def test_missing_interior_day_is_not_silently_treated_as_a_week() -> None:
    panel = _daily_panel()
    panel = panel.loc[
        ~(
            panel["date"].eq("2026-01-07")
            & panel["bakery_id"].eq(1)
            & panel["product_id"].eq(1)
        )
    ]
    result = weekly_bridge_dips(panel)
    row = result.loc[
        result["date"].eq("2026-01-12")
        & result["bakery_id"].eq(1)
        & result["product_id"].eq(1)
    ].iloc[0]
    assert row["pure_weekly_bridge_uplift"] == 0.0
