"""Causal, auditable demand regularization on synthetic daily histories."""

import pandas as pd
import pytest

from src.model_tournament.demand_regularization import (
    adjust_demand_panel,
    dense_candidate_panel,
)


def _panel() -> pd.DataFrame:
    rows = []
    for bakery_id in range(1, 7):
        for date in pd.date_range("2026-01-01", periods=70):
            rows.append(
                {
                    "date": date,
                    "bakery_id": bakery_id,
                    "product_id": 1,
                    "observed_sales_qty": 10.0,
                    "release_qty": 12.0,
                    "incoming_move_qty": 0.0,
                    "outgoing_move_qty": 0.0,
                    "written_off_qty": 0.0,
                    "last_sale_hour": 19.0,
                    "bakery_last_sale_hour": 19.0,
                }
            )
    return pd.DataFrame(rows)


def test_adjustment_is_causal_and_allows_local_reduction() -> None:
    frame = _panel()
    target_date = pd.Timestamp("2026-03-05")
    for bakery_id in (1, 3, 4):
        mask = frame["date"].eq(target_date) & frame["bakery_id"].eq(bakery_id)
        frame.loc[mask, ["observed_sales_qty", "release_qty", "last_sale_hour"]] = [
            5.0, 5.0, 14.0
        ]
    spike = frame["date"].eq(target_date) & frame["bakery_id"].eq(2)
    frame.loc[spike, "observed_sales_qty"] = 26.0
    adjusted = adjust_demand_panel(frame)
    restored = adjusted.loc[
        adjusted["date"].eq(target_date) & adjusted["bakery_id"].eq(1)
    ].iloc[0]
    reduced = adjusted.loc[
        adjusted["date"].eq(target_date) & adjusted["bakery_id"].eq(2)
    ].iloc[0]
    assert restored["restoration_uplift_qty"] == pytest.approx(2.5)
    assert reduced["outlier_reduction_qty"] == pytest.approx(5.2)
    assert reduced["reconstructed_demand_qty"] < reduced["observed_sales_qty"]
    assert (
        adjusted["reconstructed_demand_qty"].sum()
        > adjusted["observed_sales_qty"].sum()
    )

    future = frame.copy()
    future.loc[future["date"].gt(target_date), "observed_sales_qty"] = 1000.0
    past = adjust_demand_panel(future).loc[lambda x: x["date"].le(target_date)]
    original = adjusted.loc[adjusted["date"].le(target_date)]
    pd.testing.assert_series_equal(
        past["reconstructed_demand_qty"].reset_index(drop=True),
        original["reconstructed_demand_qty"].reset_index(drop=True),
    )


def test_missing_whole_bakery_day_is_not_imputed_as_zero() -> None:
    frame = _panel().loc[lambda x: x["date"].isin(
        [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-03")]
    )].copy()
    frame = frame.drop(columns="bakery_last_sale_hour")
    pairs = frame[["bakery_id", "product_id"]].drop_duplicates()
    with pytest.raises(ValueError, match="missing bakery-day"):
        dense_candidate_panel(
            frame,
            pairs,
            start=pd.Timestamp("2026-01-01"),
            end=pd.Timestamp("2026-01-03"),
        )
