from __future__ import annotations

import pandas as pd

from src.experiments_v2.stockout_demand_preprocessing import (
    aggregate_daily_training_target,
    build_bakery_share_reference,
    build_inventory_balance,
    build_uncensored_hour_reference,
    mark_stockout_days,
    reconstruct_stockout_demand,
    reconstruct_stockout_demand_from_bakery_share,
)


def _hourly() -> pd.DataFrame:
    rows = []
    daily_values = [
        ("2026-05-04", [2, 3, 2]),
        ("2026-05-11", [2, 4, 2]),
        ("2026-05-18", [3, 0, 0]),
    ]
    for day, values in daily_values:
        for hour, sold in zip([14, 15, 16], values, strict=True):
            common = {"date": pd.Timestamp(day), "bakery_id": 20, "hour": hour}
            rows.append({**common, "product_id": 100, "sold": sold})
            rows.append({**common, "product_id": 200, "sold": 5})
    return pd.DataFrame(rows)


def test_reconstructs_only_post_last_sale_stockout_zeroes() -> None:
    hourly = _hourly()
    production = pd.DataFrame(
        [
            {
                "date": pd.Timestamp(day),
                "bakery_id": 20,
                "product_id": pid,
                "produced": produced,
            }
            for day, produced in [
                ("2026-05-04", 20),
                ("2026-05-11", 20),
                ("2026-05-18", 3),
            ]
            for pid in [100]
        ]
    )
    marked = mark_stockout_days(hourly, production)
    train = marked[marked["date"] < pd.Timestamp("2026-05-18")]
    reference = build_uncensored_hour_reference(train, min_days=2)
    result = reconstruct_stockout_demand(marked, reference)

    target = result[
        (result["date"] == pd.Timestamp("2026-05-18")) & (result["product_id"] == 100)
    ]
    assert target.loc[target["hour"] == 14, "sold_demand"].item() == 3
    assert target.loc[target["hour"] == 15, "sold_demand"].item() == 3.5
    assert target.loc[target["hour"] == 16, "sold_demand"].item() == 2
    assert target["is_censored_hour"].sum() == 2

    daily = aggregate_daily_training_target(result)
    day = daily[daily["date"] == pd.Timestamp("2026-05-18")].iloc[0]
    assert day["demand_target"] > day["sales_observed"]


def test_does_not_fill_zero_on_non_stockout_day() -> None:
    hourly = _hourly()
    production = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-05-18"),
                "bakery_id": 20,
                "product_id": 100,
                "produced": 30,
            }
        ]
    )
    marked = mark_stockout_days(hourly, production)
    reference = build_uncensored_hour_reference(marked, min_days=1)
    result = reconstruct_stockout_demand(marked, reference)
    target = result[
        (result["date"] == pd.Timestamp("2026-05-18")) & (result["product_id"] == 100)
    ]
    assert target["is_censored_hour"].sum() == 0
    assert target["sold_demand"].sum() == target["sold"].sum()


def test_bakery_share_reference_scales_with_current_bakery_traffic() -> None:
    hourly = _hourly()
    production = pd.DataFrame(
        [
            {
                "date": pd.Timestamp(day),
                "bakery_id": 20,
                "product_id": 100,
                "produced": produced,
            }
            for day, produced in [
                ("2026-05-04", 20),
                ("2026-05-11", 20),
                ("2026-05-18", 3),
            ]
        ]
    )
    marked = mark_stockout_days(hourly, production)
    train = marked[marked["date"] < pd.Timestamp("2026-05-18")]
    reference = build_bakery_share_reference(train, min_days=2)
    result = reconstruct_stockout_demand_from_bakery_share(marked, reference)
    target = result[
        (result["date"] == pd.Timestamp("2026-05-18")) & (result["product_id"] == 100)
    ]

    expected = ((3 / 8 + 4 / 9) / 2) * 5
    assert target.loc[target["hour"] == 15, "sold_demand"].item() == expected
    assert target.loc[target["hour"] == 16, "sold_demand"].item() > 0


def test_inventory_balance_accounts_for_stock_and_moves() -> None:
    daily = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-03-01"),
                "bakery_id": 20,
                "product_id": 100,
                "opening_stock": 3.0,
                "produced": 10.0,
                "sold": 11.0,
                "closing_stock": 0.0,
            }
        ]
    )
    moves = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-03-01"),
                "bakery_id": 20,
                "product_id": 100,
                "incoming_move_qty": 2.0,
                "outgoing_move_qty": 4.0,
            }
        ]
    )
    row = build_inventory_balance(daily, moves).iloc[0]
    assert row["available_qty"] == 11.0
    assert row["expected_closing_stock"] == 0.0
    assert bool(row["balance_is_consistent"])
    assert bool(row["is_inventory_stockout"])
