"""Guardrails for retrospective hierarchical weekly-dip proposals."""

from __future__ import annotations

import pandas as pd
import pytest

from src.model_tournament.hierarchical_weekly_bridge import hierarchical_weekly_bridge


def _fixture(
    *, dip_bakeries: tuple[int, ...] = (1,), dip_date: str = "2026-06-15"
) -> pd.DataFrame:
    rows = []
    for date in pd.date_range("2026-06-01", "2026-06-22"):
        for bakery_id in (1, 2, 3):
            for product_id, amount in ((10, 40.0), (11, 60.0)):
                sales = amount
                if str(date.date()) == dip_date and bakery_id in dip_bakeries:
                    sales *= 0.7
                rows.append(
                    {
                        "date": date,
                        "bakery_id": bakery_id,
                        "product_id": product_id,
                        "observed_sales_qty": sales,
                        "release_qty": 100.0,
                    }
                )
    return pd.DataFrame(rows)


def _categories() -> pd.DataFrame:
    return pd.DataFrame({"product_id": [10, 11], "category_name": ["bread", "bread"]})


def test_isolated_bakery_dip_is_allocated_without_overshooting_reference() -> None:
    result = hierarchical_weekly_bridge(_fixture(), _categories())
    case = result.loc[
        (result["date"] == pd.Timestamp("2026-06-15")) & (result["bakery_id"] == 1)
    ]
    assert case["bakery_dip_pass"].all()
    assert case["bakery_only_uplift"].sum() == pytest.approx(30)
    assert case["category_guarded_uplift"].sum() == pytest.approx(30)
    assert case.set_index("product_id").loc[
        10, "category_guarded_uplift"
    ] == pytest.approx(12)
    assert result.loc[result["bakery_id"].ne(1), "category_guarded_uplift"].eq(0).all()


def test_network_wide_dip_is_not_reconstructed() -> None:
    result = hierarchical_weekly_bridge(_fixture(dip_bakeries=(1, 2, 3)), _categories())
    case = result.loc[result["date"] == pd.Timestamp("2026-06-15")]
    assert not case["bakery_dip_pass"].any()
    assert case["category_guarded_uplift"].eq(0).all()


def test_right_week_must_be_available() -> None:
    frame = _fixture().loc[lambda rows: rows["date"].le("2026-06-18")]
    result = hierarchical_weekly_bridge(frame, _categories())
    case = result.loc[result["date"] == pd.Timestamp("2026-06-15")]
    assert not case["bakery_dip_pass"].any()
    assert case["bakery_only_uplift"].eq(0).all()


def test_known_holiday_dip_is_not_reconstructed() -> None:
    result = hierarchical_weekly_bridge(_fixture(dip_date="2026-06-12"), _categories())
    case = result.loc[
        (result["date"] == pd.Timestamp("2026-06-12")) & result["bakery_id"].eq(1)
    ]
    assert not case["bakery_dip_pass"].any()
    assert case["category_guarded_uplift"].eq(0).all()


def test_future_after_right_week_cannot_change_existing_proposal() -> None:
    frame = _fixture()
    early = hierarchical_weekly_bridge(
        frame.loc[frame["date"].le("2026-06-22")], _categories()
    )
    later = pd.concat(
        [
            frame,
            frame.loc[frame["date"].eq("2026-06-22")].assign(
                date=pd.Timestamp("2026-06-23"), observed_sales_qty=500.0
            ),
        ],
        ignore_index=True,
    )
    late = hierarchical_weekly_bridge(later, _categories())
    key = (early["date"] == pd.Timestamp("2026-06-15")) & early["bakery_id"].eq(1)
    other_key = (late["date"] == pd.Timestamp("2026-06-15")) & late["bakery_id"].eq(1)
    assert (
        early.loc[key, "category_guarded_uplift"].tolist()
        == late.loc[other_key, "category_guarded_uplift"].tolist()
    )


def test_duplicate_sku_day_rejected() -> None:
    frame = _fixture()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate SKU-day"):
        hierarchical_weekly_bridge(frame, _categories())
