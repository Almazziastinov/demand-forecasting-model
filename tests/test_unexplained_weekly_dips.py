"""Research-target policy keeps facts and candidate explanations separate."""

import pandas as pd

from src.model_tournament.unexplained_weekly_dips import annotate_unexplained_dips


def test_unexplained_dip_is_smoothed_without_overwriting_sales() -> None:
    rows = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-04-13", "2026-05-01"]),
            "bakery_id": [1, 1],
            "product_id": [10, 10],
            "observed_sales_qty": [80.0, 80.0],
            "bridge_reference": [125.0, 125.0],
            "isolated_weekly_dip": [True, True],
        }
    )
    result = annotate_unexplained_dips(rows, {pd.Timestamp("2026-05-01")})
    assert result["observed_sales_qty"].tolist() == [80.0, 80.0]
    assert result["target_half"].tolist() == [102.5, 80.0]
    assert result["target_full"].tolist() == [125.0, 80.0]
    assert result["unverified_uplift"].tolist() == [45.0, 0.0]
    assert result["release_uplift"].tolist() == [0.0, 0.0]
    assert result["reason_code"].tolist() == [
        "unexplained_weekly_dip",
        "calendar_candidate_not_smoothed",
    ]


def test_release_shortfall_is_a_separate_candidate_reason() -> None:
    dates = pd.to_datetime(["2026-02-06", "2026-02-13", "2026-02-20"])
    bridge = pd.DataFrame(
        {
            "date": dates,
            "bakery_id": [5, 5, 5],
            "product_id": [36, 36, 36],
            "observed_sales_qty": [194.0, 40.0, 216.0],
            "bridge_reference": [float("nan"), 205.0, float("nan")],
            "isolated_weekly_dip": [False, True, False],
        }
    )
    releases = bridge[["date", "bakery_id", "product_id"]].copy()
    releases["release_qty"] = [196.0, 42.0, 200.0]
    result = annotate_unexplained_dips(bridge, set(), releases)
    case = result.loc[result["date"].eq(pd.Timestamp("2026-02-13"))].iloc[0]
    assert case["release_shortfall_candidate"]
    assert not case["unexplained_dip"]
    assert case["reason_code"] == "release_shortfall_candidate"
    assert case["target_full"] == 205.0
    assert case["release_uplift"] == 165.0
    assert case["unverified_uplift"] == 0.0
