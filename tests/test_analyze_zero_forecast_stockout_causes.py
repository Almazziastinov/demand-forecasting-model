import pandas as pd

from scripts.analyze_zero_forecast_stockout_causes import classify_causes
from scripts.analyze_zero_forecast_stockout_causes import membership_in_latest_batch


def test_classify_causes_respects_versioned_exclusions():
    frame = pd.DataFrame(
        {
            "assortment_asof": [False, True, True, True],
            "bakeable_asof": [True, False, True, True],
            "current_profile_present": [True, True, False, True],
        }
    )

    assert classify_causes(frame)["cause"].tolist() == [
        "excluded_by_assortment_asof",
        "excluded_by_bakeability_asof",
        "forecast_grid_drop_current_profile_missing",
        "forecast_grid_drop",
    ]


def test_membership_ignores_batch_loaded_after_historical_run():
    frame = pd.DataFrame(
        {
            "city": ["C", "C"],
            "product_id": [1, 2],
            "valid_from": ["2026-07-18", "2026-07-19"],
            "valid_to": [None, None],
            "loaded_at": ["2026-07-18T03:00:00Z", "2026-07-20T03:00:00Z"],
        }
    )
    row = pd.Series(
        {
            "city": "C",
            "product_id": 2,
            "date": pd.Timestamp("2026-07-19"),
            "run_generated_at": pd.Timestamp("2026-07-19T03:00:00Z"),
        }
    )

    assert not membership_in_latest_batch(frame, row)
