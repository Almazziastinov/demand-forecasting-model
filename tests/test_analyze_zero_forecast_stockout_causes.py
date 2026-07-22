import pandas as pd

from scripts.analyze_zero_forecast_stockout_causes import classify_causes


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
