from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_chronos2_fixed_origin import _origins, align_chronos_predictions


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_origin": [pd.Timestamp("2026-07-06")] * 2,
            "date": pd.to_datetime(["2026-07-07", "2026-07-08"]),
            "lead_days": [1, 2],
            "bakery_id": [1, 1],
            "product_id": [10, 10],
            "actual": [3.0, 0.0],
        }
    )


def test_chronos_alignment_preserves_frozen_keys_and_clips_negative() -> None:
    predicted = pd.DataFrame(
        {
            "unique_id": ["1:10", "1:10"],
            "ds": pd.to_datetime(["2026-07-08", "2026-07-07"]),
            "0.5": [-1.0, 2.5],
        }
    )
    detail = align_chronos_predictions(_rows(), predicted)
    assert detail["prediction"].tolist() == [2.5, 0.0]
    assert detail["actual"].tolist() == [3.0, 0.0]


def test_chronos_alignment_rejects_missing_and_duplicate_keys() -> None:
    predicted = pd.DataFrame(
        {
            "unique_id": ["1:10"],
            "ds": pd.to_datetime(["2026-07-07"]),
            "0.5": [2.5],
        }
    )
    with pytest.raises(ValueError, match="fully cover"):
        align_chronos_predictions(_rows(), predicted)
    duplicate = pd.concat([predicted, predicted], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        align_chronos_predictions(_rows(), duplicate)


def test_chronos_origins_must_not_overlap() -> None:
    assert _origins("2026-07-20,2026-07-06") == [
        pd.Timestamp("2026-07-06"),
        pd.Timestamp("2026-07-20"),
    ]
    with pytest.raises(ValueError, match="overlap"):
        _origins("2026-07-06,2026-07-19")
