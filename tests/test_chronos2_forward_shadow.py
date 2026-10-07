from __future__ import annotations

import pandas as pd
import pytest

from scripts.freeze_chronos2_forward_shadow import freeze_frames
from scripts.run_chronos2_lora_forward_shadow import align_shadow_predictions
from scripts.score_chronos2_lora_forward_shadow import align_actuals


def test_freeze_frames_appends_only_new_facts() -> None:
    historical = pd.DataFrame(
        {
            "date": ["2026-09-01"],
            "bakery_id": [1],
            "product_id": [10],
            "observed_sales_qty": [2.0],
            "release_qty": [3.0],
        }
    )
    sales = pd.DataFrame(
        {
            "date": ["2026-09-02"],
            "bakery_id": [1],
            "product_id": [10],
            "observed_sales_qty": [1.0],
        }
    )
    release = pd.DataFrame(
        {
            "date": ["2026-09-02"],
            "bakery_id": [1],
            "product_id": [11],
            "release_qty": [4.0],
        }
    )
    frozen, coverage = freeze_frames(
        historical, sales, release, origin=pd.Timestamp("2026-09-02")
    )
    assert len(frozen) == 3
    assert frozen["date"].max() == pd.Timestamp("2026-09-02")
    assert coverage["current_daily_min_bakeries"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        freeze_frames(
            historical, pd.concat([sales, sales]), release,
            origin=pd.Timestamp("2026-09-02"),
        )


def test_shadow_alignment_has_no_future_actuals() -> None:
    rows = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-10-03", "2026-10-04"]),
            "bakery_id": [1, 1],
            "product_id": [10, 10],
            "actual": [0.0, 0.0],
        }
    )
    predicted = pd.DataFrame(
        {
            "unique_id": ["1:10", "1:10"],
            "ds": pd.to_datetime(["2026-10-04", "2026-10-03"]),
            "0.5": [2.0, -1.0],
        }
    )
    result = align_shadow_predictions(rows, predicted, "lora")
    assert "actual" not in result.columns
    assert result["prediction"].tolist() == [0.0, 2.0]
    with pytest.raises(ValueError, match="cover"):
        align_shadow_predictions(rows, predicted.iloc[[0]], "lora")


def test_scoring_uses_same_keys_and_zero_for_missing_sku_facts() -> None:
    day = pd.Timestamp("2026-10-03")
    keys = {"date": [day], "bakery_id": [1], "product_id": [10]}
    predictions = pd.concat(
        [
            pd.DataFrame({**keys, "model": [model], "prediction": [value]})
            for model, value in [
                ("chronos2_small_zero_shot", 2.0),
                ("chronos2_small_lora_1000", 1.0),
                ("incumbent_active_at_freeze", 3.0),
            ]
        ],
        ignore_index=True,
    )
    sales = pd.DataFrame(
        {**keys, "observed_sales_qty": [4.0]}
    )
    detail, common = align_actuals(predictions, sales)
    assert len(detail) == len(common) == 3
    assert detail["actual"].tolist() == [4.0] * 3
    zero_actual, _ = align_actuals(predictions, sales.iloc[0:0])
    assert zero_actual["actual"].tolist() == [0.0] * 3
    with pytest.raises(ValueError, match="duplicate"):
        align_actuals(pd.concat([predictions, predictions.iloc[[0]]]), sales)
