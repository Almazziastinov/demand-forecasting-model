from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.finetune_chronos2_fixed_origin import require_eval_mode, training_arrays


def test_training_arrays_are_sorted_and_preserve_zero_days() -> None:
    days = pd.date_range("2026-01-01", periods=70)
    frame = pd.DataFrame(
        {
            "unique_id": ["2:20"] * 70 + ["1:10"] * 70,
            "ds": list(days) + list(days),
            "y": [2.0] * 70 + [0.0] * 69 + [3.0],
        }
    )
    arrays = training_arrays(frame)
    assert len(arrays) == 2
    assert arrays[0].dtype == np.float32
    assert arrays[0].tolist() == [0.0] * 69 + [3.0]
    assert arrays[1].tolist() == [2.0] * 70


def test_training_arrays_reject_short_or_duplicate_series() -> None:
    frame = pd.DataFrame(
        {
            "unique_id": ["1:10"] * 2,
            "ds": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "y": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="shorter"):
        training_arrays(frame)
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        training_arrays(duplicate)


def test_require_eval_mode_switches_trainer_model_and_fails_closed() -> None:
    class Model:
        training = True

        def eval(self) -> None:
            self.training = False

    class Pipeline:
        model = Model()

    pipeline = Pipeline()
    require_eval_mode(pipeline)
    assert pipeline.model.training is False

    class StubbornModel(Model):
        def eval(self) -> None:
            pass

    pipeline.model = StubbornModel()
    with pytest.raises(RuntimeError, match="evaluation mode"):
        require_eval_mode(pipeline)
