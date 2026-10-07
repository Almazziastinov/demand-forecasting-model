"""Contract tests requiring no model weights or optional Chronos package."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.model_tournament.chronos2_zero_shot import (
    build_chronos2_context,
    predict_chronos2_origin,
)
from src.model_tournament.fixed_origin import validate_flows


class FakePipeline:
    def __init__(self, *, omit_last: bool = False) -> None:
        self.contexts: list[pd.DataFrame] = []
        self.omit_last = omit_last

    def predict_df(self, context_df: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        self.contexts.append(context_df.copy())
        assert kwargs["quantile_levels"] == [0.5]
        assert kwargs["target"] == "target"
        horizon = int(kwargs["prediction_length"])
        rows = []
        for series_id, group in context_df.groupby("id"):
            last_date = group["timestamp"].max()
            last_target = group.loc[group["timestamp"].eq(last_date), "target"].iloc[0]
            for lead in range(1, horizon + 1):
                rows.append(
                    {
                        "id": series_id,
                        "timestamp": last_date + pd.Timedelta(days=lead),
                        "0.5": last_target,
                    }
                )
        if self.omit_last:
            rows.pop()
        return pd.DataFrame(rows)


def _flows() -> pd.DataFrame:
    return validate_flows(
        pd.DataFrame(
            {
                "date": [
                    "2026-01-01",
                    "2026-01-09",
                    "2026-01-10",
                    "2026-01-11",
                    "2026-01-11",
                ],
                "bakery_id": [1, 1, 1, 1, 2],
                "product_id": [10, 10, 10, 10, 10],
                "observed_sales_qty": [2.0, 3.0, 5.0, 100.0, 900.0],
                "release_qty": [4.0, 0.0, 0.0, 0.0, 5.0],
            }
        )
    )


def test_context_is_causal_dense_and_has_frozen_scope() -> None:
    context, id_map = build_chronos2_context(_flows(), "2026-01-10")
    assert len(id_map) == 1
    assert id_map[["bakery_id", "product_id"]].iloc[0].tolist() == [1, 10]
    assert len(context) == 56
    assert context["timestamp"].max() == pd.Timestamp("2026-01-10")
    assert context["target"].sum() == 10.0
    assert (
        context.loc[context["timestamp"].eq(pd.Timestamp("2026-01-08")), "target"].iloc[
            0
        ]
        == 0
    )


def test_future_actual_and_release_never_enter_chronos_inputs() -> None:
    initial = _flows()
    changed = initial.copy()
    changed.loc[
        changed["date"].eq(pd.Timestamp("2026-01-11")), "observed_sales_qty"
    ] = 5000.0
    changed.loc[changed["date"].eq(pd.Timestamp("2026-01-11")), "release_qty"] = 5000.0
    first_pipeline = FakePipeline()
    second_pipeline = FakePipeline()
    first = predict_chronos2_origin(
        initial, "2026-01-10", first_pipeline, horizon_days=2
    )
    second = predict_chronos2_origin(
        changed, "2026-01-10", second_pipeline, horizon_days=2
    )
    pd.testing.assert_frame_equal(
        first_pipeline.contexts[0], second_pipeline.contexts[0]
    )
    np.testing.assert_array_equal(first["prediction"], second["prediction"])
    assert first["prediction"].tolist() == [5.0, 5.0]
    assert first["actual"].iloc[0] == 100.0
    assert second["actual"].iloc[0] == 5000.0


def test_incomplete_forecast_is_rejected() -> None:
    with pytest.raises(ValueError, match="exactly cover"):
        predict_chronos2_origin(
            _flows(), "2026-01-10", FakePipeline(omit_last=True), horizon_days=2
        )
