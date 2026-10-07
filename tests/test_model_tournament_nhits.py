"""Causal panel and output-contract tests for the optional N-HiTS candidate."""

import numpy as np
import pandas as pd
import pytest

from src.model_tournament.fixed_origin import validate_flows
from src.model_tournament.nhits_candidate import (
    align_predictions,
    dense_history,
    evaluation_history_and_rows,
    training_history,
)


def _flows() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=150)
    rows = []
    for day in dates:
        rows.append(
            {
                "date": day,
                "bakery_id": 1,
                "product_id": 10,
                "observed_sales_qty": 2.0,
                "release_qty": 3.0,
            }
        )
    return validate_flows(pd.DataFrame(rows))


def test_dense_history_uses_cutoff_and_zero_fills_sparse_dates() -> None:
    flows = _flows()
    flows = flows.loc[flows["date"] != pd.Timestamp("2026-01-03")]
    pairs = pd.DataFrame({"bakery_id": [1], "product_id": [10]})
    history = dense_history(
        flows, pairs, pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-04")
    )
    assert history["y"].tolist() == [2.0, 2.0, 0.0, 2.0]
    assert history["ds"].max() == pd.Timestamp("2026-01-04")
    assert history["unique_id"].nunique() == 1


def test_training_and_evaluation_histories_do_not_see_future_sales() -> None:
    flows = _flows()
    origin = pd.Timestamp("2026-04-01")
    train, counts = training_history(
        flows, [pd.Timestamp("2026-03-01")], start=pd.Timestamp("2026-01-01")
    )
    assert train["ds"].max() == pd.Timestamp("2026-03-15")
    assert counts["eligible_train_pairs"] == 1
    context, rows = evaluation_history_and_rows(flows, origin)
    assert context["ds"].max() == origin
    assert context["ds"].min() == origin - pd.Timedelta(days=55)
    assert len(context) == 56
    assert len(rows) == 14
    altered = flows.copy()
    altered.loc[altered["date"] > origin, "observed_sales_qty"] = 9999.0
    altered_context, _ = evaluation_history_and_rows(altered, origin)
    pd.testing.assert_frame_equal(context, altered_context)


def test_prediction_alignment_is_exact_and_clips_negative_values() -> None:
    flows = _flows()
    _, rows = evaluation_history_and_rows(flows, pd.Timestamp("2026-04-01"))
    prediction = pd.DataFrame(
        {"unique_id": ["1:10"] * 14, "ds": rows["date"], "NHITS": [-2.0] * 14}
    )
    detail = align_predictions(rows, prediction)
    assert np.all(detail["prediction"].to_numpy() == 0.0)
    with pytest.raises(ValueError, match="frozen evaluation scope"):
        align_predictions(rows, prediction.iloc[:-1])
    non_finite = prediction.copy()
    non_finite.loc[0, "NHITS"] = np.nan
    with pytest.raises(ValueError, match="missing or non-finite"):
        align_predictions(rows, non_finite)
    with pytest.raises(ValueError, match="duplicate"):
        align_predictions(rows, pd.concat([prediction, prediction.iloc[:1]]))
    extra = pd.DataFrame(
        {"unique_id": ["9:9"], "ds": [rows["date"].iloc[0]], "NHITS": [1.0]}
    )
    with pytest.raises(ValueError, match="frozen evaluation scope"):
        align_predictions(rows, pd.concat([prediction, extra], ignore_index=True))
