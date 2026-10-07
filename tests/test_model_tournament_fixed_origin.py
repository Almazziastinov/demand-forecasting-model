from __future__ import annotations

import pandas as pd
import pytest
import numpy as np

from scripts.run_fixed_origin_tournament import run_fixed_origin_tournament
from src.model_tournament.fixed_origin import build_fixed_origin_panel, validate_flows
from src.model_tournament.fixed_origin_candidates import (
    DEFAULT_CANDIDATES,
    HURDLE_MEDIAN_ID,
    P60_ID,
    RESIDUAL_ID,
    _zero_inflated_median,
    fit_predict_candidates,
)


def _flows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-08", "2026-01-10", "2026-01-11"],
            "bakery_id": [1, 1, 2, 1],
            "product_id": [10, 10, 10, 10],
            "observed_sales_qty": [2.0, 4.0, 1.0, 100.0],
            "release_qty": [5.0, 0.0, 0.0, 0.0],
        }
    )


def test_fixed_origin_keeps_zero_activity_days_and_excludes_future_scope() -> None:
    panel = build_fixed_origin_panel(
        validate_flows(_flows()), "2026-01-10", horizon_days=2
    )
    assert set(panel["product_id"]) == {10}
    assert set(panel["bakery_id"]) == {1}
    assert panel["actual"].tolist() == [100.0, 0.0]
    assert panel["lead_days"].tolist() == [1, 2]


def test_future_truth_does_not_change_fixed_origin_forecast_features() -> None:
    first = validate_flows(_flows())
    changed = first.copy()
    changed.loc[
        changed["date"].eq(pd.Timestamp("2026-01-11")), "observed_sales_qty"
    ] = 999.0
    initial = build_fixed_origin_panel(first, "2026-01-10", horizon_days=2)
    later = build_fixed_origin_panel(changed, "2026-01-10", horizon_days=2)
    feature_cols = [
        "origin_mean1",
        "origin_mean7",
        "origin_mean14",
        "weighted_weekday_sales",
    ]
    pd.testing.assert_frame_equal(initial[feature_cols], later[feature_cols])
    assert initial.loc[0, "actual"] == 100.0
    assert later.loc[0, "actual"] == 999.0


def test_future_release_cannot_enter_fixed_origin_scope() -> None:
    flows = _flows()
    flows.loc[len(flows)] = ["2026-01-11", 3, 10, 0.0, 9.0]
    panel = build_fixed_origin_panel(validate_flows(flows), "2026-01-10")
    assert 3 not in set(panel["bakery_id"])


def test_reject_duplicate_flow_keys() -> None:
    flows = pd.concat([_flows(), _flows().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate SKU-day"):
        validate_flows(flows)


def test_trainable_fixed_origin_models_share_exact_evaluation_rows() -> None:
    dates = pd.date_range("2026-01-01", periods=80)
    flows = pd.DataFrame(
        [
            {
                "date": date,
                "bakery_id": bakery_id,
                "product_id": 10,
                "observed_sales_qty": float(date.dayofweek + bakery_id),
                "release_qty": 5.0,
            }
            for date in dates
            for bakery_id in (1, 2)
        ]
    )
    detail, leaderboard, metadata = run_fixed_origin_tournament(
        validate_flows(flows),
        [pd.Timestamp("2026-01-20"), pd.Timestamp("2026-02-10")],
        [pd.Timestamp("2026-03-01")],
        horizon_days=14,
        n_estimators=5,
        candidate_models=(P60_ID, RESIDUAL_ID),
    )
    assert len(leaderboard) == 3
    assert detail.groupby("model").size().nunique() == 1
    assert metadata["last_train_target_date"] == "2026-02-24"
    assert metadata["first_evaluation_target_date"] == "2026-03-02"
    assert (detail["prediction"] >= 0).all()


def test_all_legacy_and_new_candidate_adapters_return_finite_forecasts() -> None:
    count = 60
    X_train = pd.DataFrame(
        {
            "bakery_id": pd.Categorical([1, 2] * 30),
            "product_id": pd.Categorical([10, 11, 12] * 20),
            "lead_days": [1, 7, 14] * 20,
            "day_of_week": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2] * 6,
            "month": [6] * count,
            "day_of_month": list(range(1, 31)) * 2,
            "origin_mean1": [0.0, 2.0, 4.0] * 20,
            "origin_mean7": [1.0, 2.0, 3.0] * 20,
            "origin_mean14": [1.0, 2.0, 3.0] * 20,
            "weighted_weekday_sales": [0.0, 2.0, 3.0] * 20,
        }
    )
    X_eval = X_train.iloc[:10].copy()
    y_train = np.asarray([0.0, 2.0, 4.0] * 20)
    predictions = fit_predict_candidates(
        X_train,
        X_eval,
        y_train,
        X_train["weighted_weekday_sales"].to_numpy(dtype=float),
        X_eval["weighted_weekday_sales"].to_numpy(dtype=float),
        n_estimators=5,
    )
    assert set(predictions) == set(DEFAULT_CANDIDATES)
    assert all(len(values) == 10 for values in predictions.values())
    assert all(
        np.isfinite(values).all() and (values >= 0).all()
        for values in predictions.values()
    )


def test_zero_inflated_median_obeys_zero_mass_and_quantile_level() -> None:
    probability = np.asarray([0.0, 0.5, 0.6, 1.0])
    positive_quantiles = np.asarray(
        [[2.0, 5.0, 8.0], [2.0, 5.0, 8.0], [2.0, 5.0, 8.0], [2.0, 5.0, 8.0]]
    )
    result = _zero_inflated_median(probability, positive_quantiles)
    assert result[0] == result[1] == 0.0
    assert 2.0 < result[2] < 5.0
    assert result[3] == 8.0


def test_hurdle_median_is_deterministic() -> None:
    X_train = pd.DataFrame(
        {
            "bakery_id": pd.Categorical([1, 2] * 20),
            "product_id": pd.Categorical([10, 11] * 20),
            "lead_days": [1, 7, 14, 2] * 10,
            "day_of_week": [0, 1, 2, 3] * 10,
            "month": [6] * 40,
            "day_of_month": list(range(1, 21)) * 2,
            "origin_mean1": [0.0, 2.0] * 20,
            "origin_mean7": [1.0, 2.0] * 20,
            "origin_mean14": [1.0, 2.0] * 20,
            "weighted_weekday_sales": [0.0, 2.0] * 20,
        }
    )
    y_train = np.asarray([0.0, 3.0] * 20)
    evaluation = X_train.iloc[:6].copy()
    first = fit_predict_candidates(
        X_train,
        evaluation,
        y_train,
        X_train["weighted_weekday_sales"].to_numpy(),
        evaluation["weighted_weekday_sales"].to_numpy(),
        model_ids=(HURDLE_MEDIAN_ID,),
        n_estimators=5,
    )[HURDLE_MEDIAN_ID]
    second = fit_predict_candidates(
        X_train,
        evaluation,
        y_train,
        X_train["weighted_weekday_sales"].to_numpy(),
        evaluation["weighted_weekday_sales"].to_numpy(),
        model_ids=(HURDLE_MEDIAN_ID,),
        n_estimators=5,
    )[HURDLE_MEDIAN_ID]
    np.testing.assert_allclose(first, second)
