import pandas as pd

from scripts.experiment_demand_adjusted_bakery_target import (
    apply_training_adjustments,
    rebuild_target_features,
    summarize_predictions,
)


def test_apply_training_adjustments_respects_cutoff():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2026-06-01", "2026-06-02"]),
        "bakery_id": [1, 1],
        "bakery_sales": [10.0, 20.0],
    })
    adjustments = pd.DataFrame({
        "date": pd.to_datetime(["2026-06-01", "2026-06-02"]),
        "bakery_id": [1, 1],
        "imputed_demand": [3.0, 5.0],
    })
    result = apply_training_adjustments(
        frame,
        adjustments,
        train_end=pd.Timestamp("2026-06-01"),
    )
    assert result["bakery_sales"].tolist() == [13.0, 20.0]


def test_rebuild_target_features_uses_adjusted_history():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2026-06-01", "2026-06-02"]),
        "bakery_id": [1, 1],
        "bakery_sales": [13.0, 20.0],
        "bakery_sales_lag1": [999.0, 999.0],
    })
    result = rebuild_target_features(frame)
    assert result.loc[1, "bakery_sales_lag1"] == 13.0


def test_summarize_predictions_splits_demand_loss_days():
    predictions = pd.DataFrame({
        "date": pd.to_datetime(["2026-06-01", "2026-06-02"]),
        "bakery_id": [1, 1],
        "bakery_name": ["A", "A"],
        "bakery_sales": [10.0, 20.0],
        "prediction": [12.0, 18.0],
    })
    demand_loss_days = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-02")],
            "bakery_id": [1],
            "imputed_demand": [5.0],
        }
    )
    result = summarize_predictions(
        predictions,
        demand_loss_days,
        variant="x",
    ).set_index("scope")
    assert result.loc["no_demand_loss_pilot_bakery_days", "rows"] == 1
    assert result.loc["demand_loss_days_observed_sales", "rows"] == 1
    assert result.loc["demand_loss_days_reconstructed_demand", "actual_qty"] == 25.0
