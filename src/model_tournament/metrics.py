"""Canonical accuracy and operational metrics for model comparison."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


EPSILON = 1e-9


def canonical_metrics(
    frame: pd.DataFrame,
    *,
    actual_col: str = "actual",
    prediction_col: str = "prediction",
    sales_col: str | None = None,
) -> dict[str, float | int]:
    """Calculate one comparable metric record.

    Bias follows the forecast convention: positive values mean overforecast.
    MAPE excludes zero-actual rows and reports their count separately.
    """
    actual = pd.to_numeric(frame[actual_col], errors="coerce").to_numpy(dtype=float)
    predicted = pd.to_numeric(frame[prediction_col], errors="coerce").to_numpy(
        dtype=float
    )
    valid = np.isfinite(actual) & np.isfinite(predicted)
    if not valid.all():
        raise ValueError("Metrics received missing or non-finite actual/prediction")
    if len(actual) == 0:
        raise ValueError(
            "Cannot calculate metrics without valid actual/prediction pairs"
        )

    error = predicted - actual
    absolute = np.abs(error)
    squared = error**2
    denominator = float(np.abs(actual).sum())
    nonzero_actual = np.abs(actual) > EPSILON
    smape_denominator = np.abs(actual) + np.abs(predicted)
    nonzero_smape = smape_denominator > EPSILON
    under = np.maximum(actual - predicted, 0.0)
    over = np.maximum(predicted - actual, 0.0)
    served = np.minimum(np.maximum(predicted, 0.0), np.maximum(actual, 0.0))

    result: dict[str, float | int] = {
        "rows": int(len(actual)),
        "actual_sum": float(actual.sum()),
        "prediction_sum": float(predicted.sum()),
        "mae": float(absolute.mean()),
        "mse": float(squared.mean()),
        "rmse": float(np.sqrt(squared.mean())),
        "wmape_pct": float(100.0 * absolute.sum() / denominator)
        if denominator > EPSILON
        else math.nan,
        "mape_pct": float(
            100.0
            * np.mean(
                absolute[nonzero_actual] / np.abs(actual[nonzero_actual])
            )
        )
        if nonzero_actual.any()
        else math.nan,
        "mape_rows": int(nonzero_actual.sum()),
        "zero_actual_rows": int((~nonzero_actual).sum()),
        "false_positive_rows": int(
            ((~nonzero_actual) & (predicted > EPSILON)).sum()
        ),
        "false_positive_qty": float(predicted[~nonzero_actual].clip(min=0.0).sum()),
        "smape_pct": float(
            100.0
            * np.mean(2.0 * absolute[nonzero_smape] / smape_denominator[nonzero_smape])
        )
        if nonzero_smape.any()
        else 0.0,
        "bias_qty": float(error.sum()),
        "bias_pct": float(100.0 * error.sum() / denominator)
        if denominator > EPSILON
        else math.nan,
        "under_qty": float(under.sum()),
        "over_qty": float(over.sum()),
        "imbalance_qty": float(under.sum() + over.sum()),
        "demand_satisfaction_pct": float(100.0 * served.sum() / denominator)
        if denominator > EPSILON
        else math.nan,
    }

    if sales_col is not None and sales_col in frame.columns:
        sales_all = pd.to_numeric(frame[sales_col], errors="coerce").to_numpy(
            dtype=float
        )
        sales = sales_all
        valid_sales = np.isfinite(sales)
        if not valid_sales.all():
            raise ValueError("Metrics received missing or non-finite sales")
        hidden = np.maximum(actual - sales, 0.0)
        recognized = np.minimum(np.maximum(predicted - sales, 0.0), hidden)
        hidden_sum = float(hidden[valid_sales].sum())
        closer = absolute < np.abs(sales - actual)
        result.update(
            {
                "restored_lost_qty": hidden_sum,
                "recognized_lost_qty": float(recognized[valid_sales].sum()),
                "recognized_lost_pct": float(
                    100.0 * recognized[valid_sales].sum() / hidden_sum
                )
                if hidden_sum > EPSILON
                else math.nan,
                "closer_to_target_than_sales_rows": int((closer & valid_sales).sum()),
                "closer_to_target_than_sales_pct": float(
                    100.0 * (closer & valid_sales).sum() / valid_sales.sum()
                )
                if valid_sales.any()
                else math.nan,
            }
        )
    return result
