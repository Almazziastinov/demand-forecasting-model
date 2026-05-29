from __future__ import annotations

import pandas as pd


def add_planning_error_columns(
    df: pd.DataFrame,
    *,
    actual_col: str,
    prediction_col: str,
    abs_error_threshold: float = 50.0,
    rel_error_threshold: float = 0.20,
) -> pd.DataFrame:
    """Add operational error columns used by planning experiments.

    `large_error_flag` follows the Yandex Lavka-style criterion: absolute
    error is material in units and relative error is material as a share of
    demand. The relative denominator is clipped to 1 to keep sparse rows from
    exploding.
    """
    work = df.copy()
    actual = pd.to_numeric(work[actual_col], errors="coerce").fillna(0.0)
    prediction = pd.to_numeric(work[prediction_col], errors="coerce").fillna(0.0)

    work["error"] = actual - prediction
    work["abs_error"] = work["error"].abs()
    work["rel_abs_error"] = work["abs_error"] / actual.abs().clip(lower=1.0)
    work["large_error_flag"] = (
        (work["abs_error"] > abs_error_threshold)
        & (work["rel_abs_error"] > rel_error_threshold)
    ).astype(int)
    work["underforecast_flag"] = (work["error"] > 0).astype(int)
    work["overforecast_flag"] = (work["error"] < 0).astype(int)
    return work


def planning_metrics(
    df: pd.DataFrame,
    *,
    actual_col: str,
    prediction_col: str,
    abs_error_threshold: float = 50.0,
    rel_error_threshold: float = 0.20,
) -> dict[str, float | int]:
    """Return planning metrics for one frame or one aggregated group."""
    if df.empty:
        return {
            "rows": 0,
            "actual_sum": 0.0,
            "prediction_sum": 0.0,
            "mae": 0.0,
            "wmape": 0.0,
            "bias": 0.0,
            "bias_pct": 0.0,
            "large_error_rows": 0,
            "large_error_share": 0.0,
            "large_underforecast_rows": 0,
            "large_overforecast_rows": 0,
        }

    work = add_planning_error_columns(
        df,
        actual_col=actual_col,
        prediction_col=prediction_col,
        abs_error_threshold=abs_error_threshold,
        rel_error_threshold=rel_error_threshold,
    )
    actual = pd.to_numeric(work[actual_col], errors="coerce").fillna(0.0)
    prediction = pd.to_numeric(work[prediction_col], errors="coerce").fillna(0.0)
    actual_sum = float(actual.sum())
    prediction_sum = float(prediction.sum())
    abs_error_sum = float(work["abs_error"].sum())
    bias = actual_sum - prediction_sum
    denominator = abs(actual_sum) if abs(actual_sum) > 1e-9 else 1.0
    large_mask = work["large_error_flag"] == 1

    return {
        "rows": int(len(work)),
        "actual_sum": round(actual_sum, 6),
        "prediction_sum": round(prediction_sum, 6),
        "mae": round(float(work["abs_error"].mean()), 6),
        "wmape": round(abs_error_sum / denominator * 100.0, 6),
        "bias": round(bias, 6),
        "bias_pct": round(bias / denominator * 100.0, 6),
        "large_error_rows": int(large_mask.sum()),
        "large_error_share": round(float(large_mask.mean()), 6),
        "large_underforecast_rows": int(
            (large_mask & (work["underforecast_flag"] == 1)).sum()
        ),
        "large_overforecast_rows": int(
            (large_mask & (work["overforecast_flag"] == 1)).sum()
        ),
    }


def aggregate_planning_metrics(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    actual_col: str,
    prediction_col: str,
    abs_error_threshold: float = 50.0,
    rel_error_threshold: float = 0.20,
) -> pd.DataFrame:
    """Compute planning metrics by aggregate level, e.g. city/category/day."""
    rows: list[dict[str, object]] = []
    if df.empty:
        empty_metrics = planning_metrics(
            df,
            actual_col=actual_col,
            prediction_col=prediction_col,
        )
        return pd.DataFrame(columns=[*group_cols, *empty_metrics.keys()])

    for key, group in df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row.update(
            planning_metrics(
                group,
                actual_col=actual_col,
                prediction_col=prediction_col,
                abs_error_threshold=abs_error_threshold,
                rel_error_threshold=rel_error_threshold,
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_models_by_planning_metrics(
    df: pd.DataFrame,
    *,
    model_col: str,
    actual_col: str,
    prediction_col: str,
    abs_error_threshold: float = 50.0,
    rel_error_threshold: float = 0.20,
) -> pd.DataFrame:
    """One-row-per-model summary for forecast comparison tables."""
    summary = aggregate_planning_metrics(
        df,
        group_cols=[model_col],
        actual_col=actual_col,
        prediction_col=prediction_col,
        abs_error_threshold=abs_error_threshold,
        rel_error_threshold=rel_error_threshold,
    )
    if summary.empty:
        return summary
    sort_cols = ["large_error_share", "wmape", "mae"]
    return summary.sort_values(sort_cols).reset_index(drop=True)
