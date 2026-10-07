"""Research-only N-HiTS adapter for the fixed-origin sales tournament.

This module deliberately does not import NeuralForecast at module load time.
The neural stack is optional and is never part of the production forecast path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.model_tournament.fixed_origin import KEYS, build_fixed_origin_panel


MODEL_ID = "new_nhits_univariate_h14"
INPUT_DAYS = 56


def _series_ids(frame: pd.DataFrame) -> pd.Series:
    return frame["bakery_id"].astype(str) + ":" + frame["product_id"].astype(str)


def dense_history(
    flows: pd.DataFrame,
    pairs: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Build a regular daily sales history using only facts through ``end``.

    Missing sparse fact rows are explicitly interpreted as zero observed sales,
    the same convention as the fixed-origin tournament's future truth.
    """
    if end < start:
        raise ValueError("History end precedes start")
    distinct = pairs[KEYS].drop_duplicates().copy()
    if distinct.empty:
        raise ValueError("No bakery-product pairs supplied")
    dates = pd.DataFrame({"ds": pd.date_range(start, end, freq="D")})
    grid = distinct.merge(dates, how="cross")
    observed = flows.loc[
        flows["date"].between(start, end), ["date", *KEYS, "observed_sales_qty"]
    ].rename(columns={"date": "ds", "observed_sales_qty": "y"})
    history = grid.merge(observed, on=[*KEYS, "ds"], how="left", validate="one_to_one")
    history["y"] = history["y"].fillna(0.0).astype(float)
    history["unique_id"] = _series_ids(history)
    if not np.isfinite(history["y"].to_numpy()).all() or (history["y"] < 0).any():
        raise ValueError("Training history contains invalid sales")
    return (
        history[["unique_id", "ds", "y"]]
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )


def training_history(
    flows: pd.DataFrame,
    train_origins: list[pd.Timestamp],
    *,
    start: pd.Timestamp,
    input_days: int = INPUT_DAYS,
    horizon_days: int = 14,
    max_pairs: int | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Choose train-scope pairs causally and build regular series through the cutoff."""
    if not train_origins:
        raise ValueError("At least one training origin is required")
    cutoff = max(train_origins) + pd.Timedelta(days=horizon_days)
    pair_parts = [
        build_fixed_origin_panel(flows, origin, horizon_days=horizon_days)[KEYS]
        for origin in train_origins
    ]
    pairs = pd.concat(pair_parts, ignore_index=True).drop_duplicates().sort_values(KEYS)
    if max_pairs is not None:
        if max_pairs < 1:
            raise ValueError("max_pairs must be positive")
        pairs = pairs.head(max_pairs)
    release = flows.loc[
        flows["date"].between(start, cutoff) & flows["release_qty"].gt(0),
        ["date", *KEYS],
    ]
    first_release = release.groupby(KEYS, as_index=False)["date"].min()
    first_release = pairs.merge(
        first_release, on=KEYS, how="inner", validate="one_to_one"
    )
    first_release = first_release.loc[
        first_release["date"]
        <= cutoff - pd.Timedelta(days=input_days + horizon_days - 1)
    ]
    if first_release.empty:
        raise ValueError("No training pair has enough post-release history")
    dense = dense_history(flows, first_release[KEYS], start, cutoff)
    first_release["unique_id"] = _series_ids(first_release)
    dense = dense.merge(
        first_release[["unique_id", "date"]], on="unique_id", validate="many_to_one"
    )
    dense = dense.loc[dense["ds"] >= dense["date"], ["unique_id", "ds", "y"]]
    return dense.reset_index(drop=True), {
        "scope_pairs": len(pairs),
        "eligible_train_pairs": len(first_release),
        "train_rows": len(dense),
    }


def evaluation_history_and_rows(
    flows: pd.DataFrame,
    origin: pd.Timestamp,
    *,
    horizon_days: int = 14,
    max_pairs: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Freeze evaluation scope at origin and provide exactly 56 context days."""
    rows = build_fixed_origin_panel(flows, origin, horizon_days=horizon_days)
    if max_pairs is not None:
        if max_pairs < 1:
            raise ValueError("max_pairs must be positive")
        pairs = rows[KEYS].drop_duplicates().sort_values(KEYS).head(max_pairs)
        rows = rows.merge(pairs, on=KEYS, how="inner", validate="many_to_one")
    context = dense_history(
        flows,
        rows[KEYS],
        origin - pd.Timedelta(days=INPUT_DAYS - 1),
        origin,
    )
    return context, rows


def align_predictions(rows: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    """Fail closed if NeuralForecast output is not the exact frozen SKU-day scope."""
    required = {"unique_id", "ds", "NHITS"}
    if not required.issubset(predicted.columns):
        raise ValueError(
            f"N-HiTS output is missing {sorted(required - set(predicted.columns))}"
        )
    expected = rows[["forecast_origin", "date", "lead_days", *KEYS, "actual"]].copy()
    expected["unique_id"] = _series_ids(expected)
    if predicted.duplicated(["unique_id", "ds"]).any():
        raise ValueError("N-HiTS output has duplicate series-day keys")
    if len(predicted) != len(expected):
        raise ValueError("N-HiTS output does not match frozen evaluation scope")
    joined = expected.merge(
        predicted[["unique_id", "ds", "NHITS"]],
        left_on=["unique_id", "date"],
        right_on=["unique_id", "ds"],
        how="left",
        validate="one_to_one",
    )
    values = pd.to_numeric(joined["NHITS"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("N-HiTS output has missing or non-finite predictions")
    detail = joined[["forecast_origin", "date", "lead_days", *KEYS, "actual"]].copy()
    detail["model"] = MODEL_ID
    detail["prediction"] = np.maximum(values, 0.0)
    return detail
