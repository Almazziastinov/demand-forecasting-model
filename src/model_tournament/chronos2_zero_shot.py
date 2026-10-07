"""Research-only Chronos-2 adapter for the fixed-origin observed-sales panel.

No model package is imported here: tests use a fake pipeline, and the CLI loads
Chronos only when an actual inference run is requested. Missing historical
SKU-days follow the existing tournament convention and are treated as zero.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from src.model_tournament.fixed_origin import KEYS, build_fixed_origin_panel


MODEL_ID = "chronos2_zero_shot_p50"
CONTEXT_DAYS = 56
PREDICTION_COLUMN = "0.5"


class Chronos2Like(Protocol):
    def predict_df(
        self, context_df: pd.DataFrame, **kwargs: object
    ) -> pd.DataFrame: ...


def build_chronos2_context(
    flows: pd.DataFrame,
    origin: str | pd.Timestamp,
    *,
    context_days: int = CONTEXT_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return dense, causal daily histories and the eligible pair-to-ID map."""
    if context_days < 56:
        raise ValueError("Context must include the 56-day eligibility window")
    as_of = pd.Timestamp(origin).normalize()
    start = as_of - pd.Timedelta(days=context_days - 1)
    history = flows.loc[flows["date"].between(start, as_of)].copy()
    eligible = (
        history.loc[history["date"].ge(as_of - pd.Timedelta(days=55))]
        .groupby(KEYS, as_index=False)["release_qty"]
        .sum()
        .loc[lambda frame: frame["release_qty"].gt(0), KEYS]
        .sort_values(KEYS)
        .reset_index(drop=True)
    )
    if eligible.empty:
        raise ValueError("No pairs had positive release in the prior 56 days")
    eligible["id"] = np.arange(len(eligible), dtype=np.int64).astype(str)
    dates = pd.DataFrame({"timestamp": pd.date_range(start, as_of)})
    context = eligible.merge(dates, how="cross")
    known = history[["date", *KEYS, "observed_sales_qty"]].rename(
        columns={"date": "timestamp", "observed_sales_qty": "target"}
    )
    context = context.merge(
        known, on=["timestamp", *KEYS], how="left", validate="one_to_one"
    )
    context["target"] = context["target"].fillna(0.0)
    return context[["id", "timestamp", "target"]], eligible[[*KEYS, "id"]]


def predict_chronos2_origin(
    flows: pd.DataFrame,
    origin: str | pd.Timestamp,
    pipeline: Chronos2Like,
    *,
    horizon_days: int = 14,
    context_days: int = CONTEXT_DAYS,
    batch_series: int = 128,
) -> pd.DataFrame:
    """Score every fixed-origin SKU-day using Chronos-2's zero-shot median.

    Future facts are used only by the existing evaluation panel to attach
    labels. Only its keys, never its ``actual`` values, enter prediction input.
    """
    if horizon_days < 1 or batch_series < 1:
        raise ValueError("horizon_days and batch_series must be positive")
    as_of = pd.Timestamp(origin).normalize()
    evaluation = build_fixed_origin_panel(flows, as_of, horizon_days=horizon_days)
    context, id_map = build_chronos2_context(flows, as_of, context_days=context_days)
    expected = evaluation[
        ["date", *KEYS, "forecast_origin", "lead_days", "actual"]
    ].merge(id_map, on=KEYS, how="left", validate="many_to_one")
    if expected["id"].isna().any():
        raise ValueError("Chronos scope differs from fixed-origin evaluation scope")

    parts = []
    ids = id_map["id"].tolist()
    for offset in range(0, len(ids), batch_series):
        batch_ids = ids[offset : offset + batch_series]
        batch_context = context.loc[context["id"].isin(batch_ids)]
        forecast = pipeline.predict_df(
            batch_context,
            prediction_length=horizon_days,
            quantile_levels=[0.5],
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )
        required = {"id", "timestamp", PREDICTION_COLUMN}
        if not required.issubset(forecast.columns):
            missing = sorted(required - set(forecast.columns))
            raise ValueError(
                f"Chronos output lacks columns: {missing}"
            )
        part = forecast[["id", "timestamp", PREDICTION_COLUMN]].copy()
        part["id"] = part["id"].astype(str)
        part["date"] = pd.to_datetime(part["timestamp"], errors="raise").dt.normalize()
        parts.append(part[["id", "date", PREDICTION_COLUMN]])

    predictions = pd.concat(parts, ignore_index=True)
    if predictions.duplicated(["id", "date"]).any():
        raise ValueError("Chronos returned duplicate SKU-day predictions")
    result = expected.merge(
        predictions, on=["id", "date"], how="left", validate="one_to_one"
    )
    if result[PREDICTION_COLUMN].isna().any() or len(predictions) != len(expected):
        raise ValueError("Chronos predictions do not exactly cover the evaluation rows")
    values = pd.to_numeric(result[PREDICTION_COLUMN], errors="raise").to_numpy(
        dtype=float
    )
    if not np.isfinite(values).all():
        raise ValueError("Chronos produced non-finite predictions")
    result["prediction"] = np.maximum(values, 0.0)
    result["model"] = MODEL_ID
    return result[
        ["forecast_origin", "date", "lead_days", *KEYS, "actual", "model", "prediction"]
    ]
