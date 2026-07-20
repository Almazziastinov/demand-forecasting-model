from __future__ import annotations

import pandas as pd
import pytest

from scripts.export_clickhouse_checks import (
    PILOT_DAILY_REQUIRED_COLUMNS,
    reorder_columns,
    validate_columns,
)


def test_pilot_daily_schema_accepts_daily_export() -> None:
    frame = pd.DataFrame([{column: 1 for column in PILOT_DAILY_REQUIRED_COLUMNS}])
    frame["revenue"] = 10

    validate_columns(frame, PILOT_DAILY_REQUIRED_COLUMNS)
    ordered = reorder_columns(frame, PILOT_DAILY_REQUIRED_COLUMNS)

    assert ordered.columns[: len(PILOT_DAILY_REQUIRED_COLUMNS)].tolist() == (
        PILOT_DAILY_REQUIRED_COLUMNS
    )
    assert ordered.columns[-1] == "revenue"


def test_pilot_daily_schema_rejects_missing_balance_column() -> None:
    columns = [
        column for column in PILOT_DAILY_REQUIRED_COLUMNS if column != "stock_balance"
    ]
    frame = pd.DataFrame([{column: 1 for column in columns}])

    with pytest.raises(ValueError, match="stock_balance"):
        validate_columns(frame, PILOT_DAILY_REQUIRED_COLUMNS)
