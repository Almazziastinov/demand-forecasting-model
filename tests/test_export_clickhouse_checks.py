from __future__ import annotations

import pandas as pd
import pytest

from scripts.export_clickhouse_checks import (
    PILOT_DAILY_REQUIRED_COLUMNS,
    export_windows,
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


def test_export_retries_failed_window_without_dropping_progress(tmp_path) -> None:
    class Client:
        calls = 0

        def query_df(self, sql):
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("temporary")
            return pd.DataFrame({"value": [1]})

    client = Client()
    output = tmp_path / "export.csv"
    export_windows(
        client=client,
        sql_template_text="select 1 from '{date_from}' to '{date_to}' {limit_clause}",
        output_path=output,
        date_from="2026-07-01",
        date_to="2026-07-01",
        batch_mode="single",
        limit=None,
        required_columns=["value"],
        query_attempts=2,
        retry_seconds=0,
    )

    assert client.calls == 2
    assert pd.read_csv(output)["value"].tolist() == [1]
