from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.export_clickhouse_bakery_daily import export_daily_windows


class _FakeClient:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.queries: list[str] = []

    def query_df(self, query: str) -> pd.DataFrame:
        self.queries.append(query)
        return self.frame.copy()


def test_export_daily_windows_writes_required_columns():
    frame = pd.DataFrame(
        {
            "date": ["2026-06-10"],
            "bakery_id": [1],
            "bakery_name": ["Bakery"],
            "city": ["Kazan"],
            "bakery_sales": [100.0],
            "line_amount_sum": [1000.0],
            "priced_quantity": [100.0],
            "price_x_qty_sum": [1000.0],
        }
    )
    work_dir = Path("tests") / "_tmp_export_clickhouse_bakery_daily"
    work_dir.mkdir(parents=True, exist_ok=True)
    output = work_dir / "bakery_daily.csv"

    result = export_daily_windows(
        client=_FakeClient(frame),
        sql_template_text=(
            "select * where d between '{date_from}' and '{date_to}' "
            "{limit_clause}"
        ),
        output_path=output,
        date_from="2026-06-10",
        date_to="2026-06-10",
        batch_mode="single",
        limit=None,
    )

    exported = pd.read_csv(output, encoding="utf-8-sig")
    assert result["rows"] == 1
    assert exported["bakery_sales"].tolist() == [100.0]
    assert exported.columns.tolist()[:8] == [
        "date",
        "bakery_id",
        "bakery_name",
        "city",
        "bakery_sales",
        "line_amount_sum",
        "priced_quantity",
        "price_x_qty_sum",
    ]
    output.unlink()
