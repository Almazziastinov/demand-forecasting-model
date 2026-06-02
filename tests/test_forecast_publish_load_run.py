from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.forecast_publish import load_forecast_run as module  # noqa: E402


class _FakeClient:
    def __init__(self, lookup: pd.DataFrame | None = None):
        self.lookup = lookup if lookup is not None else pd.DataFrame()
        self.commands: list[str] = []
        self.inserts: list[tuple[str, pd.DataFrame]] = []

    def command(self, statement: str) -> None:
        self.commands.append(statement)

    def query_df(self, query: str):
        return self.lookup.copy()

    def insert_df(self, table: str, df: pd.DataFrame) -> None:
        self.inserts.append((table, df.copy()))


def test_load_product_lookup_from_clickhouse_deduplicates_rows():
    lookup = pd.DataFrame(
        {
            "bakery_id": [1, 1, 1],
            "product_id": [10, 10, 11],
            "product_name": ["A", "A", "B"],
            "category_name": ["Cat", "Cat", "Cat"],
        }
    )
    result = module.load_product_lookup_from_clickhouse(_FakeClient(lookup), "profiles")

    assert len(result) == 2
    assert set(result["product_id"]) == {10, 11}


def test_load_forecast_run_can_use_clickhouse_lookup(monkeypatch):
    work_dir = Path("tests") / "_tmp_forecast_publish"
    work_dir.mkdir(parents=True, exist_ok=True)
    bakery_path = work_dir / "bakery.csv"
    sku_day_path = work_dir / "sku_day.csv"
    sku_hour_path = work_dir / "sku_hour.csv"
    schema_path = work_dir / "schema.sql"

    pd.DataFrame(
        {
            "date": ["2026-06-01"],
            "bakery_id": [1],
            "bakery_name": ["Bakery"],
            "city": ["Kazan"],
            "bakery_day_forecast": [100.0],
            "bakery_day_forecast_bias_adj": [110.0],
        }
    ).to_csv(bakery_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "date": ["2026-06-01"],
            "dow": [0],
            "bakery_id": [1],
            "product_id": [10],
            "sku_day_forecast": [110.0],
        }
    ).to_csv(sku_day_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "date": ["2026-06-01"],
            "dow": [0],
            "hour": [9],
            "bakery_id": [1],
            "product_id": [10],
            "sku_hour_forecast": [110.0],
        }
    ).to_csv(sku_hour_path, index=False, encoding="utf-8-sig")
    schema_path.write_text("create table if not exists x (id Int64);", encoding="utf-8")

    fake = _FakeClient(
        pd.DataFrame(
            {
                "bakery_id": [1],
                "product_id": [10],
                "product_name": ["Product"],
                "category_name": ["Category"],
            }
        )
    )
    monkeypatch.setattr(module, "create_client", lambda env_file: fake)

    result = module.load_forecast_run(
        env_file=work_dir / ".env",
        schema_path=schema_path,
        bakery_path=bakery_path,
        sku_day_path=sku_day_path,
        sku_hour_path=sku_hour_path,
        profile_path=None,
        lookup_source="clickhouse",
        run_id="run_test",
    )

    assert result["run_id"] == "run_test"
    assert result["bakery_rows"] == 1
    assert result["sku_day_rows"] == 1
    assert result["sku_hour_rows"] == 1
    assert [table for table, _ in fake.inserts] == [
        "forecast_runs_embedded",
        "bakery_forecast_day_embedded",
        "sku_forecast_day_embedded",
        "sku_forecast_hour_embedded",
    ]
    sku_day_insert = fake.inserts[2][1].iloc[0]
    assert sku_day_insert["product_name"] == "Product"
    assert sku_day_insert["category_name"] == "Category"
