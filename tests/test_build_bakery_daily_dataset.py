"""Tests for raw bakery daily dataset building and strict raw-line deduplication."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments_v2.build_bakery_daily_dataset import build_bakery_daily_dataset  # noqa: E402
from src.experiments_v2.build_bakery_daily_dataset import build_summary  # noqa: E402
from src.experiments_v2.build_bakery_daily_dataset import RU_BAKERY_NAME_COL  # noqa: E402
from src.experiments_v2.build_bakery_daily_dataset import RU_CHECK_DATE_COL  # noqa: E402
from src.experiments_v2.build_bakery_daily_dataset import RU_CHECK_DATETIME_COL  # noqa: E402
from src.experiments_v2.build_bakery_daily_dataset import RU_EVENT_COL  # noqa: E402
from src.experiments_v2.build_bakery_daily_dataset import RU_PRICE_COL  # noqa: E402
from src.experiments_v2.build_bakery_daily_dataset import RU_PRODUCT_NAME_COL  # noqa: E402
from src.experiments_v2.build_bakery_daily_dataset import RU_QUANTITY_COL  # noqa: E402
from src.experiments_v2.build_bakery_daily_dataset import SALES_MISSING_FLAG_COL  # noqa: E402
from src.experiments_v2.build_bakery_daily_dataset import TARGET_COL  # noqa: E402
from src.experiments_v2.raw_sales_dedup import deduplicate_sales_chunk  # noqa: E402
from src.experiments_v2.raw_sales_dedup import prepare_sales_chunk  # noqa: E402


SALES_EVENT = "\u041f\u0440\u043e\u0434\u0430\u0436\u0430"


def test_deduplicate_sales_chunk_removes_only_strict_duplicates():
    raw = pd.DataFrame(
        [
            {
                "check_date": "2026-04-21",
                "check_datetime": "2026-04-21 09:00:00",
                "cash_event_type": SALES_EVENT,
                "quantity": 2.0,
                "price": 100.0,
                "line_amount": 200.0,
                "bakery_id": 10,
                "bakery_name": "B1",
                "city": "Kazan",
                "product_id": 101,
            },
            {
                "check_date": "2026-04-21",
                "check_datetime": "2026-04-21 09:00:00",
                "cash_event_type": SALES_EVENT,
                "quantity": 2.0,
                "price": 100.0,
                "line_amount": 200.0,
                "bakery_id": 10,
                "bakery_name": "B1",
                "city": "Kazan",
                "product_id": 101,
            },
            {
                "check_date": "2026-04-21",
                "check_datetime": "2026-04-21 09:00:00",
                "cash_event_type": SALES_EVENT,
                "quantity": 3.0,
                "price": 100.0,
                "line_amount": 300.0,
                "bakery_id": 10,
                "bakery_name": "B1",
                "city": "Kazan",
                "product_id": 101,
            },
        ]
    )

    sales = prepare_sales_chunk(raw, sales_events={SALES_EVENT})
    deduped, stats = deduplicate_sales_chunk(sales)

    assert len(deduped) == 2
    assert stats["removed_rows"] == 1
    assert stats["removed_quantity_sum"] == 2.0
    assert float(deduped["quantity"].sum()) == 5.0


def test_build_bakery_daily_dataset_applies_strict_dedup():
    source = Path("raw_sales_build_bakery_daily_dataset_test.csv")
    raw = pd.DataFrame(
        [
            {
                "check_date": "2026-04-21",
                "check_datetime": "2026-04-21 09:00:00",
                "cash_event_type": SALES_EVENT,
                "quantity": 2.0,
                "price": 100.0,
                "line_amount": 200.0,
                "bakery_id": 10,
                "bakery_name": "B1",
                "city": "Kazan",
                "product_id": 101,
            },
            {
                "check_date": "2026-04-21",
                "check_datetime": "2026-04-21 09:00:00",
                "cash_event_type": SALES_EVENT,
                "quantity": 2.0,
                "price": 100.0,
                "line_amount": 200.0,
                "bakery_id": 10,
                "bakery_name": "B1",
                "city": "Kazan",
                "product_id": 101,
            },
            {
                "check_date": "2026-04-21",
                "check_datetime": "2026-04-21 09:00:00",
                "cash_event_type": SALES_EVENT,
                "quantity": 3.0,
                "price": 100.0,
                "line_amount": 300.0,
                "bakery_id": 10,
                "bakery_name": "B1",
                "city": "Kazan",
                "product_id": 101,
            },
            {
                "check_date": "2026-04-22",
                "check_datetime": "2026-04-22 10:00:00",
                "cash_event_type": SALES_EVENT,
                "quantity": 4.0,
                "price": 90.0,
                "line_amount": 360.0,
                "bakery_id": 10,
                "bakery_name": "B1",
                "city": "Kazan",
                "product_id": 102,
            },
        ]
    )
    raw.to_csv(source, index=False, encoding="utf-8-sig")

    result = build_bakery_daily_dataset(source, chunk_size=2)
    summary = build_summary(result)

    assert len(result) == 2
    assert float(result.iloc[0][TARGET_COL]) == 5.0
    assert float(result.iloc[1][TARGET_COL]) == 4.0
    assert summary["raw_sales_dedup"]["removed_rows"] == 1
    assert summary["raw_sales_dedup"]["removed_quantity_sum"] == 2.0
    assert summary["raw_sales_dedup"]["deduped_rows"] == 3
    source.unlink()


def test_build_bakery_daily_dataset_supports_legacy_russian_columns():
    source = Path("raw_sales_build_bakery_daily_dataset_legacy_test.csv")
    raw = pd.DataFrame(
        [
            {
                RU_CHECK_DATE_COL: "21.04.2026",
                RU_CHECK_DATETIME_COL: "21.04.2026 09:00:00",
                RU_EVENT_COL: SALES_EVENT,
                RU_BAKERY_NAME_COL: "Legacy Bakery",
                RU_PRICE_COL: 120.0,
                RU_QUANTITY_COL: 2.0,
                RU_PRODUCT_NAME_COL: "Bread",
            }
        ]
    )
    raw.to_csv(source, index=False, encoding="utf-8-sig")

    result = build_bakery_daily_dataset(source, chunk_size=1)

    assert len(result) == 1
    assert result.iloc[0]["bakery_id"] == "Legacy Bakery"
    assert result.iloc[0]["city"] == "unknown"
    assert float(result.iloc[0][TARGET_COL]) == 2.0
    source.unlink()


def test_build_bakery_daily_dataset_adds_outlier_cleaning_columns():
    source = Path("raw_sales_build_bakery_daily_dataset_cleaning_test.csv")
    raw = pd.DataFrame(
        [
            {
                "check_date": "2026-04-21",
                "check_datetime": "2026-04-21 09:00:00",
                "cash_event_type": SALES_EVENT,
                "quantity": 10.0,
                "price": 100.0,
                "line_amount": 1000.0,
                "bakery_id": 60,
                "bakery_name": "B60",
                "city": "Kazan",
                "product_id": 101,
            }
        ]
    )
    raw.to_csv(source, index=False, encoding="utf-8-sig")

    result = build_bakery_daily_dataset(source, chunk_size=1)

    assert float(result.iloc[0]["bakery_sales"]) == 10.0
    assert "bakery_sales_base_quantile_capped" in result.columns
    assert "bakery_sales_base_rolling_capped" in result.columns
    assert "base_model_sample_weight" in result.columns
    assert "sales_high_outlier_flag" in result.columns

    source.unlink()


def test_build_bakery_daily_dataset_imputes_missing_calendar_days():
    source = Path("raw_sales_build_bakery_daily_dataset_missing_days_test.csv")
    dates = [
        "2026-01-01",
        "2026-01-08",
        "2026-01-15",
        "2026-01-22",
        "2026-01-29",
        "2026-02-12",
    ]
    rows = []
    for i, date in enumerate(dates):
        rows.append(
            {
                "check_date": date,
                "check_datetime": f"{date} 09:00:00",
                "cash_event_type": SALES_EVENT,
                "quantity": 100.0 + i * 10,
                "price": 100.0,
                "line_amount": (100.0 + i * 10) * 100.0,
                "bakery_id": 70,
                "bakery_name": "B70",
                "city": "Kazan",
                "product_id": 101,
            }
        )
    raw = pd.DataFrame(rows)
    raw.to_csv(source, index=False, encoding="utf-8-sig")

    result = build_bakery_daily_dataset(source, chunk_size=20)
    missing_day = result.loc[result["date"] == pd.Timestamp("2026-02-05")].iloc[0]

    assert len(result) == 43
    assert int(missing_day[SALES_MISSING_FLAG_COL]) == 1
    assert missing_day["sales_imputation_source"] == "rolling_same_dow_median"
    assert pd.isna(missing_day["bakery_sales_observed"])
    assert float(missing_day[TARGET_COL]) == 120.0
    assert float(missing_day["base_model_sample_weight"]) == 0.25

    source.unlink()
