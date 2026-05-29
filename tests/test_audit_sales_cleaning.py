from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.audit_sales_cleaning import build_holiday_hit_rate  # noqa: E402
from src.analysis.audit_sales_cleaning import build_overall_summary  # noqa: E402
from src.analysis.audit_sales_cleaning import save_audit  # noqa: E402


def _daily() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-01"),
                "bakery_id": 1,
                "bakery_name": "B1",
                "dow": 3,
                "bakery_sales": 100.0,
                "bakery_sales_base_rolling_capped": 80.0,
                "rolling_base_target_capped_flag": 1,
                "rolling_base_target_cap_delta": -20.0,
                "sales_high_outlier_flag": 1,
                "sales_low_outlier_flag": 0,
            },
            {
                "date": pd.Timestamp("2026-01-02"),
                "bakery_id": 1,
                "bakery_name": "B1",
                "dow": 4,
                "bakery_sales": 20.0,
                "bakery_sales_base_rolling_capped": 30.0,
                "rolling_base_target_capped_flag": 1,
                "rolling_base_target_cap_delta": 10.0,
                "sales_high_outlier_flag": 0,
                "sales_low_outlier_flag": 1,
            },
            {
                "date": pd.Timestamp("2026-01-03"),
                "bakery_id": 2,
                "bakery_name": "B2",
                "dow": 5,
                "bakery_sales": 50.0,
                "bakery_sales_base_rolling_capped": 50.0,
                "rolling_base_target_capped_flag": 0,
                "rolling_base_target_cap_delta": 0.0,
                "sales_high_outlier_flag": 0,
                "sales_low_outlier_flag": 0,
            },
        ]
    )


def test_build_overall_summary_counts_caps_and_outliers():
    summary = build_overall_summary(_daily())

    assert summary["rows"] == 3
    assert summary["observed_sales_sum"] == 170.0
    assert summary["base_capped_sum"] == 160.0
    assert summary["cap_delta_sum"] == -10.0
    assert summary["capped_rows"] == 2
    assert summary["high_outlier_rows"] == 1
    assert summary["low_outlier_rows"] == 1


def test_save_audit_writes_expected_files():
    output_dir = Path("reports") / "test_sales_cleaning_audit_tmp"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    paths = save_audit(_daily(), output_dir)

    assert paths["summary"].exists()
    assert paths["bakery_summary"].exists()
    assert paths["date_summary"].exists()
    assert paths["dow_summary"].exists()
    assert paths["top_capped_rows"].exists()

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["capped_rows"] == 2
    shutil.rmtree(output_dir)


def test_build_holiday_hit_rate_flags_known_holiday():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-01",
                    "2026-02-23",
                    "2026-05-09",
                    "2026-05-10",
                ]
            ),
            "sales_high_outlier_flag": [1, 1, 0, 1, 1],
        }
    )

    result = build_holiday_hit_rate(df)

    assert result["applicable"] is True
    assert "2026-01-01" in result["hits"]
    assert "2026-05-09" in result["hits"]
    assert "2026-02-23" not in result["hits"]
    assert 0.0 < result["hit_rate"] <= 1.0
