"""Tests for the n_days gate on the SKU hour-share fallback chain."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from uuid import uuid4

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments_v2.apply_bakery_profiles import MIN_TIER1_N_DAYS  # noqa: E402
from src.experiments_v2.apply_bakery_profiles import apply_profiles  # noqa: E402
from src.experiments_v2.apply_bakery_profiles import build_tier1_share_sums  # noqa: E402
from src.experiments_v2.apply_bakery_profiles import _finalize_source_stats  # noqa: E402
from src.experiments_v2.apply_bakery_profiles import _update_source_stats  # noqa: E402
from src.experiments_v2.apply_bakery_profiles import normalize_tier1_sku_shares  # noqa: E402


def _scaffold_inputs(
    tmp_path: Path,
    *,
    sku_rows: list[dict],
) -> tuple[Path, Path, Path]:
    bakery_forecast_path = tmp_path / "bakery_daily_sales.csv"
    bakery_hour_profile_path = tmp_path / "bakery_hour_profile.csv"
    sku_hour_profile_path = tmp_path / "sku_hour_share_profile.csv"

    pd.DataFrame(
        [
            {
                "date": "2026-01-05",
                "bakery_id": "B1",
                "bakery_name": "Bakery 1",
                "bakery_sales": 100.0,
            }
        ]
    ).to_csv(bakery_forecast_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "bakery_id": "B1",
                "bakery_name": "Bakery 1",
                "dow": 0,
                "hour": 8,
                "mean_hour_share_norm": 1.0,
            }
        ]
    ).to_csv(bakery_hour_profile_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(sku_rows).to_csv(
        sku_hour_profile_path, index=False, encoding="utf-8-sig"
    )
    return bakery_forecast_path, bakery_hour_profile_path, sku_hour_profile_path


def _run(tmp_path: Path, sku_rows: list[dict]) -> pd.DataFrame:
    bp, bhp, shp = _scaffold_inputs(tmp_path, sku_rows=sku_rows)
    paths = apply_profiles(bp, bhp, shp, tmp_path)
    return pd.read_csv(paths["sku_hourly"], encoding="utf-8-sig")


def _tmp_dir() -> Path:
    path = Path("tests") / "_tmp_apply_profiles" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_tier1_match_when_n_days_meets_gate():
    tmp = _tmp_dir()
    sku_rows = [
        {
            "bakery_id": "B1",
            "bakery_name": "Bakery 1",
            "product_id": "P1",
            "product_name": "P1",
            "category_name": "Cat",
            "dow": 0,
            "hour": 8,
            "mean_sku_share_in_hour_norm": 1.0,
            "n_days": MIN_TIER1_N_DAYS,
        }
    ]
    sku_hourly = _run(tmp, sku_rows)

    assert len(sku_hourly) == 1
    assert sku_hourly.iloc[0]["source"] == "exact"
    assert round(float(sku_hourly["sku_hour_forecast"].sum()), 4) == 100.0


def test_below_gate_with_tier2_match_routes_to_fallback_thin():
    tmp = _tmp_dir()
    # Thin row at (B1, dow=0, hour=8): below gate. Tier-2 builds an average
    # over (B1, hour=8) from the same profile, so we get a fallback hit.
    sku_rows = [
        {
            "bakery_id": "B1",
            "bakery_name": "Bakery 1",
            "product_id": "P1",
            "product_name": "P1",
            "category_name": "Cat",
            "dow": 0,
            "hour": 8,
            "mean_sku_share_in_hour_norm": 1.0,
            "n_days": MIN_TIER1_N_DAYS - 4,
        }
    ]
    sku_hourly = _run(tmp, sku_rows)

    assert len(sku_hourly) == 1
    assert sku_hourly.iloc[0]["source"] == "bakery_hour_fallback_thin"


def test_below_gate_with_no_tier2_match_emits_no_row():
    tmp = _tmp_dir()
    # The only row in the profile is thin AND at a different hour from the
    # forecast triple, so neither tier-1 nor tier-2 can match.
    sku_rows = [
        {
            "bakery_id": "B1",
            "bakery_name": "Bakery 1",
            "product_id": "P1",
            "product_name": "P1",
            "category_name": "Cat",
            "dow": 0,
            "hour": 17,
            "mean_sku_share_in_hour_norm": 1.0,
            "n_days": MIN_TIER1_N_DAYS - 4,
        }
    ]
    sku_hourly = _run(tmp, sku_rows)

    assert len(sku_hourly) == 0


def test_legacy_profile_without_n_days_column_falls_through_gate():
    tmp = _tmp_dir()
    sku_rows = [
        {
            "bakery_id": "B1",
            "bakery_name": "Bakery 1",
            "product_id": "P1",
            "product_name": "P1",
            "category_name": "Cat",
            "dow": 0,
            "hour": 8,
            "mean_sku_share_in_hour_norm": 1.0,
        }
    ]
    sku_hourly = _run(tmp, sku_rows)

    # n_days missing -> default 0 -> below gate. Tier-2 still has the row, so
    # we get a "thin" fallback rather than dropping the prediction.
    assert len(sku_hourly) == 1
    assert sku_hourly.iloc[0]["source"] == "bakery_hour_fallback_thin"


def test_tier1_shares_are_renormalized_after_n_days_gate():
    profile = pd.DataFrame(
        [
            {
                "bakery_id": "B1",
                "dow": 0,
                "hour": 8,
                "product_id": "P1",
                "mean_sku_share_in_hour_norm": 0.30,
                "n_days": MIN_TIER1_N_DAYS,
            },
            {
                "bakery_id": "B1",
                "dow": 0,
                "hour": 8,
                "product_id": "P2",
                "mean_sku_share_in_hour_norm": 0.20,
                "n_days": MIN_TIER1_N_DAYS,
            },
            {
                "bakery_id": "B1",
                "dow": 0,
                "hour": 8,
                "product_id": "P3",
                "mean_sku_share_in_hour_norm": 0.50,
                "n_days": MIN_TIER1_N_DAYS - 1,
            },
        ]
    )
    sums = build_tier1_share_sums(profile)
    tier1 = profile[profile["n_days"] >= MIN_TIER1_N_DAYS]

    normalized = normalize_tier1_sku_shares(tier1, sums)

    assert round(float(normalized["mean_sku_share_in_hour_norm"].sum()), 6) == 1.0
    p1_share = normalized.loc[
        normalized["product_id"] == "P1",
        "mean_sku_share_in_hour_norm",
    ].iloc[0]
    assert round(float(p1_share), 6) == 0.6


def test_source_stats_track_rows_and_forecast_share():
    stats: dict[str, dict[str, float | int]] = {}
    df = pd.DataFrame(
        [
            {"source": "exact", "sku_hour_forecast": 30.0},
            {"source": "exact", "sku_hour_forecast": 10.0},
            {"source": "bakery_hour_fallback_thin", "sku_hour_forecast": 60.0},
        ]
    )

    _update_source_stats(stats, df)
    summary = _finalize_source_stats(stats)

    exact = next(row for row in summary if row["source"] == "exact")
    fallback = next(
        row for row in summary if row["source"] == "bakery_hour_fallback_thin"
    )
    assert exact["rows"] == 2
    assert exact["row_share"] == 0.666667
    assert exact["forecast_share"] == 0.4
    assert fallback["forecast_sum"] == 60.0
