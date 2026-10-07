from __future__ import annotations

import pandas as pd
import pytest

from scripts.build_tournament_causal_scope_panel import build_panel
from src.model_tournament.scope import (
    SCOPE_KEYS,
    calendar_prior_release_56,
    snapshot_timing_audit,
)


def test_calendar_prior_release_excludes_stale_observed_row() -> None:
    rows = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-04-01", "2026-04-02", "2026-04-03"],
            "bakery_id": [1, 1, 1, 2],
            "product_id": [10, 10, 10, 10],
            "release_qty": [5.0, 0.0, 0.0, 9.0],
        }
    )
    result = calendar_prior_release_56(rows)
    prior = result.set_index(["bakery_id", "product_id", "date"])
    april_1 = prior.loc[(1, 10, pd.Timestamp("2026-04-01"))]
    april_2 = prior.loc[(1, 10, pd.Timestamp("2026-04-02"))]
    other_pair = prior.loc[(2, 10, pd.Timestamp("2026-04-03"))]
    assert pd.isna(april_1["prior_release_56_calendar"])
    assert april_2["prior_release_56_calendar"] == 0.0
    assert pd.isna(other_pair["prior_release_56_calendar"])


def test_calendar_prior_release_includes_exact_left_boundary_only() -> None:
    rows = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-02-25", "2026-02-26"],
            "bakery_id": [1, 1, 1],
            "product_id": [10, 10, 10],
            "release_qty": [5.0, 3.0, 0.0],
        }
    )
    result = calendar_prior_release_56(rows)
    prior = result.set_index("date")["prior_release_56_calendar"]
    assert prior.loc[pd.Timestamp("2026-02-26")] == pytest.approx(8.0)


def test_calendar_prior_release_is_keyed_after_grouped_rolling() -> None:
    rows = pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-02", "2026-01-01", "2026-01-01"],
            "bakery_id": [2, 1, 2, 1],
            "product_id": [10, 10, 10, 10],
            "release_qty": [0.0, 0.0, 7.0, 3.0],
        }
    )
    result = calendar_prior_release_56(rows).set_index(SCOPE_KEYS)
    january_2 = pd.Timestamp("2026-01-02")
    assert result.loc[(january_2, 1, 10), "prior_release_56_calendar"] == 3.0
    assert result.loc[(january_2, 2, 10), "prior_release_56_calendar"] == 7.0


def test_snapshot_timing_flags_record_after_morning_cutoff() -> None:
    predictions = pd.DataFrame(
        {
            "date": ["2026-08-01", "2026-08-01"],
            "bakery_id": [1, 1],
            "product_id": [10, 11],
            "scope_source": ["lead1_snapshot", "lead1_snapshot"],
        }
    )
    snapshots = pd.DataFrame(
        {
            "forecast_date": ["2026-08-01", "2026-08-01"],
            "bakery_id": [1, 1],
            "product_id": [10, 11],
            "generated_at": ["2026-08-01T04:55:00Z", "2026-08-01T06:00:00Z"],
        }
    )
    summary = snapshot_timing_audit(predictions, snapshots)
    assert summary.iloc[0]["selected_rows"] == 2
    assert summary.iloc[0]["record_after_08_msk"] == 1


def test_snapshot_free_panel_drops_stale_production() -> None:
    flows = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-04-01", "2026-04-02"],
            "bakery_id": [1, 1, 1],
            "product_id": [10, 10, 10],
            "release_qty": [5.0, 0.0, 0.0],
        }
    )
    source = flows[SCOPE_KEYS].copy()
    source["product_name"] = "Bread"
    source["category_name"] = "Bakery"
    source["observed_sales_qty"] = 0.0
    source["demand"] = 0.0
    panel = build_panel(source, flows)
    assert panel.empty


def test_snapshot_free_panel_keeps_zero_activity_forecast_day() -> None:
    flows = pd.DataFrame(
        {
            "date": ["2026-01-01"],
            "bakery_id": [1],
            "product_id": [10],
            "release_qty": [5.0],
        }
    )
    source = pd.DataFrame(
        {
            "date": ["2026-01-02"],
            "bakery_id": [1],
            "product_id": [10],
            "product_name": ["Bread"],
            "category_name": ["Bakery"],
            "observed_sales_qty": [0.0],
            "demand": [0.0],
        }
    )
    panel = build_panel(source, flows)
    assert len(panel) == 1
    assert panel.iloc[0]["prior_release_56_calendar"] == 5.0
