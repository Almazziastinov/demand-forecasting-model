"""Tests for the artifact-based demo store."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_demo_store_loads_and_builds_series():
    from web.demo_artifact_store import build_series, load_store

    store = load_store()

    assert "avg_lag_7_14_core" in store["models"]
    assert store["wide"].shape[0] > 0

    sample = store["best_by_sku"].dropna(subset=["bakery", "product"]).iloc[0]
    payload = build_series(sample["bakery"], sample["product"], "avg_lag_7_14_core", "avg_lag_7_14_core")

    assert payload["series"]
    assert payload["metrics"]["main"]["n_rows"] > 0
    assert payload["metrics"]["compare"]["n_rows"] > 0

