from __future__ import annotations

import pandas as pd
import pytest

from scripts.evaluate_demand_adjusted_pair_gate import (
    ADJUSTED,
    BASELINE,
    apply_pair_gate,
    build_pair_evidence,
    eligible_pairs,
)


def _scored() -> pd.DataFrame:
    rows = []
    for variant, p10, p20 in [(BASELINE, 8.0, 2.0), (ADJUSTED, 4.0, 6.0)]:
        for product_id, actual, prediction in [(10, 4.0, p10), (20, 6.0, p20)]:
            rows.append(
                {
                    "variant": variant,
                    "date": pd.Timestamp("2026-06-01"),
                    "bakery_id": 1,
                    "dow": 0,
                    "hour": 10,
                    "product_id": product_id,
                    "bakery_hour_sales": 10.0,
                    "actual_qty": actual,
                    "predicted_qty": prediction,
                    "is_stockout_sku_day": False,
                    "is_adjusted_pair": True,
                }
            )
    return pd.DataFrame(rows)


def test_pair_evidence_selects_only_proven_improvement() -> None:
    evidence = build_pair_evidence(_scored())
    allowed = eligible_pairs(evidence, min_clean_days=1, min_gain_qty=1.0)
    assert allowed == {(1, 10), (1, 20)}


def test_pair_gate_renormalizes_context_total() -> None:
    gated = apply_pair_gate(_scored(), {(1, 10)})
    predictions = gated.set_index("product_id")["predicted_qty"]
    assert predictions.sum() == pytest.approx(10.0)
    assert predictions.loc[10] == pytest.approx(10.0 * 4.0 / 6.0)
    assert predictions.loc[20] == pytest.approx(10.0 * 2.0 / 6.0)
