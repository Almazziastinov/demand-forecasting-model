from __future__ import annotations

import pytest

from pipelines.forecast_publish.assortment_override_store import build_override_row


def test_override_row_is_temporary_and_audited() -> None:
    row = build_override_row(
        bakery_id=270,
        product_id=11575,
        action="force_include",
        valid_from="2026-08-20",
        valid_to="2026-08-22",
        reason="source incident",
        created_by="operator",
    ).iloc[0]

    assert row["product_id"] == "000011575"
    assert row["valid_to"].isoformat() == "2026-08-22"
    assert row["reason"] == "source incident"


def test_override_rejects_unbounded_or_reversed_period() -> None:
    with pytest.raises(ValueError, match="on or after"):
        build_override_row(
            bakery_id=270,
            product_id=11575,
            action="force_include",
            valid_from="2026-08-22",
            valid_to="2026-08-20",
            reason="incident",
            created_by="operator",
        )
