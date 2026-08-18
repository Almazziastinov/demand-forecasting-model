"""Versioned product-scope rules for the bakery pilot.

Product identification is by product_id (integer from the upstream catalogue).
Product names are display attributes only.  This module encodes confirmed
renames and explicit exclusions so that the analytics pipeline can classify
every plan-row match failure with a machine-readable reason instead of
silently dropping rows.

The normalization applied here must be identical to that used in
``pilot_plan_archive.normalize_header`` so that plan-workbook names and
fact-table names are tokenized the same way before comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

PRODUCT_SCOPE_VERSION = "2026-08-12.v1"

COLD_START_PRODUCT_IDS: frozenset[int] = frozenset({11573, 11574})


class MatchType(StrEnum):
    EXACT = "exact"
    ALIAS = "alias"
    UNRESOLVED = "unresolved"


class ExclusionReason(StrEnum):
    FROZEN_SEMIFINISHED = "frozen_semifinished"
    BREAD = "bread"
    CONFECTIONERY = "confectionery"
    NON_BAKEABLE = "non_bakeable"
    COLD_START = "cold_start"
    NOT_IN_BAKING_META = "not_in_baking_meta"
    OUT_OF_SCOPE_CATEGORY = "out_of_scope_category"
    RENAMED_ALIAS_MISSING_FROM_FACTS = "renamed_alias_missing_from_facts"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ConfirmedRename:
    old_name: str
    new_name: str
    confirmed_by: str
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.old_name.strip():
            raise ValueError("old_name must not be empty")
        if not self.new_name.strip():
            raise ValueError("new_name must not be empty")
        if not self.confirmed_by.strip():
            raise ValueError("confirmed_by must not be empty")


@dataclass(frozen=True)
class ExplicitExclusion:
    product_name_pattern: str
    reason: ExclusionReason
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.product_name_pattern.strip():
            raise ValueError("product_name_pattern must not be empty")


# Confirmed catalogue renames observed in 2026-07-23..2026-08-12 plan archive.
# ~180 rows had no forecast match under the old name; product_id is unchanged.
CONFIRMED_RENAMES: tuple[ConfirmedRename, ...] = (
    ConfirmedRename(
        old_name="пирог с грибами и курицей",
        new_name="пирог с курицей и грибами",
        confirmed_by="pilot_management_analytics_2026-08-12",
        notes="180 archived plan rows (2026-07-23..2026-08-12) had no forecast match under the old name; product_id is unchanged by the catalogue rename.",
    ),
)

EXPLICIT_EXCLUSIONS: tuple[ExplicitExclusion, ...] = (
    ExplicitExclusion(
        product_name_pattern="полуфабрикат",
        reason=ExclusionReason.FROZEN_SEMIFINISHED,
        notes="Frozen semi-finished goods; not part of the fresh-bake pilot scope. Accounted for ~551 archive rows in 2026-07-23..2026-08-12.",
    ),
    ExplicitExclusion(
        product_name_pattern="хлеб",
        reason=ExclusionReason.BREAD,
        notes="No stable production/sales data; requires a separate business decision.",
    ),
)


def normalize_product_name(name: object) -> str:
    """Normalize a product name for stable comparison."""
    if name is None:
        return ""
    text = str(name).replace("ё", "е").replace("Ё", "Е")
    return " ".join(text.strip().casefold().split())


# Alias map: normalized old_name → normalized new_name
_ALIAS_MAP: dict[str, str] = {
    normalize_product_name(r.old_name): normalize_product_name(r.new_name)
    for r in CONFIRMED_RENAMES
}

_EXCLUSION_PATTERNS: list[tuple[str, ExclusionReason]] = [
    (normalize_product_name(e.product_name_pattern), e.reason)
    for e in EXPLICIT_EXCLUSIONS
]


def resolve_product_name(
    name: str,
    name_to_id: dict[str, int],
) -> tuple[int | None, MatchType, str | None]:
    """Resolve a product name to its canonical product_id.

    Returns (product_id, match_type, alias_used).
    product_id is None when unresolved.
    """
    key = normalize_product_name(name)
    if key in name_to_id:
        return name_to_id[key], MatchType.EXACT, None
    alias = _ALIAS_MAP.get(key)
    if alias and alias in name_to_id:
        return name_to_id[alias], MatchType.ALIAS, alias
    return None, MatchType.UNRESOLVED, None


def classify_unmatched_reason(name: str) -> ExclusionReason:
    """Classify why a product name could not be matched to a product_id."""
    key = normalize_product_name(name)
    for pattern, reason in _EXCLUSION_PATTERNS:
        if pattern in key:
            return reason
    return ExclusionReason.UNKNOWN
