"""Canonical definitions for bakery pilot scopes and product eligibility.

The module intentionally contains only scopes whose complete membership is
known.  A scope must pass validation before it can be added to ``PILOT_SCOPES``.
"""

from __future__ import annotations

from dataclasses import dataclass


PILOT_SCOPE_RULES_VERSION = "2026-08-11.v1"

BASE_PILOT_CATEGORY_NAMES = frozenset(
    {
        "Пироги сытные",
        "Пироги сладкие",
        "Выпечка сытная",
        "Выпечка сладкая",
        "Фастфуд",
    }
)


@dataclass(frozen=True)
class PilotExclusion:
    """A bakery deliberately excluded from a pilot scope."""

    bakery_id: int
    bakery_name: str
    reason: str

    def __post_init__(self) -> None:
        if self.bakery_id <= 0:
            raise ValueError("Excluded bakery_id must be positive")
        if not self.bakery_name.strip():
            raise ValueError("Excluded bakery_name must not be empty")
        if not self.reason.strip():
            raise ValueError("Exclusion reason must not be empty")


@dataclass(frozen=True)
class PilotScope:
    """Validated membership and category rules for one named pilot scope."""

    name: str
    bakery_ids: frozenset[int]
    category_names: frozenset[str]
    exclusions: tuple[PilotExclusion, ...] = ()
    rules_version: str = PILOT_SCOPE_RULES_VERSION

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Pilot scope name must not be empty")
        if not self.rules_version.strip():
            raise ValueError("Pilot scope rules_version must not be empty")
        if not self.bakery_ids:
            raise ValueError("Pilot scope must contain at least one bakery")
        if any(bakery_id <= 0 for bakery_id in self.bakery_ids):
            raise ValueError("Pilot bakery_ids must be positive")
        if not self.category_names:
            raise ValueError("Pilot scope must contain at least one category")
        if any(not category.strip() for category in self.category_names):
            raise ValueError("Pilot category names must not be empty")

        excluded_ids = [exclusion.bakery_id for exclusion in self.exclusions]
        if len(excluded_ids) != len(set(excluded_ids)):
            raise ValueError("Pilot exclusions must have unique bakery_ids")
        overlap = self.bakery_ids.intersection(excluded_ids)
        if overlap:
            raise ValueError(
                "Excluded bakeries cannot belong to the scope: "
                f"{sorted(overlap)}"
            )


BASE_PILOT_10 = PilotScope(
    name="base_pilot_10",
    bakery_ids=frozenset({20, 21, 22, 28, 80, 89, 107, 221, 222, 257}),
    category_names=BASE_PILOT_CATEGORY_NAMES,
    exclusions=(
        PilotExclusion(
            bakery_id=16,
            bakery_name="Кулагина 4 Казань",
            reason=(
                "Исключена из базового пилота до отдельного решения "
                "о повторном включении."
            ),
        ),
    ),
)

PILOT_SCOPES: dict[str, PilotScope] = {BASE_PILOT_10.name: BASE_PILOT_10}


def get_pilot_scope(name: str = BASE_PILOT_10.name) -> PilotScope:
    """Return a known pilot scope or raise a descriptive error."""

    try:
        return PILOT_SCOPES[name]
    except KeyError as exc:
        available = ", ".join(sorted(PILOT_SCOPES))
        raise KeyError(
            f"Unknown pilot scope {name!r}; available scopes: {available}"
        ) from exc


def is_bakery_in_scope(bakery_id: int, scope: PilotScope = BASE_PILOT_10) -> bool:
    """Return whether ``bakery_id`` belongs to ``scope``."""

    return bakery_id in scope.bakery_ids


def is_category_in_scope(category_name: str, scope: PilotScope = BASE_PILOT_10) -> bool:
    """Return whether ``category_name`` is eligible for ``scope``."""

    return category_name in scope.category_names
