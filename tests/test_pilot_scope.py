import pytest

from src.pilot_scope import (
    BASE_PILOT_10,
    PILOT_SCOPE_RULES_VERSION,
    PILOT_SCOPES,
    PilotExclusion,
    PilotScope,
    get_pilot_scope,
    is_bakery_in_scope,
    is_category_in_scope,
)


def test_base_pilot_10_has_canonical_membership() -> None:
    assert BASE_PILOT_10.bakery_ids == frozenset(
        {20, 21, 22, 28, 80, 89, 107, 221, 222, 257}
    )
    assert BASE_PILOT_10.rules_version == PILOT_SCOPE_RULES_VERSION


def test_kulagina_is_explicitly_excluded() -> None:
    exclusion = BASE_PILOT_10.exclusions[0]

    assert exclusion.bakery_id == 16
    assert exclusion.bakery_name == "Кулагина 4 Казань"
    assert exclusion.reason
    assert not is_bakery_in_scope(16)


def test_base_pilot_categories_match_production_plan_scope() -> None:
    assert BASE_PILOT_10.category_names == frozenset(
        {
            "Пироги сытные",
            "Пироги сладкие",
            "Выпечка сытная",
            "Выпечка сладкая",
            "Фастфуд",
        }
    )
    assert is_category_in_scope("Выпечка сытная")
    assert not is_category_in_scope("Напитки")


def test_only_verified_scope_is_registered() -> None:
    assert PILOT_SCOPES == {"base_pilot_10": BASE_PILOT_10}
    assert get_pilot_scope() is BASE_PILOT_10
    with pytest.raises(KeyError, match="Unknown pilot scope"):
        get_pilot_scope("expanded_pilot_38")


def test_scope_rejects_member_exclusion_overlap() -> None:
    with pytest.raises(ValueError, match="Excluded bakeries cannot belong"):
        PilotScope(
            name="invalid",
            bakery_ids=frozenset({16}),
            category_names=frozenset({"Выпечка сытная"}),
            exclusions=(PilotExclusion(16, "Кулагина 4 Казань", "Исключена"),),
        )


@pytest.mark.parametrize("bakery_id", sorted(BASE_PILOT_10.bakery_ids))
def test_each_base_pilot_bakery_is_in_scope(bakery_id: int) -> None:
    assert is_bakery_in_scope(bakery_id)
