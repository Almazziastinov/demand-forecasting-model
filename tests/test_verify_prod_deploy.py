from scripts.verify_prod_deploy import is_source_run_of_active


def test_allows_unactivated_source_run_referenced_by_weighted_active_run() -> None:
    notes = (
        "Weighted weekday calculated demand; scope from "
        "prod_base_bakery_norm_recent_20261120_h14"
    )

    assert is_source_run_of_active(notes, "prod_base_bakery_norm_recent_20261120_h14")


def test_does_not_allow_unrelated_unactivated_run() -> None:
    notes = (
        "Weighted weekday calculated demand; scope from "
        "prod_base_bakery_norm_recent_20261120_h14"
    )

    assert not is_source_run_of_active(
        notes,
        "prod_direct_alpha_025_20261120_h14",
    )
