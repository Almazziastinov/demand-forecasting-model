from __future__ import annotations

# ruff: noqa: E501

from pathlib import Path

import pandas as pd

from pipelines.forecast_publish.sync_bitrix_partner_access import (
    build_access_rows,
    build_user_index,
    is_partner_position,
    load_overrides,
)


def test_build_user_index_matches_parenthetical_last_name():
    users = [
        {
            "id": 161,
            "lastName": "\u0412\u0430\u043d\u0438\u0434\u043e\u0432\u0441\u043a\u0430\u044f (\u0410\u0445\u043c\u0435\u0442\u043e\u0432\u0430)",
            "name": "\u0410\u043d\u043d\u0430",
            "email": "partner@example.com",
            "workPosition": "\u0424\u0440\u0430\u043d\u0448\u0438\u0437\u043d\u044b\u0439 \u043f\u0430\u0440\u0442\u043d\u0435\u0440",
            "active": True,
        }
    ]

    index = build_user_index(users)

    assert "\u0432\u0430\u043d\u0438\u0434\u043e\u0432\u0441\u043a\u0430\u044f \u0430\u043d\u043d\u0430" in index
    assert "\u0430\u0445\u043c\u0435\u0442\u043e\u0432\u0430 \u0430\u043d\u043d\u0430" in index


def test_is_partner_position_accepts_franchise_partner():
    assert is_partner_position(
        "\u0424\u0440\u0430\u043d\u0448\u0438\u0437\u043d\u044b\u0439 \u043f\u0430\u0440\u0442\u043d\u0435\u0440"
    )
    assert is_partner_position(
        "\u0444\u0440\u0430\u043d\u0448\u0438\u0437\u043d\u044b\u0439 \u043f\u0430\u0440\u0442\u043d\u0435\u0440 "
    )
    assert not is_partner_position(
        "\u0423\u043f\u0440\u0430\u0432\u043b\u044f\u044e\u0449\u0438\u0439"
    )


def test_build_access_rows_allows_non_partner_position_by_default():
    management = pd.DataFrame(
        [
            {
                "bakery_id": 1,
                "bakery_name": "Bakery 1",
                "partner_name": "\u0418\u0432\u0430\u043d\u043e\u0432\u0430 \u0418\u0440\u0438\u043d\u0430",
            }
        ]
    )
    users = [
        {
            "id": 20,
            "lastName": "\u0418\u0432\u0430\u043d\u043e\u0432\u0430",
            "name": "\u0418\u0440\u0438\u043d\u0430",
            "email": "ivanova@example.com",
            "workPosition": "\u0423\u043f\u0440\u0430\u0432\u043b\u044f\u044e\u0449\u0438\u0439",
            "active": True,
        },
    ]

    access, summary = build_access_rows(
        management=management,
        users_by_key=build_user_index(users),
        portal_id="portal",
    )

    assert access["bitrix_user_id"].tolist() == ["20"]
    assert access["bakery_id"].tolist() == [1]
    assert summary["matched_partners"] == 1
    assert summary["unmatched_partners"] == []


def test_build_access_rows_can_require_partner_position():
    management = pd.DataFrame(
        [
            {
                "bakery_id": 1,
                "bakery_name": "Bakery 1",
                "partner_name": "\u0418\u0432\u0430\u043d\u043e\u0432\u0430 \u0418\u0440\u0438\u043d\u0430",
            }
        ]
    )
    users = [
        {
            "id": 20,
            "lastName": "\u0418\u0432\u0430\u043d\u043e\u0432\u0430",
            "name": "\u0418\u0440\u0438\u043d\u0430",
            "email": "ivanova@example.com",
            "workPosition": "\u0423\u043f\u0440\u0430\u0432\u043b\u044f\u044e\u0449\u0438\u0439",
            "active": True,
        },
    ]

    access, summary = build_access_rows(
        management=management,
        users_by_key=build_user_index(users),
        portal_id="portal",
        require_partner_position=True,
    )

    assert access.empty
    assert summary["unmatched_partners"] == [
        "\u0418\u0432\u0430\u043d\u043e\u0432\u0430 \u0418\u0440\u0438\u043d\u0430"
    ]
    assert summary["non_partner_position_matches"] == [
        "\u0418\u0432\u0430\u043d\u043e\u0432\u0430 \u0418\u0440\u0438\u043d\u0430"
    ]


def test_build_access_rows_uses_manual_override():
    management = pd.DataFrame(
        [
            {
                "bakery_id": 1,
                "bakery_name": "Bakery 1",
                "partner_name": "\u0413\u0430\u0440\u0438\u043f\u043e\u0432\u0430 \u041d\u0430\u0438\u043b\u044f",
            }
        ]
    )
    users = [
        {
            "id": 185,
            "lastName": "\u0413\u0430\u0440\u0438\u043f\u043e\u0432\u0430",
            "name": "\u041d\u0435\u043b\u044f",
            "email": "ngaripova@example.com",
            "workPosition": "\u0424\u0440\u0430\u043d\u0448\u0438\u0437\u043d\u044b\u0439 \u043f\u0430\u0440\u0442\u043d\u0435\u0440",
            "active": True,
        },
    ]
    overrides = {
        "\u0433\u0430\u0440\u0438\u043f\u043e\u0432\u0430 \u043d\u0430\u0438\u043b\u044f": "\u0433\u0430\u0440\u0438\u043f\u043e\u0432\u0430 \u043d\u0435\u043b\u044f"
    }

    access, _summary = build_access_rows(
        management=management,
        users_by_key=build_user_index(users),
        portal_id="portal",
        partner_user_overrides=overrides,
    )

    assert access["bitrix_user_id"].tolist() == ["185"]
    assert access["match_method"].tolist() == ["partner_name_override"]


def test_load_overrides_reads_partner_user_mapping():
    path = Path("config/bitrix_partner_access_overrides.csv")

    overrides = load_overrides(path)

    assert overrides[
        "\u0438\u043c\u0430\u0433\u0438\u043b\u043e\u0432 \u0441\u0430\u043b\u0430\u0432\u0430\u0442"
    ] == "\u0438\u0441\u043c\u0430\u0433\u0438\u043b\u043e\u0432 \u0441\u0430\u043b\u0430\u0432\u0430\u0442"
