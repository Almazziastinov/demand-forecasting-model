from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from starlette.requests import Request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "forecast_embedded"))

from app.auth import AuthContext, get_auth_context  # noqa: E402
from app.services import bakery as bakery_service  # noqa: E402
from app.settings import get_settings  # noqa: E402


class _FakeClient:
    def __init__(self):
        self.queries: list[tuple[str, dict]] = []

    def query_df(self, query: str, parameters: dict | None = None):
        self.queries.append((query, parameters or {}))
        return pd.DataFrame(
            {
                "bakery_id": [1],
                "bakery_name": ["Bakery"],
                "city": ["Kazan"],
                "forecast_final": [100.0],
            }
        )


def test_partner_bakery_list_is_filtered_by_access_table(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(bakery_service, "get_client", lambda: fake)
    auth = AuthContext(
        user_id="799",
        portal_id="portal",
        role="member",
        email="partner@example.com",
    )

    rows = bakery_service.get_bakery_list("run", "2026-06-01", auth)

    query, params = fake.queries[0]
    assert rows[0]["bakery_id"] == 1
    assert "bitrix_user_bakery_access_embedded" in query
    assert "dim_management" in query
    assert "coalesce(status, '') !=" in query
    assert "b.bakery_id in" in query
    assert "bitrix_portal_id" in query
    assert params["closed_bakery_status"] == "Закрыта"
    assert params["bitrix_portal_id"] == "portal"
    assert params["bitrix_user_id"] == "799"
    assert params["bitrix_email"] == "partner@example.com"


def test_admin_bakery_list_is_not_filtered_by_access_table(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(bakery_service, "get_client", lambda: fake)
    auth = AuthContext(user_id="1", portal_id="portal", role="admin")

    bakery_service.get_bakery_list("run", "2026-06-01", auth)

    query, params = fake.queries[0]
    assert "bitrix_user_bakery_access_embedded" not in query
    assert "bitrix_user_id" not in params
    assert "dim_management" in query
    assert params["closed_bakery_status"] == "Закрыта"


def test_access_control_requires_portal_id(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("ACCESS_CONTROL_ENABLED", "1")
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(b"x-vibe-user-id", b"799")],
    }

    with pytest.raises(Exception) as exc_info:
        get_auth_context(Request(scope))

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Missing X-Vibe-Portal-Id"
    get_settings.cache_clear()


def test_auth_display_name_repairs_utf8_mojibake():
    auth = AuthContext(
        user_id="27979",
        portal_id="portal",
        role="admin",
        user_name="Ð\x90Ð»Ð¼Ð°Ð· Ð\x91Ð¸Ð°Ñ\x81Ñ\x82Ð¸Ð½Ð¾Ð²",
        email="almaz@example.com",
    )

    assert auth.display_name == "Алмаз Биастинов"


def test_auth_display_name_decodes_encoded_header():
    auth = AuthContext(
        user_id="27979",
        portal_id="portal",
        role="admin",
        user_name_encoded="0JDQu9C80LDQtyDQkdC40LDRgdGC0LjQvdC-0LI=",
        email="almaz@example.com",
    )

    assert auth.display_name == "Алмаз Биастинов"
