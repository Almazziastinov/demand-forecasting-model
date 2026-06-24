from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from starlette.requests import Request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "forecast_embedded"))

from app import db as embedded_db  # noqa: E402
from app.auth import AuthContext, get_auth_context  # noqa: E402
from app.services import bakery as bakery_service  # noqa: E402
from app.services import runs as runs_service  # noqa: E402
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
                "forecast_date": ["2026-06-10"],
                "forecast_final": [100.0],
            }
        )


class _FakeRunsClient:
    def __init__(self):
        self.queries: list[tuple[str, dict]] = []

    def query_df(self, query: str, parameters: dict | None = None):
        self.queries.append((query, parameters or {}))
        return pd.DataFrame(
            {
                "run_id": ["run_20260615_h14"],
                "model_version": ["bakery_day_lgbm_uplifted"],
                "profile_version": ["clickhouse_norm_uplift_sku"],
                "source_kind": ["bakery_driven"],
                "horizon_start": [pd.Timestamp("2026-06-15").date()],
                "horizon_end": [pd.Timestamp("2026-06-28").date()],
                "latest_generated_at": [pd.Timestamp("2026-06-15 12:00:00")],
                "statuses": [["active"]],
                "is_active": [1],
                "is_bias_adjusted": [True],
            }
        )


class _FakeBakeableClient:
    def __init__(self, product_ids: list[str] | None = None, error=None):
        self.product_ids = product_ids or []
        self.error = error
        self.queries: list[tuple[str, dict]] = []

    def query_df(self, query: str, parameters: dict | None = None):
        self.queries.append((query, parameters or {}))
        if self.error is not None:
            raise self.error
        return pd.DataFrame(
            {
                "product_id": self.product_ids,
                "product_name": [f"Product {value}" for value in self.product_ids],
                "category_name": ["Выпечка"] * len(self.product_ids),
            }
        )


def test_bakeable_allowlist_uses_latest_effective_snapshot(monkeypatch):
    fake = _FakeBakeableClient(["000011471", "000011472"])
    monkeypatch.setattr(bakery_service, "get_client", lambda: fake)

    products = bakery_service.get_bakeable_products("Казань", "2026-06-20")

    assert {row["product_id"] for row in products} == {"000011471", "000011472"}
    query, parameters = fake.queries[0]
    assert "from bakeable_products final" in query
    assert "select max(valid_from)" in query
    assert "where city = %(city)s" in query
    assert "valid_from <= toDate(%(effective_date)s)" in query
    assert parameters == {"city": "Казань", "effective_date": "2026-06-20"}


def test_empty_bakeable_allowlist_is_an_error(monkeypatch):
    fake = _FakeBakeableClient()
    monkeypatch.setattr(bakery_service, "get_client", lambda: fake)

    with pytest.raises(RuntimeError, match="allowlist is empty"):
        bakery_service.get_bakeable_products("Казань", "2026-06-20")


def test_bakeable_allowlist_query_error_is_not_silenced(monkeypatch):
    fake = _FakeBakeableClient(error=ConnectionError("ClickHouse unavailable"))
    monkeypatch.setattr(bakery_service, "get_client", lambda: fake)

    with pytest.raises(ConnectionError, match="ClickHouse unavailable"):
        bakery_service.get_bakeable_products("Казань", "2026-06-20")


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


def test_bakery_list_reads_lead_one_snapshots(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(bakery_service, "get_client", lambda: fake)
    auth = AuthContext(user_id="1", portal_id="portal", role="admin")

    bakery_service.get_bakery_list("active_run", "2026-06-10", auth)

    query, params = fake.queries[0]
    assert "bakery_forecast_day_embedded" in query
    assert "bakery_forecast_day_snapshots" in query
    assert "lead_days = 1" in query
    assert "argMax(forecast_final, sort_key)" in query
    assert params["run_id"] == "active_run"


def test_bakery_week_reads_actuals_from_mart_sales(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(bakery_service, "get_client", lambda: fake)
    auth = AuthContext(user_id="1", portal_id="portal", role="admin")

    bakery_service.get_bakery_week("active_run", "2026-06-11", "2026-06-17", 79, auth)

    query, params = fake.queries[0]
    assert "mart_sales_60d" in query
    assert "Svezhar.fct_check_lines" not in query
    assert "hex(fcl.cash_event_type) = %(sales_event_hex)s" in query
    assert (
        "fcl.check_date between toDate(%(start_date)s) and toDate(%(end_date)s)"
        in query
    )
    assert params["sales_event_hex"] == "D09FD180D0BED0B4D0B0D0B6D0B0"


def test_sku_hour_reads_lead_one_snapshots(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(bakery_service, "get_client", lambda: fake)
    auth = AuthContext(user_id="1", portal_id="portal", role="admin")

    bakery_service.get_sku_hour("active_run", "2026-06-10", 1, 10, auth)

    query, params = fake.queries[0]
    assert "sku_forecast_hour_embedded" in query
    assert "sku_forecast_hour_snapshots" in query
    assert "sku_forecast_day_snapshots" in query
    assert "lead_days = 1" in query
    assert params["run_id"] == "active_run"


def test_run_dates_include_lead_one_snapshots(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(runs_service, "get_client", lambda: fake)

    runs_service.get_run_dates("active_run")

    query, params = fake.queries[0]
    assert "bakery_forecast_day_embedded" in query
    assert "bakery_forecast_day_snapshots" in query
    assert "lead_days = 1" in query
    assert params["run_id"] == "active_run"


def test_list_runs_adds_forecast_display_name(monkeypatch):
    fake = _FakeRunsClient()
    monkeypatch.setattr(runs_service, "get_client", lambda: fake)

    rows = runs_service.list_runs()

    assert rows[0]["display_name"] == "Прогноз 15.06-28.06"
    assert rows[0]["is_active"] is True


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


def test_db_uses_explicit_env_file(monkeypatch):
    work_dir = Path("tests") / "_tmp_forecast_publish"
    work_dir.mkdir(parents=True, exist_ok=True)
    env_file = work_dir / ".env.embedded_dev"
    env_file.write_text(
        "\n".join(
            [
                "CLICKHOUSE_HOST=dev-host",
                "CLICKHOUSE_PORT=9440",
                "CLICKHOUSE_USER=dev-user",
                "CLICKHOUSE_PASSWORD=dev-password",
                "CLICKHOUSE_DATABASE=demand_forecast_dev",
                "CLICKHOUSE_SECURE=false",
                "CLICKHOUSE_VERIFY=false",
            ]
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_get_client(**kwargs):
        captured.update(kwargs)
        return object()

    get_settings.cache_clear()
    monkeypatch.setenv("ENV_FILE", str(env_file))
    monkeypatch.setattr(embedded_db.clickhouse_connect, "get_client", fake_get_client)

    embedded_db.get_client()

    assert captured["host"] == "dev-host"
    assert captured["port"] == 9440
    assert captured["username"] == "dev-user"
    assert captured["database"] == "demand_forecast_dev"
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
