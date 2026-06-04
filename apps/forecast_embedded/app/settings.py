from __future__ import annotations

# ruff: noqa: E501

import os
from dataclasses import dataclass
from functools import lru_cache


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_title: str
    app_env: str
    port: int
    active_run_id: str | None
    bitrix_embed_mode: bool
    clickhouse_host: str | None
    clickhouse_port: int | None
    clickhouse_user: str | None
    clickhouse_password: str | None
    clickhouse_database: str | None
    clickhouse_secure: bool
    clickhouse_verify: bool
    access_control_enabled: bool
    admin_user_ids: frozenset[str]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        app_title=os.getenv("APP_TITLE", "Bakery Forecast Embedded"),
        app_env=os.getenv("APP_ENV", "dev"),
        port=int(os.getenv("PORT", "3000")),
        active_run_id=os.getenv("ACTIVE_RUN_ID") or None,
        bitrix_embed_mode=_as_bool(os.getenv("BITRIX_EMBED_MODE"), default=False),
        clickhouse_host=os.getenv("CLICKHOUSE_HOST") or None,
        clickhouse_port=int(os.getenv("CLICKHOUSE_PORT")) if os.getenv("CLICKHOUSE_PORT") else None,
        clickhouse_user=os.getenv("CLICKHOUSE_USER") or None,
        clickhouse_password=os.getenv("CLICKHOUSE_PASSWORD") or None,
        clickhouse_database=os.getenv("CLICKHOUSE_DATABASE") or None,
        clickhouse_secure=_as_bool(os.getenv("CLICKHOUSE_SECURE"), default=True),
        clickhouse_verify=_as_bool(os.getenv("CLICKHOUSE_VERIFY"), default=False),
        access_control_enabled=_as_bool(
            os.getenv("ACCESS_CONTROL_ENABLED"),
            default=_as_bool(os.getenv("BITRIX_EMBED_MODE"), default=False),
        ),
        admin_user_ids=frozenset(
            user_id.strip()
            for user_id in os.getenv("ADMIN_USER_IDS", "27979").split(",")
            if user_id.strip()
        ),
    )
