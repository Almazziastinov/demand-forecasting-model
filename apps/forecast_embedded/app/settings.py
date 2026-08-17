from __future__ import annotations

# ruff: noqa: E501
import os
from dataclasses import dataclass
from functools import lru_cache


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_env_file(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    env: dict[str, str] = {}
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except FileNotFoundError:
        return env
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def _setting(env_values: dict[str, str], key: str, default: str | None = None) -> str | None:
    return os.getenv(key) or env_values.get(key) or default


@dataclass(frozen=True)
class Settings:
    app_title: str
    app_env: str
    port: int
    env_file: str | None
    table_suffix: str
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
    pilot_user_ids: frozenset[str]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    env_file = os.getenv("ENV_FILE") or None
    env_values = _load_env_file(env_file)
    return Settings(
        app_title=_setting(env_values, "APP_TITLE", "Bakery Forecast Embedded") or "",
        app_env=_setting(env_values, "APP_ENV", "dev") or "dev",
        port=int(_setting(env_values, "PORT", "3000") or "3000"),
        env_file=env_file,
        table_suffix=_setting(env_values, "FORECAST_TABLE_SUFFIX", "") or "",
        active_run_id=_setting(env_values, "ACTIVE_RUN_ID") or None,
        bitrix_embed_mode=_as_bool(_setting(env_values, "BITRIX_EMBED_MODE"), default=False),
        clickhouse_host=_setting(env_values, "CLICKHOUSE_HOST") or None,
        clickhouse_port=(
            int(_setting(env_values, "CLICKHOUSE_PORT"))
            if _setting(env_values, "CLICKHOUSE_PORT")
            else None
        ),
        clickhouse_user=_setting(env_values, "CLICKHOUSE_USER") or None,
        clickhouse_password=_setting(env_values, "CLICKHOUSE_PASSWORD") or None,
        clickhouse_database=_setting(env_values, "CLICKHOUSE_DATABASE") or None,
        clickhouse_secure=_as_bool(_setting(env_values, "CLICKHOUSE_SECURE"), default=True),
        clickhouse_verify=_as_bool(_setting(env_values, "CLICKHOUSE_VERIFY"), default=False),
        access_control_enabled=_as_bool(
            _setting(env_values, "ACCESS_CONTROL_ENABLED"),
            default=_as_bool(_setting(env_values, "BITRIX_EMBED_MODE"), default=False),
        ),
        admin_user_ids=frozenset(
            user_id.strip()
            for user_id in (_setting(env_values, "ADMIN_USER_IDS", "27979") or "").split(",")
            if user_id.strip()
        ),
        pilot_user_ids=frozenset(
            user_id.strip()
            for user_id in (_setting(env_values, "PILOT_USER_IDS", "") or "").split(",")
            if user_id.strip()
        ),
    )
