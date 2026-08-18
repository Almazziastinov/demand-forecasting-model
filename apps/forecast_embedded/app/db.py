from __future__ import annotations

# ruff: noqa: E501
import time
from pathlib import Path

import clickhouse_connect

from app.settings import get_settings

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENV_PATH = ROOT / ".env"


def load_env_file(path: str | Path = DEFAULT_ENV_PATH) -> dict[str, str]:
    env: dict[str, str] = {}
    file_path = Path(path)
    if not file_path.exists():
        return env

    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


import threading as _threading

_local = _threading.local()


def get_client() -> clickhouse_connect.driver.Client:
    client = getattr(_local, "client", None)
    if client is not None:
        return client
    _local.client = _create_client()
    return _local.client


def _create_client() -> clickhouse_connect.driver.Client:
    settings = get_settings()
    env = load_env_file(settings.env_file or DEFAULT_ENV_PATH)

    # Prefer runtime env vars for deploys; fall back to repository .env for local work.
    host = settings.clickhouse_host or env.get("CLICKHOUSE_HOST") or env.get("HOST")
    port_value = env.get("CLICKHOUSE_PORT") or env.get("PORT")
    port = settings.clickhouse_port or (int(port_value) if port_value else None)
    username = settings.clickhouse_user or env.get("CLICKHOUSE_USER") or env.get("USER")
    password = settings.clickhouse_password or env.get("CLICKHOUSE_PASSWORD") or env.get("PASSWORD")
    database = settings.clickhouse_database or env.get("CLICKHOUSE_DATABASE") or env.get("DATABASE")

    missing = [
        key
        for key, value in {
            "HOST": host,
            "PORT": port,
            "USER": username,
            "PASSWORD": password,
            "DATABASE": database,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing ClickHouse settings in .env: {', '.join(missing)}")

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return clickhouse_connect.get_client(
                host=host,
                port=int(port),
                username=username,
                password=password,
                database=database,
                secure=settings.clickhouse_secure,
                verify=settings.clickhouse_verify,
                connect_timeout=25,
                send_receive_timeout=300,
            )
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2)
    raise last_error
