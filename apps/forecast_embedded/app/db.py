from __future__ import annotations

# ruff: noqa: E501

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


def get_client():
    settings = get_settings()
    env = load_env_file()

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

    return clickhouse_connect.get_client(
        host=host,
        port=int(port),
        username=username,
        password=password,
        database=database,
        secure=settings.clickhouse_secure,
        verify=settings.clickhouse_verify,
    )
