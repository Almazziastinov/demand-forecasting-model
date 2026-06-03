from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pipelines.forecast_publish.load_forecast_run import create_client


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = ROOT / ".env"
DEFAULT_ACCESS_TABLE = "bitrix_user_bakery_access_embedded"
DEFAULT_SOURCE = "dim_management_bitrix_name_match"
DEFAULT_OVERRIDES_PATH = ROOT / "config" / "bitrix_partner_access_overrides.csv"
VIBECODE_BASE_URL = "https://vibecode.bitrix24.tech"
PARTNER_POSITION_TOKEN = "\u043f\u0430\u0440\u0442\u043d"
CLOSED_STATUS = "\u0417\u0430\u043a\u0440\u044b\u0442\u0430"


@dataclass(frozen=True)
class BitrixUser:
    user_id: str
    full_name: str
    email: str | None
    work_position: str | None
    active: bool


def _normalize_text(value: object) -> str:
    text = str(value or "").replace("ё", "е").replace("Ё", "Е").lower()
    chars = [char if char.isalnum() else " " for char in text]
    return " ".join("".join(chars).split())


def _strip_parentheses(value: str) -> str:
    return re.sub(r"\([^)]*\)", " ", value)


def build_user_name_keys(last_name: object, name: object) -> set[str]:
    last = str(last_name or "").strip()
    first = str(name or "").strip()
    candidates = {
        f"{last} {first}",
        f"{_strip_parentheses(last)} {first}",
    }
    for parenthetical in re.findall(r"\(([^)]*)\)", last):
        candidates.add(f"{parenthetical} {first}")
    if last:
        candidates.add(f"{last.split()[0]} {first}")
    return {
        key
        for key in (_normalize_text(candidate) for candidate in candidates)
        if key
    }


def is_partner_position(value: object) -> bool:
    return PARTNER_POSITION_TOKEN in _normalize_text(value)


def _request_json(url: str, api_key: str, retries: int = 3) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"X-Api-Key": api_key})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except OSError as exc:
            last_error = exc
            if attempt + 1 == retries:
                break
            time.sleep(2**attempt)
    message = f"VibeCode request failed after {retries} attempts: {url}"
    raise RuntimeError(message) from last_error


def fetch_portal_id(api_key: str, base_url: str = VIBECODE_BASE_URL) -> str:
    response = _request_json(f"{base_url}/v1/me", api_key)
    portal = (response.get("data") or {}).get("portal")
    if not portal:
        raise RuntimeError("VibeCode /v1/me did not return data.portal")
    return str(portal)


def fetch_bitrix_users(
    api_key: str,
    base_url: str = VIBECODE_BASE_URL,
    page_size: int = 50,
) -> list[dict[str, Any]]:
    users: list[dict[str, Any]] = []
    select = "id,name,lastName,email,workPosition,departmentId,active"
    for offset in range(0, 10000, page_size):
        url = f"{base_url}/v1/users?limit={page_size}&offset={offset}&select={select}"
        response = _request_json(url, api_key)
        page = response.get("data") or []
        if not page:
            break
        users.extend(page)
        if len(page) < page_size:
            break
    return users


def build_user_index(
    users: Iterable[dict[str, Any]],
    include_inactive: bool = False,
) -> dict[str, list[BitrixUser]]:
    index: dict[str, list[BitrixUser]] = {}
    for user in users:
        if not include_inactive and not user.get("active"):
            continue
        full_name = f"{user.get('lastName') or ''} {user.get('name') or ''}".strip()
        bitrix_user = BitrixUser(
            user_id=str(user.get("id")),
            full_name=full_name,
            email=user.get("email"),
            work_position=user.get("workPosition"),
            active=bool(user.get("active")),
        )
        for key in build_user_name_keys(user.get("lastName"), user.get("name")):
            index.setdefault(key, []).append(bitrix_user)
    return index


def load_overrides(path: str | Path | None) -> dict[str, str]:
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        return {}
    frame = pd.read_csv(file_path)
    required = {"partner_name", "bitrix_user_name"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing override columns: {', '.join(sorted(missing))}")
    overrides: dict[str, str] = {}
    for record in frame.to_dict("records"):
        partner_key = _normalize_text(record.get("partner_name"))
        user_key = _normalize_text(record.get("bitrix_user_name"))
        if partner_key and user_key:
            overrides[partner_key] = user_key
    return overrides


def load_management(client) -> pd.DataFrame:
    return client.query_df(
        """
        select
            toInt64OrNull(toString(m.bakery_id)) as bakery_id,
            coalesce(b.bakery_name, toString(m.bakery_id)) as bakery_name,
            m.partner as partner_name,
            m.status,
            m.format,
            coalesce(m.city, b.city) as city,
            b.price_region
        from dim_management m
        left join dim_bakeries b
          on toString(b.bakery_id) = toString(m.bakery_id)
        where m.partner is not null
          and m.partner != ''
          and coalesce(m.status, '') != {closed_status:String}
          and toInt64OrNull(toString(m.bakery_id)) is not null
        """,
        parameters={"closed_status": CLOSED_STATUS},
    )


def build_access_rows(
    management: pd.DataFrame,
    users_by_key: dict[str, list[BitrixUser]],
    portal_id: str,
    source: str = DEFAULT_SOURCE,
    require_partner_position: bool = False,
    partner_user_overrides: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
    rows: list[dict[str, Any]] = []
    unmatched_partners: set[str] = set()
    non_partner_position_matches: set[str] = set()
    overrides = partner_user_overrides or {}

    for record in management.to_dict("records"):
        partner_name = str(record["partner_name"])
        partner_key = _normalize_text(partner_name)
        user_key = overrides.get(partner_key, partner_key)
        matched_users = users_by_key.get(user_key, [])
        if not matched_users:
            unmatched_partners.add(partner_name)
            continue

        accepted = False
        for user in matched_users:
            if require_partner_position and not is_partner_position(user.work_position):
                non_partner_position_matches.add(partner_name)
                continue
            accepted = True
            rows.append(
                {
                    "bitrix_portal_id": portal_id,
                    "bitrix_user_id": user.user_id,
                    "bitrix_email": user.email,
                    "bitrix_user_name": user.full_name,
                    "bitrix_work_position": user.work_position,
                    "partner_name": partner_name,
                    "bakery_id": int(record["bakery_id"]),
                    "bakery_name": record.get("bakery_name"),
                    "access_role": "partner",
                    "match_method": (
                        "partner_name_override"
                        if partner_key in overrides
                        else "partner_name_exact"
                    ),
                    "source": source,
                    "updated_at": updated_at,
                }
            )
        if not accepted:
            unmatched_partners.add(partner_name)

    access = pd.DataFrame(rows).drop_duplicates(
        ["bitrix_portal_id", "bitrix_user_id", "bakery_id", "source"]
    )
    summary = {
        "management_rows": int(len(management)),
        "management_partners": int(management["partner_name"].nunique()),
        "access_rows": int(len(access)),
        "matched_partners": (
            int(access["partner_name"].nunique()) if not access.empty else 0
        ),
        "matched_users": (
            int(access["bitrix_user_id"].nunique()) if not access.empty else 0
        ),
        "unmatched_partners": sorted(unmatched_partners),
        "non_partner_position_matches": sorted(non_partner_position_matches),
    }
    return access, summary


def replace_access_rows(
    client,
    table: str,
    access: pd.DataFrame,
    portal_id: str,
    source: str,
) -> None:
    safe_portal = portal_id.replace("\\", "\\\\").replace("'", "\\'")
    safe_source = source.replace("\\", "\\\\").replace("'", "\\'")
    client.command(
        f"""
        alter table {table}
        delete where bitrix_portal_id = '{safe_portal}'
          and source = '{safe_source}'
        """
    )
    if not access.empty:
        client.insert_df(table, access)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync Bitrix partner users to bakery access table."
    )
    parser.add_argument("--env-path", default=str(DEFAULT_ENV_PATH))
    parser.add_argument("--api-key", default=os.getenv("VIBECODE_API_KEY"))
    parser.add_argument("--base-url", default=VIBECODE_BASE_URL)
    parser.add_argument("--access-table", default=DEFAULT_ACCESS_TABLE)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--overrides-path", default=str(DEFAULT_OVERRIDES_PATH))
    parser.add_argument("--portal-id", default=None)
    parser.add_argument("--include-inactive", action="store_true")
    parser.add_argument("--require-partner-position", action="store_true")
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise RuntimeError("Set VIBECODE_API_KEY or pass --api-key")

    client = create_client(Path(args.env_path))
    portal_id = args.portal_id or fetch_portal_id(args.api_key, args.base_url)
    users = fetch_bitrix_users(args.api_key, args.base_url)
    users_by_key = build_user_index(users, include_inactive=args.include_inactive)
    management = load_management(client)
    overrides = load_overrides(args.overrides_path)
    access, summary = build_access_rows(
        management=management,
        users_by_key=users_by_key,
        portal_id=portal_id,
        source=args.source,
        require_partner_position=args.require_partner_position,
        partner_user_overrides=overrides,
    )
    summary["portal_id"] = portal_id
    summary["bitrix_users_loaded"] = len(users)
    summary["apply"] = bool(args.apply)

    if args.apply:
        replace_access_rows(client, args.access_table, access, portal_id, args.source)

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
