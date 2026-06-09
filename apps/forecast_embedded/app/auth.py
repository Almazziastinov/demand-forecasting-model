from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass
from urllib.parse import unquote_plus

from fastapi import Header, HTTPException, Request, status

from app.settings import get_settings


ADMIN_ROLES = {"admin", "administrator", "portal_admin", "owner"}
BASE64_RE = re.compile(r"^[A-Za-z0-9_-]+={0,2}$")


@dataclass(frozen=True)
class AuthContext:
    user_id: str | None
    portal_id: str | None
    role: str
    email: str | None = None
    user_name: str | None = None
    user_name_encoded: str | None = None
    authorization: str | None = None
    unrestricted: bool = False

    @property
    def is_admin(self) -> bool:
        settings = get_settings()
        return (
            self.unrestricted
            or self.role.lower() in ADMIN_ROLES
            or (self.user_id is not None and self.user_id in settings.admin_user_ids)
        )

    @property
    def display_name(self) -> str | None:
        return (
            _normalize_header_text(self.user_name_encoded)
            or _normalize_header_text(self.user_name)
            or self.email
        )


def _decode_possible_base64(value: str) -> str | None:
    if len(value) < 8 or not BASE64_RE.fullmatch(value):
        return None
    padded = value + ("=" * (-len(value) % 4))
    for decoder in (base64.urlsafe_b64decode, base64.b64decode):
        try:
            decoded = decoder(padded.encode("ascii")).decode("utf-8").strip()
        except (UnicodeEncodeError, UnicodeDecodeError, ValueError, binascii.Error):
            continue
        if decoded:
            return decoded
    return None


def _repair_mojibake(value: str) -> str:
    candidates = [value]
    for source_encoding in ("latin1", "cp1252"):
        try:
            repaired = value.encode(source_encoding).decode("utf-8")
        except UnicodeError:
            continue
        candidates.append(repaired)
    return min(
        candidates,
        key=lambda item: item.count("Ð") + item.count("Ñ") + item.count("�"),
    )


def _normalize_header_text(value: str | None) -> str | None:
    if not value:
        return None
    decoded = unquote_plus(value).strip()
    if not decoded:
        return None
    base64_decoded = _decode_possible_base64(decoded)
    return _repair_mojibake(base64_decoded or decoded)


def get_auth_context(request: Request) -> AuthContext:
    settings = get_settings()
    headers = request.headers
    user_id = headers.get("x-vibe-user-id")
    portal_id = headers.get("x-vibe-portal-id")
    role = (headers.get("x-vibe-user-role") or "").strip().lower()

    if not settings.access_control_enabled:
        return AuthContext(
            user_id=user_id,
            portal_id=portal_id,
            role=role or "dev",
            email=headers.get("x-vibe-user-email"),
            user_name=headers.get("x-vibe-user-name"),
            user_name_encoded=headers.get("x-vibe-user-name-encoded"),
            authorization=headers.get("x-vibe-authorization"),
            unrestricted=True,
        )

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Vibe-User-Id",
        )
    if not portal_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Vibe-Portal-Id",
        )

    return AuthContext(
        user_id=user_id,
        portal_id=portal_id,
        role=role or "member",
        email=headers.get("x-vibe-user-email"),
        user_name=headers.get("x-vibe-user-name"),
        user_name_encoded=headers.get("x-vibe-user-name-encoded"),
        authorization=headers.get("x-vibe-authorization"),
    )


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    # Placeholder for future admin protection.
    if x_admin_token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Admin-Token",
        )
