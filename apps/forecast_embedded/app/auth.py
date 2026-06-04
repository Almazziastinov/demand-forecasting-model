from __future__ import annotations

from dataclasses import dataclass

from fastapi import Header, HTTPException, Request, status

from app.settings import get_settings


ADMIN_ROLES = {"admin", "administrator", "portal_admin", "owner"}


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
