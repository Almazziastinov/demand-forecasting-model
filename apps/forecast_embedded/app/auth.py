from __future__ import annotations

from fastapi import Header, HTTPException, status


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    # Placeholder for future admin protection.
    if x_admin_token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Admin-Token",
        )
