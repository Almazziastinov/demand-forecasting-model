from __future__ import annotations

import re

from app.settings import get_settings


def table_name(base_name: str) -> str:
    suffix = get_settings().table_suffix
    if suffix and not re.fullmatch(r"_[A-Za-z0-9_]+", suffix):
        raise ValueError(
            "FORECAST_TABLE_SUFFIX must be empty or start with '_' and contain "
            "only letters, digits, and underscores"
        )
    return f"{base_name}{suffix}" if suffix else base_name
