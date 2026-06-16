from __future__ import annotations

import re
from pathlib import Path

EMBEDDED_TABLES = [
    "forecast_runs_embedded",
    "bakery_forecast_day_embedded",
    "forecast_day_context_embedded",
    "sku_forecast_day_embedded",
    "sku_forecast_hour_embedded",
    "bakery_forecast_day_snapshots",
    "sku_forecast_day_snapshots",
    "sku_forecast_hour_snapshots",
    "sku_hour_share_profile_smoothed_embedded",
    "sku_hour_uplift_multiplier_embedded",
    "bitrix_user_bakery_access_embedded",
    "bakery_month_revenue_embedded",
]


def validate_table_suffix(suffix: str) -> str:
    if not suffix:
        return ""
    if not re.fullmatch(r"_[A-Za-z0-9_]+", suffix):
        raise ValueError(
            "FORECAST_TABLE_SUFFIX must be empty or start with '_' and contain "
            "only letters, digits, and underscores"
        )
    return suffix


def table_name(base_name: str, suffix: str = "") -> str:
    suffix = validate_table_suffix(suffix)
    return f"{base_name}{suffix}" if suffix else base_name


def apply_table_suffix_to_sql(sql_text: str, suffix: str = "") -> str:
    suffix = validate_table_suffix(suffix)
    if not suffix:
        return sql_text
    result = sql_text
    for base_name in sorted(EMBEDDED_TABLES, key=len, reverse=True):
        result = re.sub(rf"\b{re.escape(base_name)}\b", f"{base_name}{suffix}", result)
    return result


def get_table_suffix_from_env_file(env_file: str | Path) -> str:
    from pipelines.forecast_publish.load_forecast_run import load_env_file

    env = load_env_file(env_file)
    return validate_table_suffix(env.get("FORECAST_TABLE_SUFFIX", "").strip())
