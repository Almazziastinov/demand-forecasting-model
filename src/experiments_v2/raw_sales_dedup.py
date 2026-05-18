from __future__ import annotations

import pandas as pd

from src.experiments_v2.raw_snapshot_schema import normalize_snapshot_chunk
from src.experiments_v2.raw_snapshot_schema import parse_snapshot_date
from src.experiments_v2.raw_snapshot_schema import parse_snapshot_datetime


RAW_DATE_COL = "check_date"
RAW_DATETIME_COL = "check_datetime"
RAW_EVENT_COL = "cash_event_type"
RAW_QTY_COL = "quantity"
RAW_PRICE_COL = "price"
RAW_AMOUNT_COL = "line_amount"
RAW_BAKERY_ID_COL = "bakery_id"
RAW_BAKERY_NAME_COL = "bakery_name"
RAW_CITY_COL = "city"
RAW_PRODUCT_ID_COL = "product_id"

REQUIRED_COLS = [
    RAW_DATE_COL,
    RAW_DATETIME_COL,
    RAW_EVENT_COL,
    RAW_QTY_COL,
    RAW_PRICE_COL,
    RAW_AMOUNT_COL,
    RAW_BAKERY_ID_COL,
    RAW_BAKERY_NAME_COL,
    RAW_CITY_COL,
    RAW_PRODUCT_ID_COL,
]

STRICT_DUP_KEYS = [
    RAW_DATETIME_COL,
    RAW_BAKERY_ID_COL,
    RAW_PRODUCT_ID_COL,
    RAW_QTY_COL,
    RAW_PRICE_COL,
    RAW_AMOUNT_COL,
    RAW_EVENT_COL,
]


def prepare_sales_chunk(chunk: pd.DataFrame, *, sales_events: set[str]) -> pd.DataFrame:
    work = normalize_snapshot_chunk(chunk)
    for col in REQUIRED_COLS:
        if col not in work.columns:
            work[col] = pd.NA

    work = work[work[RAW_EVENT_COL].isin(sales_events)].copy()
    if work.empty:
        return work[REQUIRED_COLS].copy()

    work[RAW_DATE_COL] = parse_snapshot_date(work[RAW_DATE_COL]).dt.normalize()
    work[RAW_DATETIME_COL] = parse_snapshot_datetime(work[RAW_DATETIME_COL])
    work[RAW_QTY_COL] = (
        pd.to_numeric(work[RAW_QTY_COL], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    work[RAW_PRICE_COL] = pd.to_numeric(work[RAW_PRICE_COL], errors="coerce")
    work[RAW_AMOUNT_COL] = pd.to_numeric(work[RAW_AMOUNT_COL], errors="coerce")

    if RAW_CITY_COL in work.columns:
        work[RAW_CITY_COL] = work[RAW_CITY_COL].fillna("unknown")

    work = work.dropna(
        subset=[
            RAW_DATE_COL,
            RAW_BAKERY_ID_COL,
            RAW_BAKERY_NAME_COL,
            RAW_PRODUCT_ID_COL,
        ]
    ).copy()
    return work[REQUIRED_COLS].copy()


def deduplicate_sales_chunk(
    sales: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    if sales.empty:
        return sales.copy(), _build_stats(sales, sales, duplicate_groups=0)

    duplicate_mask = sales.duplicated(subset=STRICT_DUP_KEYS, keep="first")
    duplicate_groups = int(
        sales.duplicated(subset=STRICT_DUP_KEYS, keep=False).sum()
        - duplicate_mask.sum()
    )
    deduped = sales.loc[~duplicate_mask].reset_index(drop=True)
    return deduped, _build_stats(sales, deduped, duplicate_groups=duplicate_groups)


def _build_stats(
    raw_sales: pd.DataFrame,
    deduped_sales: pd.DataFrame,
    *,
    duplicate_groups: int,
) -> dict[str, float | int]:
    raw_qty = (
        float(raw_sales[RAW_QTY_COL].sum()) if RAW_QTY_COL in raw_sales.columns else 0.0
    )
    deduped_qty = (
        float(deduped_sales[RAW_QTY_COL].sum())
        if RAW_QTY_COL in deduped_sales.columns
        else 0.0
    )
    raw_amount = (
        float(raw_sales[RAW_AMOUNT_COL].fillna(0.0).sum())
        if RAW_AMOUNT_COL in raw_sales.columns
        else 0.0
    )
    deduped_amount = (
        float(deduped_sales[RAW_AMOUNT_COL].fillna(0.0).sum())
        if RAW_AMOUNT_COL in deduped_sales.columns
        else 0.0
    )

    return {
        "raw_rows": int(len(raw_sales)),
        "deduped_rows": int(len(deduped_sales)),
        "removed_rows": int(len(raw_sales) - len(deduped_sales)),
        "duplicate_groups": int(duplicate_groups),
        "raw_quantity_sum": raw_qty,
        "deduped_quantity_sum": deduped_qty,
        "removed_quantity_sum": raw_qty - deduped_qty,
        "raw_line_amount_sum": raw_amount,
        "deduped_line_amount_sum": deduped_amount,
        "removed_line_amount_sum": raw_amount - deduped_amount,
    }
