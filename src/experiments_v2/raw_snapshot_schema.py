from __future__ import annotations

import pandas as pd


COLUMN_ALIASES: dict[str, list[str]] = {
    "check_datetime": ["check_datetime", "Дата время чека"],
    "check_date": ["check_date", "Дата продажи"],
    "cash_event_type": ["cash_event_type", "Вид события по кассе"],
    "quantity": ["quantity", "Кол-во"],
    "price": ["price", "Цена"],
    "line_amount": ["line_amount"],
    "freshness": ["freshness", "Свежесть"],
    "bakery_id": ["bakery_id"],
    "bakery_name": ["bakery_name", "Касса.Торговая точка"],
    "city": ["city"],
    "product_id": ["product_id"],
    "product_name": ["product_name", "Номенклатура"],
    "category_name": ["category_name", "Категория"],
}


def normalize_snapshot_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    work = chunk.copy()

    rename_map: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in work.columns:
            continue
        for alias in aliases:
            if alias in work.columns:
                rename_map[alias] = canonical
                break
    if rename_map:
        work = work.rename(columns=rename_map)

    if "bakery_id" not in work.columns and "bakery_name" in work.columns:
        work["bakery_id"] = work["bakery_name"].astype(str)

    if "product_id" not in work.columns and "product_name" in work.columns:
        work["product_id"] = work["product_name"].astype(str)

    if "city" not in work.columns:
        work["city"] = "unknown"

    if "line_amount" not in work.columns and {"price", "quantity"}.issubset(work.columns):
        price = pd.to_numeric(work["price"], errors="coerce")
        qty = pd.to_numeric(work["quantity"], errors="coerce")
        work["line_amount"] = price * qty

    if "check_date" not in work.columns and "check_datetime" in work.columns:
        work["check_date"] = parse_snapshot_datetime(work["check_datetime"]).dt.normalize()

    return work


def parse_snapshot_date(series: pd.Series) -> pd.Series:
    sample = series.dropna().astype(str)
    if not sample.empty and sample.iloc[0][:4].isdigit() and sample.iloc[0][4:5] == "-":
        return pd.to_datetime(series, errors="coerce")
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def parse_snapshot_datetime(series: pd.Series) -> pd.Series:
    sample = series.dropna().astype(str)
    if not sample.empty and sample.iloc[0][:4].isdigit() and sample.iloc[0][4:5] == "-":
        return pd.to_datetime(series, errors="coerce")
    return pd.to_datetime(series, errors="coerce", dayfirst=True)
