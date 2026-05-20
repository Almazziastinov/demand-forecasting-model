from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments_v2.raw_sales_dedup import RAW_AMOUNT_COL
from src.experiments_v2.raw_sales_dedup import RAW_BAKERY_ID_COL
from src.experiments_v2.raw_sales_dedup import RAW_BAKERY_NAME_COL
from src.experiments_v2.raw_sales_dedup import RAW_CITY_COL
from src.experiments_v2.raw_sales_dedup import RAW_DATE_COL
from src.experiments_v2.raw_sales_dedup import RAW_DATETIME_COL
from src.experiments_v2.raw_sales_dedup import RAW_PRICE_COL
from src.experiments_v2.raw_sales_dedup import RAW_PRODUCT_ID_COL
from src.experiments_v2.raw_sales_dedup import RAW_QTY_COL
from src.experiments_v2.raw_snapshot_schema import normalize_snapshot_chunk
from src.experiments_v2.raw_snapshot_schema import parse_snapshot_date
from src.experiments_v2.raw_snapshot_schema import parse_snapshot_datetime
from src.experiments_v2.raw_sales_dedup import deduplicate_sales_chunk


ROOT = Path(__file__).resolve().parents[2]
SALES_EVENT = "Продажа"
CHUNK_SIZE = 1_000_000

DATE_COL = "date"
BAKERY_ID_COL = "bakery_id"
BAKERY_NAME_COL = "bakery_name"
CITY_COL = "city"
PRODUCT_ID_COL = "product_id"
PRODUCT_NAME_COL = "product_name"
CATEGORY_COL = "category_name"

OUTPUT_NAME = "sku_daily_research_base.csv"
PANEL_OUTPUT_NAME = "sku_daily_research_panel.csv"
SUMMARY_OUTPUT_NAME = "sku_daily_research_base_summary.json"
AUDIT_DIR_NAME = "sku_daily_research_base_audit"


def _empty_sales_stats() -> dict[str, float | int]:
    return {
        "raw_rows": 0,
        "deduped_rows": 0,
        "removed_rows": 0,
        "duplicate_groups": 0,
        "raw_quantity_sum": 0.0,
        "deduped_quantity_sum": 0.0,
        "removed_quantity_sum": 0.0,
        "raw_line_amount_sum": 0.0,
        "deduped_line_amount_sum": 0.0,
        "removed_line_amount_sum": 0.0,
    }


def _sum_stats(total: dict[str, float | int], part: dict[str, float | int]) -> None:
    for key, value in part.items():
        total[key] += value


def _to_bool_flag(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool).astype(int)


def _mode_or_nan(series: pd.Series) -> str | float:
    mode = series.dropna().astype(str).mode()
    if mode.empty:
        return np.nan
    return mode.iloc[0]


def _ensure_flag_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        df[col] = 0
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def _coerce_datetime(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce")


def _read_csv(path: str | Path, usecols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", usecols=usecols)


def aggregate_sales_to_sku_day(source_path: str | Path, *, chunk_size: int = CHUNK_SIZE) -> tuple[pd.DataFrame, dict[str, float | int]]:
    usecols = [
        RAW_DATE_COL,
        RAW_DATETIME_COL,
        "cash_event_type",
        RAW_QTY_COL,
        RAW_PRICE_COL,
        RAW_AMOUNT_COL,
        RAW_BAKERY_ID_COL,
        RAW_BAKERY_NAME_COL,
        RAW_CITY_COL,
        RAW_PRODUCT_ID_COL,
        PRODUCT_NAME_COL,
        CATEGORY_COL,
        "check_date",
        "check_datetime",
        "product_name",
        "category_name",
        "bakery_id",
        "bakery_name",
        "city",
        "product_id",
        "quantity",
        "price",
        "line_amount",
        "Дата продажи",
        "Дата время чека",
        "Вид события по кассе",
        "Кол-во",
        "Цена",
        "Касса.Торговая точка",
        "Номенклатура",
        "Категория",
    ]
    parts: list[pd.DataFrame] = []
    stats = _empty_sales_stats()

    reader = pd.read_csv(
        source_path,
        encoding="utf-8-sig",
        usecols=lambda c: c in usecols,
        chunksize=chunk_size,
    )
    for chunk in reader:
        prepared_sales = normalize_snapshot_chunk(chunk)
        for col in [RAW_DATE_COL, RAW_DATETIME_COL, "cash_event_type", RAW_QTY_COL, RAW_PRICE_COL, RAW_AMOUNT_COL, RAW_BAKERY_ID_COL, RAW_BAKERY_NAME_COL, RAW_CITY_COL, RAW_PRODUCT_ID_COL]:
            if col not in prepared_sales.columns:
                prepared_sales[col] = pd.NA
        if PRODUCT_NAME_COL not in prepared_sales.columns:
            prepared_sales[PRODUCT_NAME_COL] = np.nan
        if CATEGORY_COL not in prepared_sales.columns:
            prepared_sales[CATEGORY_COL] = np.nan

        prepared_sales = prepared_sales[prepared_sales["cash_event_type"].isin({SALES_EVENT})].copy()
        if prepared_sales.empty:
            continue
        prepared_sales[RAW_DATE_COL] = parse_snapshot_date(prepared_sales[RAW_DATE_COL]).dt.normalize()
        prepared_sales[RAW_DATETIME_COL] = parse_snapshot_datetime(prepared_sales[RAW_DATETIME_COL])
        prepared_sales[RAW_QTY_COL] = pd.to_numeric(prepared_sales[RAW_QTY_COL], errors="coerce").fillna(0.0).clip(lower=0.0)
        prepared_sales[RAW_PRICE_COL] = pd.to_numeric(prepared_sales[RAW_PRICE_COL], errors="coerce")
        prepared_sales[RAW_AMOUNT_COL] = pd.to_numeric(prepared_sales[RAW_AMOUNT_COL], errors="coerce")
        prepared_sales[RAW_CITY_COL] = prepared_sales[RAW_CITY_COL].fillna("unknown")
        prepared_sales = prepared_sales.dropna(subset=[RAW_DATE_COL, RAW_BAKERY_ID_COL, RAW_BAKERY_NAME_COL, RAW_PRODUCT_ID_COL]).copy()
        if prepared_sales.empty:
            continue
        prepared_sales, part_stats = deduplicate_sales_chunk(prepared_sales)
        _sum_stats(stats, part_stats)

        prepared_sales[DATE_COL] = prepared_sales[RAW_DATE_COL]
        prepared_sales["hour"] = prepared_sales[RAW_DATETIME_COL].dt.hour
        prepared_sales["price_x_qty"] = prepared_sales[RAW_PRICE_COL].fillna(0.0) * prepared_sales[RAW_QTY_COL]
        prepared_sales["priced_qty"] = np.where(prepared_sales[RAW_PRICE_COL].notna(), prepared_sales[RAW_QTY_COL], 0.0)

        grouped = (
            prepared_sales.groupby(
                [DATE_COL, RAW_BAKERY_ID_COL, RAW_BAKERY_NAME_COL, RAW_CITY_COL, RAW_PRODUCT_ID_COL, PRODUCT_NAME_COL, CATEGORY_COL],
                as_index=False,
            )
            .agg(
                observed_sales_qty=(RAW_QTY_COL, "sum"),
                observed_sales_amount=(RAW_AMOUNT_COL, "sum"),
                sales_rows_count=(RAW_QTY_COL, "size"),
                sales_hours_count=("hour", "nunique"),
                first_sale_hour=("hour", "min"),
                last_sale_hour=("hour", "max"),
                price_x_qty=("price_x_qty", "sum"),
                priced_qty=("priced_qty", "sum"),
            )
            .rename(
                columns={
                    RAW_BAKERY_ID_COL: BAKERY_ID_COL,
                    RAW_BAKERY_NAME_COL: BAKERY_NAME_COL,
                    RAW_CITY_COL: CITY_COL,
                    RAW_PRODUCT_ID_COL: PRODUCT_ID_COL,
                }
            )
        )
        parts.append(grouped)

    if not parts:
        return pd.DataFrame(), stats

    sales_daily = pd.concat(parts, ignore_index=True)
    sales_daily = (
        sales_daily.groupby(
            [DATE_COL, BAKERY_ID_COL, BAKERY_NAME_COL, CITY_COL, PRODUCT_ID_COL, PRODUCT_NAME_COL, CATEGORY_COL],
            as_index=False,
        )
        .agg(
            observed_sales_qty=("observed_sales_qty", "sum"),
            observed_sales_amount=("observed_sales_amount", "sum"),
            sales_rows_count=("sales_rows_count", "sum"),
            sales_hours_count=("sales_hours_count", "max"),
            first_sale_hour=("first_sale_hour", "min"),
            last_sale_hour=("last_sale_hour", "max"),
            price_x_qty=("price_x_qty", "sum"),
            priced_qty=("priced_qty", "sum"),
        )
        .sort_values([BAKERY_ID_COL, PRODUCT_ID_COL, DATE_COL])
        .reset_index(drop=True)
    )
    sales_daily["avg_sales_price"] = np.where(
        sales_daily["priced_qty"] > 0,
        sales_daily["price_x_qty"] / sales_daily["priced_qty"],
        np.nan,
    )
    sales_daily["sales_present_flag"] = 1
    sales_daily["sales_dedup_applied_flag"] = 1
    return sales_daily, stats


def add_sales_context(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    bakery_daily = (
        work.groupby([DATE_COL, BAKERY_ID_COL], as_index=False)
        .agg(
            bakery_sales_qty_total=("observed_sales_qty", "sum"),
            bakery_sales_amount_total=("observed_sales_amount", "sum"),
        )
    )
    bakery_category_daily = (
        work.groupby([DATE_COL, BAKERY_ID_COL, CATEGORY_COL], as_index=False)
        .agg(
            category_sales_qty_in_bakery_day=("observed_sales_qty", "sum"),
            category_sales_amount_in_bakery_day=("observed_sales_amount", "sum"),
        )
    )
    work = work.merge(bakery_daily, on=[DATE_COL, BAKERY_ID_COL], how="left")
    work = work.merge(bakery_category_daily, on=[DATE_COL, BAKERY_ID_COL, CATEGORY_COL], how="left")
    work["sku_sales_share_in_bakery_day"] = work["observed_sales_qty"] / work["bakery_sales_qty_total"].replace(0, np.nan)
    work["sku_sales_share_in_category_day"] = work["observed_sales_qty"] / work["category_sales_qty_in_bakery_day"].replace(0, np.nan)
    return work


def deduplicate_exact_and_find_conflicts(
    df: pd.DataFrame,
    *,
    exact_keys: list[str],
    entity_keys: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    exact_dedup = df.drop_duplicates(subset=exact_keys).copy()
    if exact_dedup.empty:
        return exact_dedup, pd.DataFrame(columns=entity_keys), {
            "raw_rows": int(len(df)),
            "deduped_rows": 0,
            "removed_exact_duplicates": int(len(df)),
            "conflict_groups": 0,
            "conflict_rows": 0,
        }

    nunique = exact_dedup.groupby(entity_keys, dropna=False).nunique(dropna=False)
    conflict_keys = nunique[nunique.gt(1).any(axis=1)].reset_index()[entity_keys]
    if conflict_keys.empty:
        conflicts = exact_dedup.iloc[0:0].copy()
    else:
        conflicts = exact_dedup.merge(conflict_keys, on=entity_keys, how="inner")

    stats = {
        "raw_rows": int(len(df)),
        "deduped_rows": int(len(exact_dedup)),
        "removed_exact_duplicates": int(len(df) - len(exact_dedup)),
        "conflict_groups": int(len(conflict_keys)),
        "conflict_rows": int(len(conflicts)),
    }
    return exact_dedup, conflicts, stats


def aggregate_release_to_sku_day(source_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    df = _read_csv(source_path)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), {"raw_rows": 0, "deduped_rows": 0, "removed_exact_duplicates": 0, "conflict_groups": 0, "conflict_rows": 0}
    df["release_date"] = _coerce_datetime(df, "release_date").dt.normalize()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0)
    exact_keys = ["_UUID", "release_id", "line_id", "release_date", "bakery_id", "product_id", "quantity", "baker_name"]
    entity_keys = ["release_id", "line_id"]
    clean, conflicts, stats = deduplicate_exact_and_find_conflicts(df, exact_keys=exact_keys, entity_keys=entity_keys)
    daily = (
        clean.groupby(["release_date", "bakery_id", "product_id"], as_index=False)
        .agg(
            release_qty=("quantity", "sum"),
            release_rows_count=("quantity", "size"),
            release_bakers_count=("baker_name", "nunique"),
            main_baker_name=("baker_name", _mode_or_nan),
        )
        .rename(columns={"release_date": DATE_COL})
    )
    daily["release_has_data_flag"] = 1
    conflict_keys = conflicts[["release_date", "bakery_id", "product_id"]].drop_duplicates() if not conflicts.empty else pd.DataFrame(columns=["release_date", "bakery_id", "product_id"])
    if not conflict_keys.empty:
        conflict_keys = conflict_keys.rename(columns={"release_date": DATE_COL, "bakery_id": BAKERY_ID_COL, "product_id": PRODUCT_ID_COL})
        daily = daily.merge(conflict_keys.assign(release_conflict_flag=1), on=[DATE_COL, BAKERY_ID_COL, PRODUCT_ID_COL], how="left")
    daily = _ensure_flag_column(daily, "release_conflict_flag")
    return daily, conflicts, stats


def aggregate_moves_to_sku_day(source_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    df = _read_csv(source_path)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), {"raw_rows": 0, "deduped_rows": 0, "removed_exact_duplicates": 0, "conflict_groups": 0, "conflict_rows": 0}
    df["move_date"] = _coerce_datetime(df, "move_date").dt.normalize()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0)
    exact_keys = ["move_id", "move_date", "product_id", "sender_id", "receiver_id", "quantity"]
    entity_keys = ["move_id"]
    clean, conflicts, stats = deduplicate_exact_and_find_conflicts(df, exact_keys=exact_keys, entity_keys=entity_keys)

    incoming = (
        clean.groupby(["move_date", "receiver_id", "product_id"], as_index=False)
        .agg(
            incoming_move_qty=("quantity", "sum"),
            incoming_move_rows_count=("quantity", "size"),
        )
        .rename(columns={"move_date": DATE_COL, "receiver_id": BAKERY_ID_COL, "product_id": PRODUCT_ID_COL})
    )
    outgoing = (
        clean.groupby(["move_date", "sender_id", "product_id"], as_index=False)
        .agg(
            outgoing_move_qty=("quantity", "sum"),
            outgoing_move_rows_count=("quantity", "size"),
        )
        .rename(columns={"move_date": DATE_COL, "sender_id": BAKERY_ID_COL, "product_id": PRODUCT_ID_COL})
    )
    daily = incoming.merge(outgoing, on=[DATE_COL, BAKERY_ID_COL, PRODUCT_ID_COL], how="outer")
    daily["incoming_move_qty"] = daily["incoming_move_qty"].fillna(0.0)
    daily["outgoing_move_qty"] = daily["outgoing_move_qty"].fillna(0.0)
    daily["incoming_move_rows_count"] = daily["incoming_move_rows_count"].fillna(0).astype(int)
    daily["outgoing_move_rows_count"] = daily["outgoing_move_rows_count"].fillna(0).astype(int)
    daily["net_move_qty"] = daily["incoming_move_qty"] - daily["outgoing_move_qty"]
    daily["has_incoming_move_flag"] = (daily["incoming_move_qty"] > 0).astype(int)
    daily["has_outgoing_move_flag"] = (daily["outgoing_move_qty"] > 0).astype(int)
    daily["moves_present_flag"] = ((daily["incoming_move_rows_count"] + daily["outgoing_move_rows_count"]) > 0).astype(int)

    conflict_keys = pd.concat(
        [
            conflicts[["move_date", "receiver_id", "product_id"]].rename(columns={"move_date": DATE_COL, "receiver_id": BAKERY_ID_COL, "product_id": PRODUCT_ID_COL}),
            conflicts[["move_date", "sender_id", "product_id"]].rename(columns={"move_date": DATE_COL, "sender_id": BAKERY_ID_COL, "product_id": PRODUCT_ID_COL}),
        ],
        ignore_index=True,
    ).drop_duplicates() if not conflicts.empty else pd.DataFrame(columns=[DATE_COL, BAKERY_ID_COL, PRODUCT_ID_COL])
    if not conflict_keys.empty:
        daily = daily.merge(conflict_keys.assign(moves_conflict_flag=1), on=[DATE_COL, BAKERY_ID_COL, PRODUCT_ID_COL], how="left")
    daily = _ensure_flag_column(daily, "moves_conflict_flag")
    return daily, conflicts, stats


def build_partner_map(source_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    df = _read_csv(source_path)
    if df.empty:
        empty = pd.DataFrame(columns=[BAKERY_ID_COL, "organization_id", "organization_name", "organization_conflict_flag"])
        return empty, pd.DataFrame(), {"raw_rows": 0, "deduped_rows": 0, "removed_exact_duplicates": 0, "conflict_groups": 0, "conflict_rows": 0}
    exact_keys = ["kkt_id", "kkt_name", "kkt_number", "organization_id", "organization_name", "bakery_id"]
    entity_keys = ["bakery_id", "organization_id", "organization_name"]
    clean, _, dedup_stats = deduplicate_exact_and_find_conflicts(df, exact_keys=exact_keys, entity_keys=entity_keys)

    pair_counts = (
        clean.groupby(["bakery_id", "organization_id", "organization_name"], dropna=False)
        .size()
        .rename("pair_count")
        .reset_index()
    )
    pair_counts = pair_counts.sort_values(["bakery_id", "pair_count", "organization_id", "organization_name"], ascending=[True, False, True, True])
    top_pairs = pair_counts.groupby("bakery_id", as_index=False).first()
    org_counts = pair_counts.groupby("bakery_id", as_index=False).size().rename(columns={"size": "organization_variants"})
    partner_map = top_pairs.merge(org_counts, on="bakery_id", how="left")
    partner_map["organization_conflict_flag"] = (partner_map["organization_variants"] > 1).astype(int)
    conflicts = pair_counts.merge(org_counts[org_counts["organization_variants"] > 1][["bakery_id"]], on="bakery_id", how="inner")
    stats = {
        **dedup_stats,
        "partner_conflict_bakeries": int((partner_map["organization_conflict_flag"] == 1).sum()),
    }
    return partner_map[[ "bakery_id", "organization_id", "organization_name", "organization_conflict_flag"]], conflicts, stats


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["dow"] = work[DATE_COL].dt.dayofweek
    work["day"] = work[DATE_COL].dt.day
    work["month"] = work[DATE_COL].dt.month
    work["iso_week"] = work[DATE_COL].dt.isocalendar().week.astype(int)
    work["is_weekend"] = (work["dow"] >= 5).astype(int)
    work["is_month_start"] = (work["day"] <= 5).astype(int)
    work["is_month_end"] = (work["day"] >= 25).astype(int)
    work["is_payday_week"] = work["day"].isin([4, 5, 6, 19, 20, 21]).astype(int)
    return work


def build_full_panel(df: pd.DataFrame, *, min_observed_days: int = 7) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    panel_keys = [BAKERY_ID_COL, BAKERY_NAME_COL, CITY_COL, PRODUCT_ID_COL, PRODUCT_NAME_COL, CATEGORY_COL]
    groups = []

    for key_values, group in df.groupby(panel_keys, observed=True, dropna=False):
        group = group.sort_values(DATE_COL).copy()
        observed_days = int((group["sales_present_flag"] == 1).sum())
        if observed_days < min_observed_days:
            groups.append(group)
            continue

        date_candidates = [group[DATE_COL]]
        for qty_col in ["release_qty", "incoming_move_qty", "outgoing_move_qty"]:
            if qty_col in group.columns:
                mask = pd.to_numeric(group[qty_col], errors="coerce").fillna(0.0) > 0
                if mask.any():
                    date_candidates.append(group.loc[mask, DATE_COL])

        active_start = min(series.min() for series in date_candidates if not series.empty)
        active_end = max(series.max() for series in date_candidates if not series.empty)
        full_dates = pd.date_range(active_start, active_end, freq="D")

        base = pd.DataFrame({DATE_COL: full_dates})
        for col_name, value in zip(panel_keys, key_values):
            base[col_name] = value

        merged = base.merge(group, on=[DATE_COL] + panel_keys, how="left", suffixes=("", "_orig"))

        zero_fill_cols = [
            "observed_sales_qty",
            "observed_sales_amount",
            "sales_rows_count",
            "sales_hours_count",
            "price_x_qty",
            "priced_qty",
            "sales_present_flag",
            "release_qty",
            "release_rows_count",
            "release_bakers_count",
            "release_has_data_flag",
            "release_present_flag",
            "release_conflict_flag",
            "incoming_move_qty",
            "incoming_move_rows_count",
            "outgoing_move_qty",
            "outgoing_move_rows_count",
            "net_move_qty",
            "has_incoming_move_flag",
            "has_outgoing_move_flag",
            "moves_present_flag",
            "moves_conflict_flag",
            "organization_conflict_flag",
            "sales_dedup_applied_flag",
        ]
        for col in zero_fill_cols:
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

        for float_col in [
            "observed_sales_qty",
            "observed_sales_amount",
            "price_x_qty",
            "priced_qty",
            "release_qty",
            "incoming_move_qty",
            "outgoing_move_qty",
            "net_move_qty",
        ]:
            if float_col in merged.columns:
                merged[float_col] = merged[float_col].astype(float)

        for int_col in [
            "sales_rows_count",
            "sales_hours_count",
            "sales_present_flag",
            "release_rows_count",
            "release_bakers_count",
            "release_has_data_flag",
            "release_present_flag",
            "release_conflict_flag",
            "incoming_move_rows_count",
            "outgoing_move_rows_count",
            "has_incoming_move_flag",
            "has_outgoing_move_flag",
            "moves_present_flag",
            "moves_conflict_flag",
            "organization_conflict_flag",
            "sales_dedup_applied_flag",
        ]:
            if int_col in merged.columns:
                merged[int_col] = merged[int_col].astype(int)

        groups.append(merged)

    panel = pd.concat(groups, ignore_index=True)
    panel = panel.sort_values([BAKERY_ID_COL, PRODUCT_ID_COL, DATE_COL]).reset_index(drop=True)
    return panel


def finalize_dataset(
    sales_daily: pd.DataFrame,
    release_daily: pd.DataFrame,
    moves_daily: pd.DataFrame,
    partner_map: pd.DataFrame,
) -> pd.DataFrame:
    work = add_sales_context(sales_daily)
    if not release_daily.empty:
        work = work.merge(release_daily, on=[DATE_COL, BAKERY_ID_COL, PRODUCT_ID_COL], how="left")
    if not moves_daily.empty:
        work = work.merge(moves_daily, on=[DATE_COL, BAKERY_ID_COL, PRODUCT_ID_COL], how="left")
    if not partner_map.empty:
        work = work.merge(partner_map, on=BAKERY_ID_COL, how="left")

    for col in [
        "release_qty",
        "incoming_move_qty",
        "outgoing_move_qty",
        "net_move_qty",
    ]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
    for col in [
        "release_rows_count",
        "release_bakers_count",
        "incoming_move_rows_count",
        "outgoing_move_rows_count",
    ]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0).astype(int)

    work["release_present_flag"] = _to_bool_flag(work.get("release_has_data_flag", pd.Series([False] * len(work), index=work.index)))
    work["moves_present_flag"] = _to_bool_flag(work.get("moves_present_flag", pd.Series([False] * len(work), index=work.index)))
    work["all_sources_present_flag"] = ((work["sales_present_flag"] == 1) & (work["release_present_flag"] == 1) & (work["moves_present_flag"] == 1)).astype(int)

    work["release_to_sales_ratio"] = work.get("release_qty", 0.0) / work["observed_sales_qty"].replace(0, np.nan)
    work["release_gt_sales_flag"] = (work.get("release_qty", 0.0) > work["observed_sales_qty"]).astype(int)
    work["release_zero_sales_positive_flag"] = ((work.get("release_qty", 0.0) > 0) & (work["observed_sales_qty"] <= 0)).astype(int)
    work["available_qty_proxy"] = work.get("release_qty", 0.0) + work.get("incoming_move_qty", 0.0) - work.get("outgoing_move_qty", 0.0)
    work["available_to_sales_ratio"] = work["available_qty_proxy"] / work["observed_sales_qty"].replace(0, np.nan)

    work["release_conflict_flag"] = _to_bool_flag(work.get("release_conflict_flag", pd.Series([False] * len(work), index=work.index)))
    work["moves_conflict_flag"] = _to_bool_flag(work.get("moves_conflict_flag", pd.Series([False] * len(work), index=work.index)))
    work["organization_conflict_flag"] = _to_bool_flag(work.get("organization_conflict_flag", pd.Series([False] * len(work), index=work.index)))

    work["row_quality_score"] = (
        1.0
        - 0.30 * work["release_conflict_flag"]
        - 0.20 * work["moves_conflict_flag"]
        - 0.10 * work["organization_conflict_flag"]
        - 0.10 * (1 - work["release_present_flag"])
        - 0.05 * (1 - work["moves_present_flag"])
    ).clip(lower=0.0, upper=1.0)
    work = add_calendar_features(work)
    return work.sort_values([BAKERY_ID_COL, PRODUCT_ID_COL, DATE_COL]).reset_index(drop=True)


def build_summary(
    df: pd.DataFrame,
    *,
    panel_df: pd.DataFrame,
    sales_stats: dict[str, float | int],
    release_stats: dict[str, int],
    moves_stats: dict[str, int],
    partner_stats: dict[str, int],
) -> dict[str, object]:
    return {
        "rows": int(len(df)),
        "date_min": None if df.empty else str(df[DATE_COL].min().date()),
        "date_max": None if df.empty else str(df[DATE_COL].max().date()),
        "dates": int(df[DATE_COL].nunique()) if len(df) else 0,
        "bakeries": int(df[BAKERY_ID_COL].nunique()) if len(df) else 0,
        "products": int(df[PRODUCT_ID_COL].nunique()) if len(df) else 0,
        "categories": int(df[CATEGORY_COL].nunique()) if CATEGORY_COL in df.columns and len(df) else 0,
        "panel_rows": int(len(panel_df)),
        "panel_zero_share": round(float((panel_df["observed_sales_qty"] <= 0).mean()), 6) if len(panel_df) else 0.0,
        "release_present_share": round(float(df["release_present_flag"].mean()), 6) if len(df) else 0.0,
        "moves_present_share": round(float(df["moves_present_flag"].mean()), 6) if len(df) else 0.0,
        "mean_row_quality_score": round(float(df["row_quality_score"].mean()), 6) if len(df) else 0.0,
        "sales_dedup": {k: round(float(v), 6) if isinstance(v, float) else int(v) for k, v in sales_stats.items()},
        "release_dedup": {k: int(v) for k, v in release_stats.items()},
        "moves_dedup": {k: int(v) for k, v in moves_stats.items()},
        "partner_map": {k: int(v) for k, v in partner_stats.items()},
    }


def save_outputs(
    output_dir: str | Path,
    audit_dir: str | Path,
    df: pd.DataFrame,
    panel_df: pd.DataFrame,
    summary: dict[str, object],
    *,
    release_conflicts: pd.DataFrame,
    moves_conflicts: pd.DataFrame,
    partner_conflicts: pd.DataFrame,
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    audit = Path(audit_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit.mkdir(parents=True, exist_ok=True)

    dataset_path = out_dir / OUTPUT_NAME
    panel_path = out_dir / PANEL_OUTPUT_NAME
    summary_path = out_dir / SUMMARY_OUTPUT_NAME
    df.to_csv(dataset_path, index=False, encoding="utf-8-sig")
    panel_df.to_csv(panel_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    release_conflicts.to_csv(audit / "release_conflicts.csv", index=False, encoding="utf-8-sig")
    moves_conflicts.to_csv(audit / "moves_conflicts.csv", index=False, encoding="utf-8-sig")
    partner_conflicts.to_csv(audit / "partner_conflicts.csv", index=False, encoding="utf-8-sig")
    return {
        "dataset": dataset_path,
        "panel": panel_path,
        "summary": summary_path,
        "audit_dir": audit,
    }


def build_sku_daily_research_base(
    *,
    sales_path: str | Path,
    release_path: str | Path,
    moves_path: str | Path,
    partner_path: str | Path,
    output_dir: str | Path,
    audit_dir: str | Path,
    chunk_size: int = CHUNK_SIZE,
    panel_min_observed_days: int = 7,
) -> dict[str, Path]:
    sales_daily, sales_stats = aggregate_sales_to_sku_day(sales_path, chunk_size=chunk_size)
    release_daily, release_conflicts, release_stats = aggregate_release_to_sku_day(release_path)
    moves_daily, moves_conflicts, moves_stats = aggregate_moves_to_sku_day(moves_path)
    partner_map, partner_conflicts, partner_stats = build_partner_map(partner_path)

    final_df = finalize_dataset(sales_daily, release_daily, moves_daily, partner_map)
    panel_df = build_full_panel(final_df, min_observed_days=panel_min_observed_days)
    summary = build_summary(
        final_df,
        panel_df=panel_df,
        sales_stats=sales_stats,
        release_stats=release_stats,
        moves_stats=moves_stats,
        partner_stats=partner_stats,
    )
    return save_outputs(
        output_dir,
        audit_dir,
        final_df,
        panel_df,
        summary,
        release_conflicts=release_conflicts,
        moves_conflicts=moves_conflicts,
        partner_conflicts=partner_conflicts,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SKU daily research base for normative demand analytics")
    parser.add_argument("--sales-path", default=str(ROOT / "data" / "raw" / "sales_hrs_all_clickhouse.csv"))
    parser.add_argument("--release-path", default=str(ROOT / "data" / "raw" / "production_release_clickhouse.csv"))
    parser.add_argument("--moves-path", default=str(ROOT / "data" / "raw" / "moves_clickhouse.csv"))
    parser.add_argument("--partner-path", default=str(ROOT / "data" / "raw" / "dim_kkt_clickhouse.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--audit-dir", default=str(ROOT / "reports" / AUDIT_DIR_NAME))
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--panel-min-observed-days", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_sku_daily_research_base(
        sales_path=args.sales_path,
        release_path=args.release_path,
        moves_path=args.moves_path,
        partner_path=args.partner_path,
        output_dir=args.output_dir,
        audit_dir=args.audit_dir,
        chunk_size=args.chunk_size,
        panel_min_observed_days=args.panel_min_observed_days,
    )
    print("=" * 72)
    print("SKU DAILY RESEARCH BASE")
    print("=" * 72)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
