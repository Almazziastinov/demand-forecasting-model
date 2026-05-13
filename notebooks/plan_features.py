from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd


SLOT_PATTERN = re.compile(r"^\s*(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})\s*$")
QTY_PATTERN = re.compile(r"\d+(?:[.,]\d+)?")
SPACE_PATTERN = re.compile(r"\s+")
PAREN_PATTERN = re.compile(r"\s*\([^)]*\)")
TRAILING_VARIANT_PATTERN = re.compile(
    r"(?:\s+(?:к|безд|мск|зкз|заказ|новый|п|постный))+$"
)
TRAILING_NUMBER_PATTERN = re.compile(r"(?:\s+\d+[а-яa-z]*)+$")


SALES_NAME_ALIASES = {
    "жарпицца оригинальная": "жарпицца оригинальная",
    "жарпицца оригинальная к": "жарпицца оригинальная",
    "жарпицца пикантная": "жарпицца пикантная",
    "жарпицца пикантная к": "жарпицца пикантная",
    "жар киш курица": "жар киш курица",
    "жар киш курица к": "жар киш курица",
    "киш курица": "киш курица",
    "киш курица к": "киш курица",
    "клубника и банан": "клубника и банан",
    "конвертик курица": "конвертик курица",
    "конвертик курица к": "конвертик курица",
    "конвертик курица мск": "конвертик курица",
    "кыстыбый п": "кыстыбый",
    "печенье детское 250": "печенье детское",
    "горбуша саго": "горбуша саго",
    "горбуша саго постный": "горбуша саго",
    "капуста и курица": "капуста и курица",
    "капуста и мясо": "капуста и мясо",
    "капустный": "капустный",
    "картофель и мясо": "картофель и мясо",
    "пирог хуплу": "пирог хуплу",
    "пирожок капуста и курица": "пирожок капуста и курица",
    "пирожок с яблоком": "пирожок с яблоком",
    "булочка с яблоком": "булочка с яблоком",
    "пирожок яблоко": "пирожок яблоко",
    "треугольник говядина": "треугольник говядина",
    "треугольник курица": "треугольник курица",
    "треугольник острый": "треугольник острый",
    "трехслойник": "трехслойник",
    "трехслойник мск": "трехслойник",
    "трехслойник новый": "трехслойник",
    "ханский": "ханский",
    "пирог ханский": "ханский",
    "хуплу": "хуплу",
    "элеш с курицей": "элеш с курицей",
}


PLAN_TO_SALES_ALIASES = {
    "жар пицца оригинальная": "жарпицца оригинальная",
    "жар пицца пикантная": "жарпицца пикантная",
    "жар киш с курицей": "жар киш курица",
    "киш с курицей": "киш курица",
    "клубника банан": "клубника и банан",
    "конвертик с курицей": "конвертик курица",
    "пирог горбуша саго": "горбуша саго",
    "пирог капуста курица": "капуста и курица",
    "пирог капуста мясо": "капуста и мясо",
    "пирог капустный": "капустный",
    "пирог картофель мсо чебоксары": "картофель и мясо",
    "пирог хуплу чебоксары": "пирог хуплу",
    "пирожок капуста курица": "пирожок капуста и курица",
    "пирожок/булочка с яблоками": "пирожок с яблоком",
    "хуплу чебоксары": "хуплу",
    "элеш": "элеш с курицей",
}


def normalize_product_name(name: str) -> str:
    """Normalize sales-side product names."""
    value = "" if name is None else str(name)
    value = value.replace("\xa0", " ").strip().lower()
    value = PAREN_PATTERN.sub("", value)
    value = SPACE_PATTERN.sub(" ", value)
    value = TRAILING_VARIANT_PATTERN.sub("", value)
    value = TRAILING_NUMBER_PATTERN.sub("", value)
    value = SPACE_PATTERN.sub(" ", value).strip()
    return SALES_NAME_ALIASES.get(value, value)


def normalize_plan_product_name(name: str) -> str:
    """Normalize plan-side product names to sales-side canonical names."""
    value = normalize_product_name(name)
    return PLAN_TO_SALES_ALIASES.get(value, value)


def add_normalized_product_column(
    df: pd.DataFrame,
    product_col: str,
    *,
    output_col: str = "product_name_norm",
) -> pd.DataFrame:
    result = df.copy()
    result[output_col] = result[product_col].map(normalize_product_name)
    return result


def parse_slot_label(slot_label: str) -> tuple[int, int]:
    match = SLOT_PATTERN.match(str(slot_label))
    if not match:
        raise ValueError(f"Invalid slot label: {slot_label!r}")
    start_hour = int(match.group(1))
    end_hour = int(match.group(3))
    return start_hour, end_hour


def parse_plan_quantity(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    match = QTY_PATTERN.search(text)
    if not match:
        return None
    return float(match.group(0).replace(",", "."))


def _clean_raw_plan(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", dtype=str)
    df.columns = [str(col).strip() for col in df.columns]

    normalized_cols = [col.lower() for col in df.columns[:2]]
    has_expected_header = normalized_cols == ["стол", "наименование"]
    if not has_expected_header:
        header_row_idx = None
        for idx, row in df.iterrows():
            first = "" if pd.isna(row.iloc[0]) else str(row.iloc[0]).strip().lower()
            second = "" if pd.isna(row.iloc[1]) else str(row.iloc[1]).strip().lower()
            if first == "стол" and second == "наименование":
                header_row_idx = idx
                break

        if header_row_idx is not None:
            new_columns = [
                "" if pd.isna(value) else str(value).strip()
                for value in df.iloc[header_row_idx].tolist()
            ]
            df = df.iloc[header_row_idx + 1 :].copy()
            df.columns = new_columns
        else:
            raise ValueError(
                f"Could not find plan header row with 'Стол' and 'Наименование' in {path!s}"
            )

    keep_cols = []
    for idx, col in enumerate(df.columns):
        col_text = str(col).strip()
        if idx < 2:
            keep_cols.append(col)
            continue
        if SLOT_PATTERN.match(col_text):
            keep_cols.append(col)

    df = df.loc[:, keep_cols].copy()
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"": pd.NA, "nan": pd.NA})
    df = df[df.iloc[:, 0].notna() & df.iloc[:, 1].notna()].copy()
    return df


def load_plan_schedule(path: str | Path, schedule_name: str) -> pd.DataFrame:
    """
    Parse a bakery plan file into a long dataframe by production slot.

    Output columns:
    - schedule_name
    - table_name
    - raw_product_name
    - product_name_norm
    - slot_label
    - slot_start_hour
    - slot_end_hour
    - slot_duration_hours
    - slot_mid_hour
    - planned_qty
    """
    df = _clean_raw_plan(path)
    base_cols = list(df.columns[:2])
    slot_cols = list(df.columns[2:])

    melted = df.melt(
        id_vars=base_cols,
        value_vars=slot_cols,
        var_name="slot_label",
        value_name="planned_qty_raw",
    )
    melted["planned_qty"] = melted["planned_qty_raw"].map(parse_plan_quantity)
    melted = melted[melted["planned_qty"].notna()].copy()
    melted = melted.rename(
        columns={
            base_cols[0]: "table_name",
            base_cols[1]: "raw_product_name",
        }
    )

    slot_bounds = melted["slot_label"].map(parse_slot_label)
    melted["slot_start_hour"] = slot_bounds.map(lambda x: x[0])
    melted["slot_end_hour"] = slot_bounds.map(lambda x: x[1])
    melted["slot_duration_hours"] = melted["slot_end_hour"] - melted["slot_start_hour"]
    melted["slot_mid_hour"] = (
        melted["slot_start_hour"] + melted["slot_duration_hours"] / 2.0
    )
    melted["schedule_name"] = schedule_name
    melted["plan_product_name_norm"] = melted["raw_product_name"].map(normalize_product_name)
    melted["product_name_norm"] = melted["raw_product_name"].map(normalize_plan_product_name)

    melted = melted[
        [
            "schedule_name",
            "table_name",
            "raw_product_name",
            "plan_product_name_norm",
            "product_name_norm",
            "slot_label",
            "slot_start_hour",
            "slot_end_hour",
            "slot_duration_hours",
            "slot_mid_hour",
            "planned_qty",
        ]
    ].sort_values(["product_name_norm", "slot_start_hour", "table_name"])

    return melted.reset_index(drop=True)


def load_plan_schedules(
    weekday_path: str | Path,
    weekend_path: str | Path,
) -> pd.DataFrame:
    """
    Combine weekday and weekend schedules into one dataframe.

    Assumption:
    - `weekday_path` contains the weekday schedule.
    - `weekend_path` contains the weekend schedule.
    """
    weekday_df = load_plan_schedule(weekday_path, schedule_name="weekday")
    weekend_df = load_plan_schedule(weekend_path, schedule_name="weekend")
    combined = pd.concat([weekday_df, weekend_df], ignore_index=True)
    combined["is_weekend"] = combined["schedule_name"].eq("weekend")
    return combined


def expand_plan_to_dates(
    plans_df: pd.DataFrame,
    dates: Iterable[object],
) -> pd.DataFrame:
    """
    Expand slot plans to concrete dates using weekday/weekend routing.
    """
    calendar = pd.DataFrame({"date": pd.to_datetime(list(dates))}).drop_duplicates()
    calendar["dow"] = calendar["date"].dt.dayofweek
    calendar["is_weekend"] = calendar["dow"] >= 5

    merged = calendar.merge(plans_df, on="is_weekend", how="left")
    merged["schedule_match"] = merged["schedule_name"].eq(
        merged["is_weekend"].map({True: "weekend", False: "weekday"})
    )
    merged = merged[merged["schedule_match"]].drop(columns=["schedule_match"])
    return merged.reset_index(drop=True)


def build_hourly_plan_features(
    plans_by_date: pd.DataFrame,
    *,
    allocate_strategy: str = "arrival",
) -> pd.DataFrame:
    """
    Convert slot plans to hourly features suitable for merging with sales.

    Strategies:
    - `arrival`: assign the full slot quantity to the first hour when the batch
      should be available on the shelf, i.e. `slot_end_hour`.
    - `start`: assign the full slot quantity to the slot start hour.
    - `uniform`: spread the quantity evenly across all hours in the slot.
    """
    if allocate_strategy not in {"arrival", "start", "uniform"}:
        raise ValueError("allocate_strategy must be 'arrival', 'start' or 'uniform'")

    work = plans_by_date.copy()
    hour_rows: list[dict[str, object]] = []

    for row in work.itertuples(index=False):
        hours = list(range(int(row.slot_start_hour), int(row.slot_end_hour)))
        if not hours:
            continue
        if allocate_strategy == "arrival":
            allocations = {int(row.slot_end_hour): float(row.planned_qty)}
        elif allocate_strategy == "start":
            allocations = {hours[0]: float(row.planned_qty)}
        else:
            qty_per_hour = float(row.planned_qty) / len(hours)
            allocations = {hour: qty_per_hour for hour in hours}

        for hour, qty in allocations.items():
            hour_rows.append(
                {
                    "date": row.date,
                    "dow": row.dow,
                    "is_weekend": row.is_weekend,
                    "schedule_name": row.schedule_name,
                    "table_name": row.table_name,
                    "raw_product_name": row.raw_product_name,
                    "plan_product_name_norm": row.plan_product_name_norm,
                    "product_name_norm": row.product_name_norm,
                    "hour": hour,
                    "planned_qty_hour": qty,
                    "source_slot_label": row.slot_label,
                    "source_slot_start_hour": row.slot_start_hour,
                    "source_slot_end_hour": row.slot_end_hour,
                    "availability_hour": int(row.slot_end_hour),
                }
            )

    hourly = pd.DataFrame(hour_rows)
    if hourly.empty:
        return hourly

    hourly = hourly.sort_values(["date", "product_name_norm", "hour", "table_name"])
    grouped = (
        hourly.groupby(
            [
                "date",
                "dow",
                "is_weekend",
                "schedule_name",
                "product_name_norm",
                "raw_product_name",
                "plan_product_name_norm",
                "hour",
            ],
            as_index=False,
        )
        .agg(
            planned_qty_hour=("planned_qty_hour", "sum"),
            table_count=("table_name", "nunique"),
            source_slot_count=("source_slot_label", "nunique"),
            latest_slot_start_hour=("source_slot_start_hour", "max"),
            latest_slot_end_hour=("source_slot_end_hour", "max"),
            availability_hour=("availability_hour", "max"),
        )
        .sort_values(["date", "product_name_norm", "hour"])
    )

    grouped["planned_qty_cum"] = grouped.groupby(
        ["date", "product_name_norm"]
    )["planned_qty_hour"].cumsum()
    grouped["has_planned_replenishment"] = grouped["planned_qty_hour"] > 0
    grouped["hours_since_last_bake_slot"] = grouped["hour"] - grouped["latest_slot_start_hour"]
    grouped["hours_since_shelf_arrival"] = grouped["hour"] - grouped["latest_slot_end_hour"]

    return grouped.reset_index(drop=True)


def load_hourly_plan_features(
    dates: Iterable[object],
    *,
    weekday_path: str | Path,
    weekend_path: str | Path,
    allocate_strategy: str = "arrival",
) -> pd.DataFrame:
    plans_df = load_plan_schedules(
        weekday_path=weekday_path,
        weekend_path=weekend_path,
    )
    plans_by_date = expand_plan_to_dates(plans_df, dates)
    return build_hourly_plan_features(
        plans_by_date,
        allocate_strategy=allocate_strategy,
    )


def merge_sales_with_plan_features(
    sales_df: pd.DataFrame,
    *,
    date_col: str,
    hour_col: str,
    product_col: str,
    weekday_path: str | Path,
    weekend_path: str | Path,
    allocate_strategy: str = "arrival",
    product_norm_col: str = "product_name_norm",
) -> pd.DataFrame:
    """
    Merge sales rows with hourly plan features.

    The sales side remains the source of truth for SKU naming. Plan SKU names are
    translated to sales-side canonical names before the merge.
    """
    work = add_normalized_product_column(
        sales_df,
        product_col=product_col,
        output_col=product_norm_col,
    )
    work = work.copy()
    work["_plan_date"] = pd.to_datetime(work[date_col]).dt.normalize()

    plan_hourly = load_hourly_plan_features(
        dates=work["_plan_date"].dropna().unique(),
        weekday_path=weekday_path,
        weekend_path=weekend_path,
        allocate_strategy=allocate_strategy,
    )

    merged = work.merge(
        plan_hourly,
        left_on=["_plan_date", hour_col, product_norm_col],
        right_on=["date", "hour", "product_name_norm"],
        how="left",
        suffixes=("", "_plan"),
    )

    fill_zero_cols = [
        "planned_qty_hour",
        "planned_qty_cum",
        "table_count",
        "source_slot_count",
    ]
    for col in fill_zero_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    fill_false_cols = ["has_planned_replenishment", "is_weekend"]
    for col in fill_false_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(False).astype(bool)

    return merged


def add_plan_execution_features(
    df: pd.DataFrame,
    *,
    date_col: str,
    hour_col: str,
    sales_qty_col: str,
    product_col: str | None = None,
    product_norm_col: str = "product_name_norm",
    planned_qty_hour_col: str = "planned_qty_hour",
    planned_qty_cum_col: str = "planned_qty_cum",
    latest_slot_start_col: str = "latest_slot_start_hour",
    latest_slot_end_col: str = "latest_slot_end_hour",
) -> pd.DataFrame:
    """
    Add plan-vs-sales execution features to an hourly SKU dataframe.

    Expected input:
    - one row per date x hour x SKU
    - sales quantity in `sales_qty_col`
    - plan columns already merged in, typically via `load_hourly_plan_features()`
      or `merge_sales_with_plan_features()`

    Output features:
    - `sales_qty_cum`
    - `sales_qty_cum_before_hour`
    - `planned_qty_cum_before_hour`
    - `plan_balance_before_hour`
    - `plan_balance_after_hour`
    - `has_plan_before_hour`
    - `has_plan_by_hour`
    - `plan_depleted_before_hour`
    - `plan_depleted_after_hour`
    - forward-filled latest slot markers and recency-from-plan features
    """
    work = df.copy()
    if product_norm_col not in work.columns:
        if product_col is None:
            raise ValueError(
                f"{product_norm_col!r} is missing; provide product_col to normalize SKU names."
            )
        work = add_normalized_product_column(
            work,
            product_col=product_col,
            output_col=product_norm_col,
        )

    work["_plan_date"] = pd.to_datetime(work[date_col]).dt.normalize()
    work[hour_col] = pd.to_numeric(work[hour_col], errors="coerce")
    work[sales_qty_col] = pd.to_numeric(work[sales_qty_col], errors="coerce").fillna(0.0)

    if planned_qty_hour_col not in work.columns:
        work[planned_qty_hour_col] = 0.0
    work[planned_qty_hour_col] = pd.to_numeric(
        work[planned_qty_hour_col], errors="coerce"
    ).fillna(0.0)

    if planned_qty_cum_col not in work.columns:
        work[planned_qty_cum_col] = (
            work.sort_values([hour_col])
            .groupby(["_plan_date", product_norm_col])[planned_qty_hour_col]
            .cumsum()
        )
    else:
        work[planned_qty_cum_col] = pd.to_numeric(
            work[planned_qty_cum_col], errors="coerce"
        ).fillna(0.0)

    group_keys = ["_plan_date", product_norm_col]
    work = work.sort_values(group_keys + [hour_col]).copy()

    work["sales_qty_cum"] = work.groupby(group_keys)[sales_qty_col].cumsum()
    work["sales_qty_cum_before_hour"] = work["sales_qty_cum"] - work[sales_qty_col]
    work["planned_qty_cum_before_hour"] = work[planned_qty_cum_col] - work[planned_qty_hour_col]

    work["plan_balance_before_hour"] = (
        work["planned_qty_cum_before_hour"] - work["sales_qty_cum_before_hour"]
    )
    work["plan_balance_after_hour"] = work[planned_qty_cum_col] - work["sales_qty_cum"]

    work["has_plan_before_hour"] = work["planned_qty_cum_before_hour"] > 0
    work["has_plan_by_hour"] = work[planned_qty_cum_col] > 0
    work["plan_depleted_before_hour"] = work["has_plan_before_hour"] & (
        work["plan_balance_before_hour"] <= 0
    )
    work["plan_depleted_after_hour"] = work["has_plan_by_hour"] & (
        work["plan_balance_after_hour"] <= 0
    )

    for slot_col, output_col in (
        (latest_slot_start_col, "latest_slot_start_hour_ffill"),
        (latest_slot_end_col, "latest_slot_end_hour_ffill"),
    ):
        if slot_col in work.columns:
            slot_values = pd.to_numeric(work[slot_col], errors="coerce")
            work[output_col] = slot_values.groupby(
                [work["_plan_date"], work[product_norm_col]]
            ).ffill()

    if "latest_slot_start_hour_ffill" in work.columns:
        work["hours_since_last_planned_bake_start"] = (
            work[hour_col] - work["latest_slot_start_hour_ffill"]
        )
    if "latest_slot_end_hour_ffill" in work.columns:
        work["hours_since_last_planned_arrival"] = (
            work[hour_col] - work["latest_slot_end_hour_ffill"]
        )

    return work


def build_plan_sales_match_report(
    sales_df: pd.DataFrame,
    sales_product_col: str,
    *,
    weekday_path: str | Path,
    weekend_path: str | Path,
) -> pd.DataFrame:
    plan_df = load_plan_schedules(
        weekday_path=weekday_path,
        weekend_path=weekend_path,
    )
    plan_names = (
        plan_df[["raw_product_name", "plan_product_name_norm", "product_name_norm"]]
        .drop_duplicates()
        .sort_values(["product_name_norm", "raw_product_name"])
    )

    sales_names = (
        sales_df[[sales_product_col]]
        .dropna()
        .drop_duplicates()
        .rename(columns={sales_product_col: "sales_raw_name"})
    )
    sales_names["product_name_norm"] = sales_names["sales_raw_name"].map(normalize_product_name)

    report = plan_names.merge(
        sales_names,
        on="product_name_norm",
        how="left",
    )
    report["matched_in_sales"] = report["sales_raw_name"].notna()
    return report.sort_values(["matched_in_sales", "product_name_norm", "sales_raw_name"]).reset_index(drop=True)
