from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CHUNK_SIZE = 500_000

CITY = "Казань"
CATEGORY = "Выпечка сытная"

DAILY_OUTPUT_NAME = "kazan_sitnaya_daily_sample.csv"
BAKERY_DAILY_OUTPUT_NAME = "kazan_bakery_daily_sample.csv"
BAKERY_CATEGORY_DAILY_OUTPUT_NAME = "kazan_sitnaya_bakery_category_daily_sample.csv"
HOURLY_OUTPUT_NAME = "kazan_sitnaya_hourly_sample.csv"
BAKERY_METRICS_OUTPUT_NAME = "kazan_sitnaya_bakery_selection.csv"
SUMMARY_OUTPUT_NAME = "kazan_sitnaya_sample_summary.json"


def _read_filtered_chunks(
    path: str | Path,
    *,
    usecols: list[str],
    chunk_size: int,
    city: str,
    category: str,
) -> list[pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    reader = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=lambda col: col in usecols,
        chunksize=chunk_size,
    )
    for chunk in reader:
        if "city" in chunk.columns:
            mask = chunk["city"].astype(str).eq(city)
        else:
            mask = pd.Series(True, index=chunk.index)
        mask &= chunk["category_name"].astype(str).eq(category)
        filtered = chunk.loc[mask].copy()
        if not filtered.empty:
            parts.append(filtered)
    return parts


def load_daily_scope(
    daily_path: str | Path,
    *,
    city: str = CITY,
    category: str = CATEGORY,
    chunk_size: int = CHUNK_SIZE,
) -> pd.DataFrame:
    usecols = [
        "date",
        "bakery_id",
        "bakery_name",
        "city",
        "product_id",
        "product_name",
        "category_name",
        "observed_sales_qty",
        "sales_hours_count",
        "sales_present_flag",
        "release_qty",
        "release_present_flag",
        "row_quality_score",
    ]
    parts = _read_filtered_chunks(
        daily_path,
        usecols=usecols,
        chunk_size=chunk_size,
        city=city,
        category=category,
    )
    if not parts:
        return pd.DataFrame(columns=usecols)
    daily = pd.concat(parts, ignore_index=True)
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily["observed_sales_qty"] = pd.to_numeric(daily["observed_sales_qty"], errors="coerce").fillna(0.0)
    daily["sales_hours_count"] = pd.to_numeric(daily["sales_hours_count"], errors="coerce").fillna(0)
    daily["sales_present_flag"] = pd.to_numeric(daily["sales_present_flag"], errors="coerce").fillna(0).astype(int)
    daily["release_qty"] = pd.to_numeric(daily["release_qty"], errors="coerce").fillna(0.0)
    daily["release_present_flag"] = pd.to_numeric(daily["release_present_flag"], errors="coerce").fillna(0).astype(int)
    daily["row_quality_score"] = pd.to_numeric(daily["row_quality_score"], errors="coerce")
    return daily.dropna(subset=["date", "bakery_id", "product_id"]).copy()


def compute_bakery_selection_metrics(daily_scope: pd.DataFrame) -> pd.DataFrame:
    if daily_scope.empty:
        return pd.DataFrame(
            columns=[
                "bakery_id",
                "bakery_name",
                "city",
                "history_days",
                "active_days",
                "sales_row_count",
                "unique_sku_count",
                "positive_sku_days",
                "positive_sku_day_share",
                "mean_sales_hours_count",
                "release_present_share",
                "mean_row_quality_score",
                "completeness_score",
            ]
        )

    bakery_day = (
        daily_scope.groupby(["bakery_id", "date"], as_index=False)
        .agg(
            bakery_sales_qty=("observed_sales_qty", "sum"),
            bakery_release_qty=("release_qty", "sum"),
            bakery_sales_hours_count=("sales_hours_count", "max"),
            bakery_row_quality_score=("row_quality_score", "mean"),
        )
    )

    metrics = (
        daily_scope.groupby(["bakery_id", "bakery_name", "city"], as_index=False)
        .agg(
            history_days=("date", "nunique"),
            sales_row_count=("product_id", "size"),
            unique_sku_count=("product_id", "nunique"),
            positive_sku_days=("sales_present_flag", "sum"),
            release_present_days=("release_present_flag", "sum"),
            mean_row_quality_score=("row_quality_score", "mean"),
        )
    )

    day_metrics = (
        bakery_day.groupby("bakery_id", as_index=False)
        .agg(
            active_days=("bakery_sales_qty", lambda s: int((s > 0).sum())),
            positive_day_share=("bakery_sales_qty", lambda s: float((s > 0).mean())),
            mean_sales_hours_count=("bakery_sales_hours_count", "mean"),
            release_present_share=("bakery_release_qty", lambda s: float((s > 0).mean())),
        )
    )

    metrics = metrics.merge(day_metrics, on="bakery_id", how="left")
    metrics["positive_sku_day_share"] = metrics["positive_sku_days"] / metrics["sales_row_count"].replace(0, pd.NA)

    score_components = [
        "history_days",
        "active_days",
        "unique_sku_count",
        "positive_sku_day_share",
        "mean_sales_hours_count",
        "release_present_share",
        "mean_row_quality_score",
    ]
    for col in score_components:
        col_min = metrics[col].min()
        col_max = metrics[col].max()
        if pd.isna(col_min) or pd.isna(col_max) or col_max == col_min:
            metrics[f"{col}_score"] = 1.0
        else:
            metrics[f"{col}_score"] = (metrics[col] - col_min) / (col_max - col_min)

    metrics["completeness_score"] = (
        0.25 * metrics["history_days_score"]
        + 0.25 * metrics["active_days_score"]
        + 0.15 * metrics["unique_sku_count_score"]
        + 0.10 * metrics["positive_sku_day_share_score"]
        + 0.10 * metrics["mean_sales_hours_count_score"]
        + 0.05 * metrics["release_present_share_score"]
        + 0.10 * metrics["mean_row_quality_score_score"]
    )

    return metrics.sort_values(
        ["completeness_score", "history_days", "active_days", "unique_sku_count", "bakery_id"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)


def select_top_bakeries(selection_metrics: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    return selection_metrics.head(top_n).copy()


def build_daily_sample(daily_scope: pd.DataFrame, selected_bakery_ids: set[str]) -> pd.DataFrame:
    sample = daily_scope[daily_scope["bakery_id"].astype(str).isin(selected_bakery_ids)].copy()
    return sample.sort_values(["bakery_id", "product_id", "date"]).reset_index(drop=True)


def build_bakery_category_daily_sample(daily_sample: pd.DataFrame) -> pd.DataFrame:
    if daily_sample.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "bakery_id",
                "bakery_name",
                "city",
                "category_name",
                "category_sales_qty",
                "category_release_qty",
                "active_sku_count",
                "selling_sku_count",
                "mean_row_quality_score",
                "bakery_total_sales_qty",
                "category_share_in_bakery_total",
            ]
        )

    grouped = (
        daily_sample.groupby(["date", "bakery_id", "bakery_name", "city", "category_name"], as_index=False)
        .agg(
            category_sales_qty=("observed_sales_qty", "sum"),
            category_release_qty=("release_qty", "sum"),
            active_sku_count=("product_id", "nunique"),
            selling_sku_count=("sales_present_flag", "sum"),
            mean_row_quality_score=("row_quality_score", "mean"),
            bakery_total_sales_qty=("bakery_total_sales_qty", "first"),
        )
    )
    grouped["category_share_in_bakery_total"] = (
        grouped["category_sales_qty"] / grouped["bakery_total_sales_qty"].replace(0, pd.NA)
    )
    return grouped.sort_values(["bakery_id", "date", "category_name"]).reset_index(drop=True)


def load_bakery_daily_sample(
    bakery_daily_path: str | Path,
    *,
    city: str,
    selected_bakery_ids: set[str],
    chunk_size: int,
) -> pd.DataFrame:
    usecols = [
        "date",
        "bakery_id",
        "bakery_name",
        "city",
        "bakery_sales",
        "line_amount_sum",
        "priced_quantity",
        "price_x_qty_sum",
        "avg_price",
        "dow",
        "day",
        "month",
        "iso_week",
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "is_payday_week",
    ]
    parts: list[pd.DataFrame] = []
    reader = pd.read_csv(
        bakery_daily_path,
        encoding="utf-8-sig",
        usecols=lambda col: col in usecols,
        chunksize=chunk_size,
    )
    selected_ids = {str(value) for value in selected_bakery_ids}
    for chunk in reader:
        mask = chunk["city"].astype(str).eq(city)
        mask &= chunk["bakery_id"].astype(str).isin(selected_ids)
        filtered = chunk.loc[mask].copy()
        if not filtered.empty:
            parts.append(filtered)

    if not parts:
        return pd.DataFrame(columns=usecols)

    bakery_daily = pd.concat(parts, ignore_index=True)
    bakery_daily["date"] = pd.to_datetime(bakery_daily["date"], errors="coerce")
    bakery_daily["bakery_sales"] = pd.to_numeric(bakery_daily["bakery_sales"], errors="coerce").fillna(0.0)
    return bakery_daily.sort_values(["bakery_id", "date"]).reset_index(drop=True)


def enrich_daily_sample_with_bakery_totals(
    daily_sample: pd.DataFrame,
    bakery_daily_sample: pd.DataFrame,
) -> pd.DataFrame:
    if daily_sample.empty or bakery_daily_sample.empty:
        return daily_sample.copy()

    merge_cols = ["date", "bakery_id", "bakery_name", "city"]
    bakery_cols = merge_cols + ["bakery_sales", "avg_price"]
    work = daily_sample.merge(
        bakery_daily_sample[bakery_cols],
        on=merge_cols,
        how="left",
        validate="many_to_one",
    )
    work = work.rename(
        columns={
            "bakery_sales": "bakery_total_sales_qty",
            "avg_price": "bakery_avg_price_all_categories",
        }
    )
    work["sku_sales_share_in_bakery_total"] = work["observed_sales_qty"] / work["bakery_total_sales_qty"].replace(0, pd.NA)
    return work


def build_hourly_sample(
    hourly_path: str | Path,
    *,
    selected_bakery_ids: set[str],
    category: str = CATEGORY,
    chunk_size: int = CHUNK_SIZE,
) -> pd.DataFrame:
    usecols = [
        "date",
        "dow",
        "hour",
        "bakery_id",
        "bakery_name",
        "product_id",
        "product_name",
        "category_name",
        "sku_hour_sales",
        "bakery_hour_sales",
        "sku_share_in_hour",
    ]
    parts: list[pd.DataFrame] = []
    reader = pd.read_csv(
        hourly_path,
        encoding="utf-8-sig",
        usecols=lambda col: col in usecols,
        chunksize=chunk_size,
    )
    selected_ids = {str(value) for value in selected_bakery_ids}
    for chunk in reader:
        mask = chunk["category_name"].astype(str).eq(category)
        mask &= chunk["bakery_id"].astype(str).isin(selected_ids)
        filtered = chunk.loc[mask].copy()
        if not filtered.empty:
            parts.append(filtered)

    if not parts:
        return pd.DataFrame(columns=usecols)

    hourly = pd.concat(parts, ignore_index=True)
    hourly["date"] = pd.to_datetime(hourly["date"], errors="coerce")
    return hourly.sort_values(["bakery_id", "product_id", "date", "hour"]).reset_index(drop=True)


def build_summary(
    *,
    city: str,
    category: str,
    top_n: int,
    selection_metrics: pd.DataFrame,
    selected_bakeries: pd.DataFrame,
    daily_sample: pd.DataFrame,
    bakery_daily_sample: pd.DataFrame,
    bakery_category_daily_sample: pd.DataFrame,
    hourly_sample: pd.DataFrame,
) -> dict[str, object]:
    return {
        "city": city,
        "category": category,
        "requested_bakeries": int(top_n),
        "selected_bakeries": int(len(selected_bakeries)),
        "selection_score_min": 0.0 if selected_bakeries.empty else round(float(selected_bakeries["completeness_score"].min()), 6),
        "selection_score_max": 0.0 if selected_bakeries.empty else round(float(selected_bakeries["completeness_score"].max()), 6),
        "candidate_bakeries_in_scope": int(selection_metrics["bakery_id"].nunique()) if not selection_metrics.empty else 0,
        "daily_rows": int(len(daily_sample)),
        "daily_dates": int(daily_sample["date"].nunique()) if not daily_sample.empty else 0,
        "daily_products": int(daily_sample["product_id"].nunique()) if not daily_sample.empty else 0,
        "bakery_daily_rows": int(len(bakery_daily_sample)),
        "bakery_daily_dates": int(bakery_daily_sample["date"].nunique()) if not bakery_daily_sample.empty else 0,
        "bakery_category_daily_rows": int(len(bakery_category_daily_sample)),
        "bakery_category_daily_dates": int(bakery_category_daily_sample["date"].nunique()) if not bakery_category_daily_sample.empty else 0,
        "hourly_rows": int(len(hourly_sample)),
        "hourly_dates": int(hourly_sample["date"].nunique()) if not hourly_sample.empty else 0,
        "hourly_products": int(hourly_sample["product_id"].nunique()) if not hourly_sample.empty else 0,
    }


def save_outputs(
    output_dir: str | Path,
    *,
    daily_sample: pd.DataFrame,
    bakery_daily_sample: pd.DataFrame,
    bakery_category_daily_sample: pd.DataFrame,
    hourly_sample: pd.DataFrame,
    bakery_metrics: pd.DataFrame,
    summary: dict[str, object],
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    daily_path = out_dir / DAILY_OUTPUT_NAME
    bakery_daily_path = out_dir / BAKERY_DAILY_OUTPUT_NAME
    bakery_category_daily_path = out_dir / BAKERY_CATEGORY_DAILY_OUTPUT_NAME
    hourly_path = out_dir / HOURLY_OUTPUT_NAME
    metrics_path = out_dir / BAKERY_METRICS_OUTPUT_NAME
    summary_path = out_dir / SUMMARY_OUTPUT_NAME

    daily_sample.to_csv(daily_path, index=False, encoding="utf-8-sig")
    bakery_daily_sample.to_csv(bakery_daily_path, index=False, encoding="utf-8-sig")
    bakery_category_daily_sample.to_csv(bakery_category_daily_path, index=False, encoding="utf-8-sig")
    hourly_sample.to_csv(hourly_path, index=False, encoding="utf-8-sig")
    bakery_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "daily_sample": daily_path,
        "bakery_daily_sample": bakery_daily_path,
        "bakery_category_daily_sample": bakery_category_daily_path,
        "hourly_sample": hourly_path,
        "bakery_selection": metrics_path,
        "summary": summary_path,
    }


def build_kazan_sitnaya_sample(
    *,
    daily_path: str | Path,
    bakery_daily_path: str | Path,
    hourly_path: str | Path,
    output_dir: str | Path,
    city: str = CITY,
    category: str = CATEGORY,
    top_n_bakeries: int = 30,
    chunk_size: int = CHUNK_SIZE,
) -> dict[str, Path]:
    daily_scope = load_daily_scope(
        daily_path,
        city=city,
        category=category,
        chunk_size=chunk_size,
    )
    bakery_metrics = compute_bakery_selection_metrics(daily_scope)
    selected_bakeries = select_top_bakeries(bakery_metrics, top_n=top_n_bakeries)
    selected_bakery_ids = set(selected_bakeries["bakery_id"].astype(str))

    bakery_daily_sample = load_bakery_daily_sample(
        bakery_daily_path,
        city=city,
        selected_bakery_ids=selected_bakery_ids,
        chunk_size=chunk_size,
    )
    daily_sample = enrich_daily_sample_with_bakery_totals(
        build_daily_sample(daily_scope, selected_bakery_ids),
        bakery_daily_sample,
    )
    bakery_category_daily_sample = build_bakery_category_daily_sample(daily_sample)
    hourly_sample = build_hourly_sample(
        hourly_path,
        selected_bakery_ids=selected_bakery_ids,
        category=category,
        chunk_size=chunk_size,
    )
    summary = build_summary(
        city=city,
        category=category,
        top_n=top_n_bakeries,
        selection_metrics=bakery_metrics,
        selected_bakeries=selected_bakeries,
        daily_sample=daily_sample,
        bakery_daily_sample=bakery_daily_sample,
        bakery_category_daily_sample=bakery_category_daily_sample,
        hourly_sample=hourly_sample,
    )
    return save_outputs(
        output_dir,
        daily_sample=daily_sample,
        bakery_daily_sample=bakery_daily_sample,
        bakery_category_daily_sample=bakery_category_daily_sample,
        hourly_sample=hourly_sample,
        bakery_metrics=selected_bakeries,
        summary=summary,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Kazan sitnaya sample for multi-level normative analysis")
    parser.add_argument("--daily-path", default=str(ROOT / "data" / "processed" / "sku_daily_research_base.csv"))
    parser.add_argument("--bakery-daily-path", default=str(ROOT / "data" / "processed" / "bakery_daily_sales.csv"))
    parser.add_argument("--hourly-path", default=str(ROOT / "data" / "processed" / "sku_hour_share_profile_daily.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--city", default=CITY)
    parser.add_argument("--category", default=CATEGORY)
    parser.add_argument("--top-n-bakeries", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_kazan_sitnaya_sample(
        daily_path=args.daily_path,
        bakery_daily_path=args.bakery_daily_path,
        hourly_path=args.hourly_path,
        output_dir=args.output_dir,
        city=args.city,
        category=args.category,
        top_n_bakeries=args.top_n_bakeries,
        chunk_size=args.chunk_size,
    )
    print("=" * 72)
    print("KAZAN SITNAYA SAMPLE")
    print("=" * 72)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
