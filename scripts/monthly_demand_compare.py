"""
Extract the last full calendar month from sales_hrs_all.csv and compare
actual vs expected sales for one bakery-product pair.

Expected sales are estimated from the bakery's monthly average hourly
product share:

    expected_qty(hour) = bakery_total_qty(hour) * avg_product_share(hour)

Set TARGET_BAKERY and TARGET_PRODUCT below before running the comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SOURCE_CSV = ROOT / "data" / "raw" / "sales_hrs_all.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports" / "monthly_demand_review"

DATE_COL = "Дата продажи"
DATETIME_COL = "Дата время чека"
EVENT_COL = "Вид события по кассе"
BAKERY_COL = "Касса.Торговая точка"
CATEGORY_COL = "Категория"
PRODUCT_COL = "Номенклатура"
QTY_COL = "Кол-во"

# Fill these in manually before running the comparison.
TARGET_BAKERY = ""
TARGET_PRODUCT = ""

CHUNK_SIZE = 500_000
SALES_EVENT = "Продажа"


def sanitize_filename(value: str) -> str:
    return (
        value.strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
    )


def parse_month_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, dayfirst=True, errors="coerce").dt.to_period("M")


def find_last_full_month(source_csv: Path) -> pd.Period:
    month_days: dict[pd.Period, set] = {}

    reader = pd.read_csv(
        source_csv,
        usecols=[DATE_COL],
        chunksize=CHUNK_SIZE,
        encoding="utf-8-sig",
    )

    for chunk in reader:
        dates = pd.to_datetime(chunk[DATE_COL], dayfirst=True, errors="coerce")
        dates = dates.dropna().dt.normalize()
        if dates.empty:
            continue

        month_index = dates.dt.to_period("M")
        for month, month_dates in dates.groupby(month_index):
            month_days.setdefault(month, set()).update(month_dates.dt.date.tolist())

    full_months = [month for month, days in month_days.items() if len(days) == month.days_in_month]
    if not full_months:
        raise ValueError("No complete calendar month was found in the source CSV.")

    return max(full_months)


def export_month_slice(source_csv: Path, target_month: pd.Period, output_csv: Path) -> pd.DataFrame:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        output_csv.unlink()

    usecols = [
        DATE_COL,
        DATETIME_COL,
        EVENT_COL,
        BAKERY_COL,
        CATEGORY_COL,
        PRODUCT_COL,
        QTY_COL,
    ]

    header_written = False
    chunk_frames: list[pd.DataFrame] = []

    reader = pd.read_csv(
        source_csv,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        encoding="utf-8-sig",
    )

    for chunk in reader:
        month_mask = parse_month_series(chunk[DATE_COL]) == target_month
        sales_mask = chunk[EVENT_COL] == SALES_EVENT
        month_sales = chunk.loc[month_mask & sales_mask].copy()

        if month_sales.empty:
            continue

        chunk_frames.append(month_sales)
        month_sales.to_csv(output_csv, index=False, encoding="utf-8-sig", mode="a", header=not header_written)
        header_written = True

    if not chunk_frames:
        raise ValueError(f"No rows found for month {target_month}.")

    month_df = pd.concat(chunk_frames, ignore_index=True)
    month_df[DATE_COL] = pd.to_datetime(month_df[DATE_COL], dayfirst=True, errors="coerce")
    month_df[DATETIME_COL] = pd.to_datetime(month_df[DATETIME_COL], dayfirst=True, errors="coerce")
    month_df[QTY_COL] = pd.to_numeric(month_df[QTY_COL], errors="coerce").fillna(0.0)

    return month_df


def build_product_comparison(
    month_df: pd.DataFrame, bakery: str, product: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bakery_df = month_df[month_df[BAKERY_COL] == bakery].copy()
    if bakery_df.empty:
        raise ValueError(f"Bakery not found in selected month: {bakery}")

    bakery_df["hour"] = bakery_df[DATETIME_COL].dt.hour
    bakery_df["date"] = bakery_df[DATE_COL].dt.normalize()

    product_df = bakery_df[bakery_df[PRODUCT_COL] == product].copy()
    if product_df.empty:
        raise ValueError(f"Product not found for bakery '{bakery}' in selected month: {product}")

    hourly_bakery = (
        bakery_df.groupby(["date", "hour"], as_index=False)[QTY_COL]
        .sum()
        .rename(columns={QTY_COL: "bakery_total_qty"})
    )

    hourly_product = (
        product_df.groupby(["date", "hour"], as_index=False)[QTY_COL]
        .sum()
        .rename(columns={QTY_COL: "actual_product_qty"})
    )

    comparison = hourly_bakery.merge(hourly_product, on=["date", "hour"], how="left")
    comparison["actual_product_qty"] = comparison["actual_product_qty"].fillna(0.0)
    comparison["hour_share"] = comparison["actual_product_qty"] / comparison["bakery_total_qty"].where(
        comparison["bakery_total_qty"] > 0
    )

    hourly_profile = (
        comparison.dropna(subset=["hour_share"])
        .groupby("hour", as_index=False)["hour_share"]
        .mean()
        .rename(columns={"hour_share": "avg_product_share"})
    )

    overall_share = 0.0
    total_bakery_qty = float(hourly_bakery["bakery_total_qty"].sum())
    if total_bakery_qty > 0:
        overall_share = float(hourly_product["actual_product_qty"].sum()) / total_bakery_qty

    full_hours = pd.DataFrame({"hour": list(range(24))})
    hourly_profile = full_hours.merge(hourly_profile, on="hour", how="left")
    hourly_profile["avg_product_share"] = hourly_profile["avg_product_share"].fillna(overall_share)

    comparison = comparison.merge(hourly_profile, on="hour", how="left")
    comparison["avg_product_share"] = comparison["avg_product_share"].fillna(overall_share)
    comparison["expected_product_qty"] = comparison["bakery_total_qty"] * comparison["avg_product_share"]
    comparison["gap_qty"] = comparison["expected_product_qty"] - comparison["actual_product_qty"]
    comparison["abs_gap_qty"] = comparison["gap_qty"].abs()

    daily_comparison = (
        comparison.groupby("date", as_index=False)[
            ["bakery_total_qty", "actual_product_qty", "expected_product_qty", "gap_qty", "abs_gap_qty"]
        ]
        .sum()
    )
    daily_comparison["wmape"] = daily_comparison["abs_gap_qty"] / daily_comparison["actual_product_qty"].where(
        daily_comparison["actual_product_qty"] > 0
    )

    hourly_profile["bakery"] = bakery
    hourly_profile["product"] = product

    return comparison, daily_comparison, hourly_profile


def save_outputs(
    target_month: pd.Period,
    month_df: pd.DataFrame,
    comparison: pd.DataFrame,
    daily_comparison: pd.DataFrame,
    hourly_profile: pd.DataFrame,
    bakery: str,
    product: str,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    month_tag = target_month.strftime("%Y_%m")
    bakery_tag = sanitize_filename(bakery)
    product_tag = sanitize_filename(product)

    month_file = PROCESSED_DIR / f"sales_hrs_last_full_month_{month_tag}.csv"
    comparison_file = REPORTS_DIR / f"{month_tag}_{bakery_tag}_{product_tag}_hourly_comparison.csv"
    daily_file = REPORTS_DIR / f"{month_tag}_{bakery_tag}_{product_tag}_daily_comparison.csv"
    profile_file = REPORTS_DIR / f"{month_tag}_{bakery_tag}_{product_tag}_hourly_profile.csv"
    plot_file = REPORTS_DIR / f"{month_tag}_{bakery_tag}_{product_tag}_comparison.png"

    month_df.to_csv(month_file, index=False, encoding="utf-8-sig")
    comparison.to_csv(comparison_file, index=False, encoding="utf-8-sig")
    daily_comparison.to_csv(daily_file, index=False, encoding="utf-8-sig")
    hourly_profile.to_csv(profile_file, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 1, figsize=(18, 12), gridspec_kw={"height_ratios": [2, 1]})

    daily_plot = daily_comparison.copy()
    daily_plot["date"] = pd.to_datetime(daily_plot["date"])
    axes[0].plot(daily_plot["date"], daily_plot["actual_product_qty"], label="Actual", linewidth=2)
    axes[0].plot(daily_plot["date"], daily_plot["expected_product_qty"], label="Expected", linewidth=2)
    axes[0].set_title(f"{bakery} / {product} - daily actual vs expected")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Qty")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    profile_plot = hourly_profile.sort_values("hour")
    axes[1].bar(profile_plot["hour"], profile_plot["avg_product_share"], color="#d95f02", width=0.8)
    axes[1].set_title("Average hourly share profile")
    axes[1].set_xlabel("Hour of day")
    axes[1].set_ylabel("Avg product share")
    axes[1].set_xticks(list(range(24)))
    axes[1].set_xticklabels([f"{h:02d}:00" for h in range(24)])
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(plot_file, dpi=160, bbox_inches="tight")
    plt.close(fig)

    total_actual = float(comparison["actual_product_qty"].sum())
    total_expected = float(comparison["expected_product_qty"].sum())
    uplift = ((total_expected - total_actual) / total_actual * 100) if total_actual > 0 else 0.0

    print("\nSaved outputs:")
    print(f"  Month slice: {month_file}")
    print(f"  Hourly comparison: {comparison_file}")
    print(f"  Daily comparison: {daily_file}")
    print(f"  Hourly profile: {profile_file}")
    print(f"  Plot: {plot_file}")
    print("\nSummary:")
    print(f"  Total actual:   {total_actual:.2f}")
    print(f"  Total expected: {total_expected:.2f}")
    print(f"  Uplift:         {uplift:+.2f}%")
    print(f"  Avg hourly share: {hourly_profile['avg_product_share'].mean():.4f}")


def main() -> None:
    print("=" * 80)
    print("MONTHLY DEMAND COMPARISON")
    print("=" * 80)

    target_month = find_last_full_month(SOURCE_CSV)
    month_tag = target_month.strftime("%Y_%m")
    month_file = PROCESSED_DIR / f"sales_hrs_last_full_month_{month_tag}.csv"

    print(f"Selected month: {target_month}")
    print(f"Source file: {SOURCE_CSV}")
    print(f"Output file:  {month_file}")

    month_df = export_month_slice(SOURCE_CSV, target_month, month_file)
    print(f"Rows in selected month: {len(month_df):,}")
    print(f"Date range: {month_df[DATE_COL].min().date()} - {month_df[DATE_COL].max().date()}")
    print(f"Bakeries: {month_df[BAKERY_COL].nunique():,}")
    print(f"Products: {month_df[PRODUCT_COL].nunique():,}")

    if not TARGET_BAKERY or not TARGET_PRODUCT:
        print("\nTARGET_BAKERY and TARGET_PRODUCT are empty.")
        print("Fill them in at the top of the script, then rerun for the comparison step.")
        return

    comparison, daily_comparison, hourly_profile = build_product_comparison(
        month_df,
        TARGET_BAKERY,
        TARGET_PRODUCT,
    )

    print("\nSelected pair:")
    print(f"  Bakery:  {TARGET_BAKERY}")
    print(f"  Product: {TARGET_PRODUCT}")
    print(f"  Hourly rows: {len(comparison):,}")
    print(f"  Daily rows:  {len(daily_comparison):,}")

    save_outputs(
        target_month=target_month,
        month_df=month_df,
        comparison=comparison,
        daily_comparison=daily_comparison,
        hourly_profile=hourly_profile,
        bakery=TARGET_BAKERY,
        product=TARGET_PRODUCT,
    )


if __name__ == "__main__":
    main()
