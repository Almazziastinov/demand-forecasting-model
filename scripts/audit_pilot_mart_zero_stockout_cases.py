from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


RECONSTRUCTION_DIR = Path("reports/pilot_mart_zero_demand_reconstruction")
BACKTEST_DIR = Path("reports/pilot_mart_zero_pseudo_stockout_backtest")
OUTPUT_DIR = Path("reports/pilot_mart_zero_case_audit")

HOURLY_RECONSTRUCTED_PATH = RECONSTRUCTION_DIR / "hourly_reconstructed.csv"
DAILY_SIGNALS_PATH = RECONSTRUCTION_DIR / "daily_stockout_signals.csv"
PSEUDO_CASES_PATH = BACKTEST_DIR / "cases.csv"

TOP_OVERALL_CASES = 40
TOP_PER_BAKERY_CASES = 5
PSEUDO_HISTORY_DAYS = 56
PSEUDO_GAP_HOURS = 3


def _round_numeric(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    result = df.copy()
    numeric_cols = result.select_dtypes(include="number").columns
    result[numeric_cols] = result[numeric_cols].round(digits)
    return result


def _format_float(value: float | int | None, digits: int = 1) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def build_real_case_summary(hourly: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    adjusted = hourly[hourly["imputed_demand"] > 0].copy()
    if adjusted.empty:
        return pd.DataFrame()

    group_cols = ["date", "bakery_id", "product_id"]
    daily_cols = [
        "date",
        "bakery_id",
        "bakery_name",
        "product_id",
        "product_name",
        "category_name",
        "daily_sold",
        "hourly_sold",
        "qty_produced",
        "stock_balance",
        "last_sale_hour",
        "normal_days",
        "normal_daily_sold",
        "normal_last_hour",
        "bakery_sales_after_last",
        "last_hour_gap",
        "is_strong_temporal_stockout",
    ]

    summary = (
        adjusted.groupby(group_cols)
        .agg(
            qty_received=("qty_received", "first"),
            qty_sent=("qty_sent", "first"),
            adjusted_hours=("hour", "count"),
            first_adjusted_hour=("hour", "min"),
            last_adjusted_hour=("hour", "max"),
            imputed_units=("imputed_demand", "sum"),
            raw_imputed_units=("raw_imputed_demand", "sum"),
            sold_observed_units=("sold_observed", "sum"),
            demand_units=("sold_demand", "sum"),
            max_hour_imputed=("imputed_demand", "max"),
            mean_policy_scale=("policy_scale", "mean"),
            policy_adjusted_hours=("is_policy_adjusted", "sum"),
        )
        .reset_index()
    )
    summary = summary.merge(daily[daily_cols], on=group_cols, how="left")
    summary["imputed_to_sold_ratio"] = summary["imputed_units"] / summary["daily_sold"].replace(0, pd.NA)
    summary["produced_sellthrough"] = summary["daily_sold"] / summary["qty_produced"].replace(0, pd.NA)
    summary["case_flags"] = ""
    summary.loc[summary["normal_days"] < 3, "case_flags"] += "weak_reference;"
    summary.loc[summary["stock_balance"] < 0, "case_flags"] += "negative_stock;"
    summary.loc[(summary["qty_received"] != 0) | (summary["qty_sent"] != 0), "case_flags"] += "has_transfers;"
    summary.loc[summary["imputed_to_sold_ratio"] > 1, "case_flags"] += "large_relative_imputation;"
    summary["case_flags"] = summary["case_flags"].str.rstrip(";")
    summary["confidence"] = "high"
    summary.loc[summary["normal_days"] < 3, "confidence"] = "review"
    summary.loc[summary["stock_balance"] < 0, "confidence"] = "review"
    summary.loc[summary["imputed_to_sold_ratio"] > 1.5, "confidence"] = "review"

    ordered_cols = [
        "date",
        "bakery_id",
        "bakery_name",
        "product_id",
        "product_name",
        "category_name",
        "daily_sold",
        "qty_produced",
        "qty_received",
        "qty_sent",
        "stock_balance",
        "produced_sellthrough",
        "last_sale_hour",
        "normal_daily_sold",
        "normal_last_hour",
        "last_hour_gap",
        "bakery_sales_after_last",
        "normal_days",
        "adjusted_hours",
        "first_adjusted_hour",
        "last_adjusted_hour",
        "sold_observed_units",
        "imputed_units",
        "raw_imputed_units",
        "demand_units",
        "imputed_to_sold_ratio",
        "max_hour_imputed",
        "mean_policy_scale",
        "policy_adjusted_hours",
        "confidence",
        "case_flags",
    ]
    return summary[ordered_cols].sort_values(["imputed_units", "adjusted_hours"], ascending=[False, False])


def select_audit_cases(case_summary: pd.DataFrame) -> pd.DataFrame:
    top_overall = case_summary.head(TOP_OVERALL_CASES).copy()
    top_per_bakery = (
        case_summary.sort_values(["bakery_id", "imputed_units"], ascending=[True, False])
        .groupby("bakery_id", as_index=False)
        .head(TOP_PER_BAKERY_CASES)
    )

    selected = pd.concat([top_overall, top_per_bakery], ignore_index=True)
    selected = selected.drop_duplicates(["date", "bakery_id", "product_id"])
    selected = selected.sort_values(["imputed_units", "adjusted_hours"], ascending=[False, False]).reset_index(drop=True)
    selected.insert(0, "case_rank", range(1, len(selected) + 1))
    return selected


def build_hourly_curves(hourly: pd.DataFrame, selected_cases: pd.DataFrame) -> pd.DataFrame:
    keys = selected_cases[["case_rank", "date", "bakery_id", "product_id"]]
    curves = hourly.merge(keys, on=["date", "bakery_id", "product_id"], how="inner")
    columns = [
        "case_rank",
        "date",
        "bakery_id",
        "bakery_name",
        "product_id",
        "product_name",
        "category_name",
        "hour",
        "sold",
        "sold_demand",
        "imputed_demand",
        "raw_imputed_demand",
        "expected_demand",
        "bakery_hour_sales",
        "mean_sku_share",
        "last_sale_hour",
        "is_censored_hour",
        "policy_scale",
        "is_policy_adjusted",
    ]
    return curves[columns].sort_values(["case_rank", "hour"])


def build_pseudo_case_samples() -> pd.DataFrame:
    if not PSEUDO_CASES_PATH.exists():
        return pd.DataFrame()

    cases = pd.read_csv(PSEUDO_CASES_PATH)
    cases = cases[
        (cases["history_days"] == PSEUDO_HISTORY_DAYS)
        & (cases["gap_hours"] == PSEUDO_GAP_HOURS)
        & (cases["true_hidden"] > 0)
    ].copy()
    if cases.empty:
        return cases

    cases["recovery_ratio_policy"] = cases["predicted_hidden_policy"] / cases["true_hidden"]
    cases["abs_error_pct_policy"] = cases["abs_error_policy"] / cases["true_hidden"]

    worst_under = cases.sort_values("error_policy").head(20).assign(sample_type="worst_under")
    worst_over = cases.sort_values("error_policy", ascending=False).head(20).assign(sample_type="worst_over")
    best_fit = cases.assign(abs_error_sort=cases["error_policy"].abs()).sort_values("abs_error_sort").head(20)
    best_fit = best_fit.drop(columns=["abs_error_sort"]).assign(sample_type="best_fit")

    samples = pd.concat([worst_under, worst_over, best_fit], ignore_index=True)
    samples = samples.drop_duplicates(["date", "bakery_id", "product_id", "sample_type"])
    columns = [
        "sample_type",
        "date",
        "bakery_id",
        "bakery_name",
        "product_id",
        "product_name",
        "category_name",
        "volume_band",
        "daily_sold",
        "hidden_hours",
        "true_hidden",
        "predicted_hidden_policy",
        "recovery_ratio_policy",
        "error_policy",
        "abs_error_pct_policy",
    ]
    return samples[columns].sort_values(["sample_type", "abs_error_pct_policy"], ascending=[True, False])


def write_markdown_report(
    case_summary: pd.DataFrame,
    selected_cases: pd.DataFrame,
    pseudo_samples: pd.DataFrame,
    output_path: Path,
) -> None:
    top = selected_cases.head(25)
    lines = [
        "# Pilot Mart Zero Stockout Case Audit",
        "",
        "Offline audit of real stockout reconstruction cases for pilot bakeries.",
        "",
        "## Scope",
        "",
        f"- Real adjusted daily cases: {len(case_summary):,}",
        f"- Selected audit cases: {len(selected_cases):,}",
        f"- Selection: top {TOP_OVERALL_CASES} overall plus top {TOP_PER_BAKERY_CASES} per bakery by imputed units",
        "- Production state was not changed; this report uses local CSV exports only.",
        "",
        "## Top Real Cases",
        "",
        "| rank | date | bakery | product | sold | produced | stock | last sale | ref days | normal sold | adjusted h | imputed | confidence | flags |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]

    for _, row in top.iterrows():
        note = ""
        if row["mean_policy_scale"] < 0.999:
            note = f"policy scale {_format_float(row['mean_policy_scale'], 2)}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(row["case_rank"])),
                    str(row["date"]),
                    str(int(row["bakery_id"])),
                    str(row["product_name"]),
                    _format_float(row["daily_sold"]),
                    _format_float(row["qty_produced"]),
                    _format_float(row["stock_balance"]),
                    _format_float(row["last_sale_hour"], 0),
                    _format_float(row["normal_days"], 0),
                    _format_float(row["normal_daily_sold"]),
                    str(int(row["adjusted_hours"])),
                    _format_float(row["imputed_units"]),
                    str(row["confidence"]),
                    str(row["case_flags"] or note),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## How To Read",
            "",
            "- `sold` is observed daily sales from mart_zero.",
            "- `produced` and `stock` are daily production and final stock balance.",
            "- `last sale` is the last hour with observed positive sales for the SKU.",
            "- `normal sold` is the non-stockout benchmark for the same bakery, SKU, and weekday.",
            "- `ref days` below 3 means the case is useful for inspection but should not be trusted as a stable coefficient yet.",
            "- `imputed` is added demand after policy caps.",
            "- Full hourly curves are in `case_hourly_curves.csv`.",
            "",
            "## Pseudo-Stockout Samples",
            "",
            f"Samples use history_days={PSEUDO_HISTORY_DAYS}, gap_hours={PSEUDO_GAP_HOURS}.",
            "",
        ]
    )

    if pseudo_samples.empty:
        lines.append("No pseudo-stockout samples were available.")
    else:
        lines.extend(
            [
                "| type | date | bakery | product | daily sold | hidden | predicted | recovery | error |",
                "|---|---|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in pseudo_samples.groupby("sample_type", sort=False).head(8).iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["sample_type"]),
                        str(row["date"]),
                        str(int(row["bakery_id"])),
                        str(row["product_name"]),
                        _format_float(row["daily_sold"]),
                        _format_float(row["true_hidden"]),
                        _format_float(row["predicted_hidden_policy"]),
                        _format_float(row["recovery_ratio_policy"], 2),
                        _format_float(row["error_policy"]),
                    ]
                )
                + " |"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hourly = pd.read_csv(HOURLY_RECONSTRUCTED_PATH)
    daily = pd.read_csv(DAILY_SIGNALS_PATH)

    case_summary = build_real_case_summary(hourly, daily)
    selected_cases = select_audit_cases(case_summary)
    hourly_curves = build_hourly_curves(hourly, selected_cases)
    pseudo_samples = build_pseudo_case_samples()

    case_summary_out = _round_numeric(case_summary)
    selected_cases_out = _round_numeric(selected_cases)
    hourly_curves_out = _round_numeric(hourly_curves)
    pseudo_samples_out = _round_numeric(pseudo_samples)

    case_summary_out.to_csv(OUTPUT_DIR / "real_case_summary.csv", index=False, encoding="utf-8-sig")
    selected_cases_out.to_csv(OUTPUT_DIR / "selected_real_cases.csv", index=False, encoding="utf-8-sig")
    hourly_curves_out.to_csv(OUTPUT_DIR / "case_hourly_curves.csv", index=False, encoding="utf-8-sig")
    pseudo_samples_out.to_csv(OUTPUT_DIR / "pseudo_case_samples.csv", index=False, encoding="utf-8-sig")

    write_markdown_report(
        case_summary=case_summary_out,
        selected_cases=selected_cases_out,
        pseudo_samples=pseudo_samples_out,
        output_path=OUTPUT_DIR / "case_audit.md",
    )

    summary = {
        "real_adjusted_cases": int(len(case_summary)),
        "selected_real_cases": int(len(selected_cases)),
        "selected_hourly_rows": int(len(hourly_curves)),
        "pseudo_sample_cases": int(len(pseudo_samples)),
        "top_case": selected_cases_out.head(1).to_dict(orient="records"),
        "outputs": {
            "real_case_summary": str(OUTPUT_DIR / "real_case_summary.csv"),
            "selected_real_cases": str(OUTPUT_DIR / "selected_real_cases.csv"),
            "case_hourly_curves": str(OUTPUT_DIR / "case_hourly_curves.csv"),
            "pseudo_case_samples": str(OUTPUT_DIR / "pseudo_case_samples.csv"),
            "case_audit": str(OUTPUT_DIR / "case_audit.md"),
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
