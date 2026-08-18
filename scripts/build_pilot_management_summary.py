"""Build a read-only canonical management summary for the base 10 pilot.

The ClickHouse adapter only extracts data.  All selection, validation and KPI
calculation functions operate on DataFrames and can be tested without I/O.
No execution KPI is published because these sources do not prove which plan
was actually delivered to a bakery.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.export_clickhouse_checks import create_client  # noqa: E402
from src.pilot_metrics import aggregate_forecast_metrics  # noqa: E402
from src.pilot_plan_archive import normalize_header, parse_pilot_plan  # noqa: E402
from src.pilot_performance import (  # noqa: E402
    PerformanceContract,
    PerformanceInput,
    build_performance_rows,
)
from src.pilot_product_scope import (  # noqa: E402
    COLD_START_PRODUCT_IDS,
    classify_unmatched_reason,
    normalize_product_name,
    resolve_product_name,
)

PILOT_IDS = (
    1, 3, 12, 13, 14, 20, 21, 22, 26, 27, 28, 39, 41, 56, 57, 66, 67, 69,
    80, 89, 99, 107, 113, 125, 149, 153, 155, 160, 191, 221, 222, 229, 230,
    246, 257, 260, 268, 270,
)
SCOPE_CATEGORIES = (
    "Пироги сытные",
    "Пироги сладкие",
    "Выпечка сытная",
    "Выпечка сладкая",
    "Фастфуд",
)
SCOPE_VERSION = "pilot_38_v1"
METRIC_VERSION = "pilot_management_v1"
FORECAST_RELIABILITY_VERSION = "forecast_reliability_v1"
OUTPUT_DIR = ROOT / "reports" / "pilot_management_summary"
DEFAULT_PLAN_MANIFEST = ROOT / ".codex_tmp" / "pilot_plan_archive" / "manifest.json"


def load_effective_plan_archive(
    manifest_path: Path, facts: pd.DataFrame, *, date_to: date
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load effective Bitrix attachments and map workbook names to fact IDs.

    Join key is ``product_id`` resolved from the fact table.  Unresolved names
    are classified by ``classify_unmatched_reason`` and written to the exclusions
    frame with a machine-readable ``reason`` column so callers can distinguish
    confirmed renames, frozen products, cold-start SKUs and genuine unknowns.
    """
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = [
        item
        for item in payload
        if item.get("is_effective")
        and date.fromisoformat(item["target_date"]) <= date_to
    ]
    lookup = facts.copy()
    lookup["bakery_key"] = lookup["bakery_name"].map(normalize_header)
    bakery_map = lookup.drop_duplicates("bakery_key").set_index("bakery_key")[
        "bakery_id"
    ]
    # Build normalized-name → product_id from the fact universe.  product_id is
    # the canonical join key; the name is used only for the initial lookup.
    name_to_id: dict[str, int] = {}
    if "product_name" in lookup.columns and "product_id" in lookup.columns:
        for _, row in lookup.drop_duplicates("product_name").iterrows():
            if pd.notna(row.get("product_name")) and pd.notna(row.get("product_id")):
                key = normalize_product_name(row["product_name"])
                if key:
                    name_to_id[key] = int(row["product_id"])

    rows, exclusions = [], []
    for item in items:
        plan = parse_pilot_plan(item["path"])
        for source in plan.rows:
            bakery_id = bakery_map.get(normalize_header(source.bakery))
            product_id, _match_type, _alias = resolve_product_name(
                source.product_name, name_to_id
            )
            if pd.isna(bakery_id) or product_id is None:
                if product_id is None:
                    exc_reason = classify_unmatched_reason(source.product_name)
                    reason = f"plan_{exc_reason.value}"
                else:
                    reason = "bakery_outside_scope"
                exclusions.append(
                    {
                        "date": plan.target_date,
                        "bakery_id": int(bakery_id) if pd.notna(bakery_id) else None,
                        "product_name": source.product_name,
                        "reason": reason,
                    }
                )
                continue
            rows.append(
                {
                    "date": plan.target_date,
                    "bakery_id": int(bakery_id),
                    "product_id": int(product_id),
                    "source_run_id": f"bitrix:{item['message_id']}:{plan.sha256[:12]}",
                    "forecast_qty": source.issued_forecast,
                    "issued_plan_qty": source.plan_qty,
                    "plan_format_version": plan.format_version,
                    "source_quality": item["source_quality"],
                    "published_at": item.get("published_at"),
                    "issued_yesterday_stock": source.yesterday_stock,
                    "issued_total_for_sale": source.total_for_sale,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(exclusions)


def select_coherent_lead1(
    forecast: pd.DataFrame,
    *,
    scope_bakery_ids: tuple[int, ...] = PILOT_IDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one pre-08:00 run covering the entire scope for each date.

    VM forecast-production.timer fires at 03:30 UTC (06:30 MSK) and publishes
    to ClickHouse by ~06:40 MSK. Cutoff is 08:00 MSK so nightly runs land
    inside the window while ad-hoc afternoon reruns are excluded.
    """

    required = {
        "date",
        "bakery_id",
        "product_id",
        "source_run_id",
        "generated_at",
        "forecast_qty",
    }
    missing = required.difference(forecast.columns)
    if missing:
        raise ValueError(f"missing forecast columns: {sorted(missing)}")
    work = forecast.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.date
    work["generated_at"] = pd.to_datetime(work["generated_at"], errors="coerce")
    exclusions: list[dict[str, object]] = []
    selected: list[pd.DataFrame] = []
    for business_date, group in work.groupby("date"):
        cutoff = pd.Timestamp(business_date).tz_localize(
            "Europe/Moscow"
        ) + pd.Timedelta(hours=8)
        generated = group["generated_at"]
        generated = (
            generated.dt.tz_localize("Europe/Moscow")
            if generated.dt.tz is None
            else generated.dt.tz_convert("Europe/Moscow")
        )
        group = group[generated.lt(cutoff)].copy()
        if group.empty or group["source_run_id"].isna().any():
            exclusions.append(
                {
                    "date": business_date,
                    "bakery_id": None,
                    "reason": "no_pre_cutoff_run",
                }
            )
            continue
        run_coverage = group.groupby("source_run_id")["bakery_id"].nunique()
        full_runs = run_coverage[run_coverage.eq(len(scope_bakery_ids))].index
        candidates = group[group["source_run_id"].isin(full_runs)]
        if candidates.empty:
            exclusions.append(
                {
                    "date": business_date,
                    "bakery_id": None,
                    "reason": "no_full_scope_run",
                }
            )
            continue
        run_times = candidates.groupby("source_run_id")["generated_at"].max()
        latest_time = run_times.max()
        latest_runs = run_times[run_times.eq(latest_time)].index.tolist()
        if len(latest_runs) != 1:
            exclusions.append(
                {
                    "date": business_date,
                    "bakery_id": None,
                    "reason": "ambiguous_full_run",
                }
            )
            continue
        chosen = candidates[candidates["source_run_id"].eq(latest_runs[0])].copy()
        if chosen.duplicated(["date", "bakery_id", "product_id"]).any():
            exclusions.append(
                {
                    "date": business_date,
                    "bakery_id": None,
                    "reason": "duplicate_sku",
                }
            )
            continue
        selected.append(chosen)
    return (
        pd.concat(selected, ignore_index=True) if selected else work.iloc[0:0].copy(),
        pd.DataFrame(exclusions, columns=["date", "bakery_id", "reason"]),
    )


def build_detail(
    forecast: pd.DataFrame,
    facts: pd.DataFrame,
    *,
    scope_bakery_ids: tuple[int, ...] = PILOT_IDS,
    plans_are_confirmed: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    facts = facts.copy()
    facts["date"] = pd.to_datetime(facts["date"]).dt.date
    if plans_are_confirmed:
        selected = forecast.copy()
        selected["date"] = pd.to_datetime(selected["date"]).dt.date
        exclusions = pd.DataFrame(columns=["date", "bakery_id", "reason"])
    else:
        selected, exclusions = select_coherent_lead1(
            forecast, scope_bakery_ids=scope_bakery_ids
        )
    forecast_dates = set(pd.to_datetime(forecast["date"]).dt.date)
    absent_dates = sorted(set(facts["date"]).difference(forecast_dates))
    if absent_dates:
        exclusions = pd.concat(
            [
                exclusions,
                pd.DataFrame(
                    {
                        "date": absent_dates,
                        "bakery_id": [None] * len(absent_dates),
                        "reason": ["no_forecast_snapshot"] * len(absent_dates),
                    }
                ),
            ],
            ignore_index=True,
        )
    if selected.empty:
        return pd.DataFrame(), exclusions
    facts = facts[facts["date"].isin(set(selected["date"]))].copy()
    keys = ["date", "bakery_id", "product_id"]
    extras = selected.merge(facts[keys], on=keys, how="left", indicator=True)
    extras = extras[extras["_merge"].eq("left_only")]
    if not extras.empty:
        extra_rows = extras[keys].copy()
        extra_rows["reason"] = "forecast_outside_fact_universe"
        exclusions = pd.concat([exclusions, extra_rows], ignore_index=True)
    plan_columns = [
        column
        for column in (
            "issued_plan_qty",
            "plan_format_version",
            "source_quality",
            "published_at",
            "issued_yesterday_stock",
            "issued_total_for_sale",
        )
        if column in selected.columns
    ]
    merged = facts.merge(
        selected[keys + ["source_run_id", "forecast_qty"] + plan_columns],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    run_by_date = selected.groupby("date")["source_run_id"].first()
    merged["source_run_id"] = merged["source_run_id"].fillna(
        merged["date"].map(run_by_date)
    )
    rows = []
    for run_id, run_group in merged.groupby("source_run_id"):
        inputs = []
        for row in run_group.itertuples(index=False):
            available = float(row.qty_produced)
            inputs.append(
                PerformanceInput(
                    business_date=row.date,
                    bakery_id=int(row.bakery_id),
                    product_id=int(row.product_id),
                    forecast_run_id=str(run_id),
                    forecast_qty=(
                        float(row.forecast_qty) if pd.notna(row.forecast_qty) else None
                    ),
                    plan_qty=(
                        float(row.issued_plan_qty)
                        if "issued_plan_qty" in plan_columns
                        and pd.notna(row.issued_plan_qty)
                        and row.plan_format_version == "stock_aware_v2"
                        else None
                    ),
                    produced_qty=float(row.qty_produced),
                    sold_qty=float(row.qty_sold),
                    closing_stock_qty=float(row.stock_balance),
                    last_sale_at=pd.to_datetime(
                        row.last_sale_time, errors="coerce"
                    ).to_pydatetime()
                    if pd.notna(row.last_sale_time)
                    else None,
                    opening_stock_qty=0.0,
                    received_qty=0.0,
                    sent_qty=0.0,
                    transfers_complete=True,
                    available_to_sell_qty=available,
                    available_to_sell_basis="fct_tables_v1",
                )
            )
        contract = PerformanceContract(
            metric_version=METRIC_VERSION,
            scope_version=SCOPE_VERSION,
            forecast_run_id=str(run_id),
        )
        rows.extend(
            build_performance_rows(
                inputs, contract=contract, scope_bakery_ids=scope_bakery_ids
            )
        )
    detail = pd.DataFrame(asdict(row) for row in rows)
    name_columns = [
        column
        for column in ("bakery_name", "product_name", "fact_category_name")
        if column in merged.columns
    ]
    extra_columns = name_columns + plan_columns
    if extra_columns:
        detail = detail.merge(
            merged[keys + extra_columns],
            left_on=["business_date", "bakery_id", "product_id"],
            right_on=keys,
            how="left",
        ).drop(columns=["date"])
    valid_forecast = detail["forecast_qty"].notna() & detail["forecast_qty"].ge(0)
    detail["eligible_forecast_summary"] = detail["eligible_forecast"] | (
        valid_forecast
        & detail["lost_demand_rejection_reason"].eq("closed_or_after_close")
        & detail["lost_demand_flags"].map(
            lambda flags: all("negative_stock" not in str(flag) for flag in flags)
        )
    )
    closed = detail["lost_demand_rejection_reason"].eq("closed_or_after_close")
    detail.loc[closed, "demand_qty"] = detail.loc[closed, "sold_qty"]
    for column in ("forecast_status", "execution_status", "availability_status"):
        detail[column] = detail[column].astype(str)
    return detail, exclusions.drop_duplicates().reset_index(drop=True)


def _kpi(group: pd.DataFrame) -> dict[str, object]:
    eligibility_column = (
        "eligible_forecast_summary"
        if "eligible_forecast_summary" in group.columns
        else "eligible_forecast"
    )
    scored = group[group[eligibility_column]]
    metrics = aggregate_forecast_metrics(
        zip(scored["forecast_qty"], scored["demand_qty"], strict=False)
    )
    result = asdict(metrics)
    present = group[group["forecast_qty"].notna() & group["forecast_qty"].ge(0)]
    observed_metrics = aggregate_forecast_metrics(
        zip(present["forecast_qty"], present["sold_qty"], strict=False)
    )
    formats = group.get(
        "plan_format_version", pd.Series(index=group.index, dtype=object)
    )
    execution = group[formats.eq("stock_aware_v2") & group["eligible_execution"]]
    execution_error = execution["produced_qty"] - execution["plan_qty"]
    execution_plan = execution["plan_qty"].sum()
    errors = scored["forecast_qty"] - scored["demand_qty"]
    result.update(
        {
            "rows_total": int(len(group)),
            "rows_forecast_eligible": int(len(scored)),
            "forecast_coverage": float(len(scored) / len(group)) if len(group) else 0.0,
            "expected_fact_volume": float(group["sold_qty"].sum()),
            "scored_fact_volume": float(scored["sold_qty"].sum()),
            "forecast_volume_coverage": (
                float(scored["sold_qty"].sum() / group["sold_qty"].sum())
                if group["sold_qty"].sum() > 0
                else None
            ),
            "missing_forecast_rows": int(group["forecast_qty"].isna().sum()),
            "forecast_presence_row_coverage": float(len(present) / len(group)),
            "forecast_presence_volume_coverage": (
                float(present["sold_qty"].sum() / group["sold_qty"].sum())
                if group["sold_qty"].sum() > 0
                else None
            ),
            "observed_sales_wape": observed_metrics.wape,
            "observed_sales_bias": observed_metrics.bias,
            "complete_demand_wape": metrics.wape,
            "complete_demand_bias": metrics.bias,
            "underforecast_qty": float((-errors).clip(lower=0).sum()),
            "overforecast_qty": float(errors.clip(lower=0).sum()),
            "missing_forecast_volume": float(
                group.loc[group["forecast_qty"].isna(), "sold_qty"].sum()
            ),
            "recognized_lost_qty": float(group["lost_demand_recognized_qty"].sum()),
            "execution_kpi_included": bool(len(execution)),
            "execution_rows": int(len(execution)),
            "execution_wape": (
                float(execution_error.abs().sum() / execution_plan)
                if execution_plan > 0
                else None
            ),
            "execution_bias": (
                float(execution_error.sum() / execution_plan)
                if execution_plan > 0
                else None
            ),
            "execution_coverage": float(len(execution) / len(group))
            if len(group)
            else 0.0,
        }
    )
    return result


def build_kpis(
    detail: pd.DataFrame, scope_version: str = SCOPE_VERSION
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if detail.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    company = pd.DataFrame([{"scope_version": scope_version, **_kpi(detail)}])
    bakery_rows = []
    for bakery_id, group in detail.groupby("bakery_id"):
        bakery_rows.append({"bakery_id": int(bakery_id), **_kpi(group)})
    work = detail.copy()
    dates = pd.to_datetime(work["business_date"])
    work["week_start"] = (dates - pd.to_timedelta(dates.dt.dayofweek, unit="D")).dt.date
    week_rows = []
    for week_start, group in work.groupby("week_start"):
        week_rows.append(
            {
                "week_start": week_start,
                "partial_period": group["business_date"].nunique() < 7,
                **_kpi(group),
            }
        )
    return company, pd.DataFrame(bakery_rows), pd.DataFrame(week_rows)


def build_rankings(
    detail: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build transparent bakery and SKU priorities from material error."""

    eligible = detail[detail["eligible_forecast_summary"]].copy()
    eligible["error_qty"] = eligible["forecast_qty"] - eligible["demand_qty"]
    eligible["abs_error_qty"] = eligible["error_qty"].abs()
    total_abs_error = eligible["abs_error_qty"].sum()

    bakery_rows = []
    for bakery_id, group in detail.groupby("bakery_id"):
        row = {"bakery_id": int(bakery_id), **_kpi(group)}
        bakery_abs = eligible.loc[
            eligible["bakery_id"].eq(bakery_id), "abs_error_qty"
        ].sum()
        row["contribution_to_total_abs_error"] = (
            float(bakery_abs / total_abs_error) if total_abs_error else None
        )
        row["model_priority_status"] = (
            "model_action"
            if row["forecast_volume_coverage"] >= 0.60
            and row["complete_demand_bias"] is not None
            and abs(row["complete_demand_bias"]) >= 0.15
            else "model_watch"
        )
        row["execution_priority_status"] = (
            "execution_action"
            if row["execution_coverage"] >= 0.60
            and row["execution_bias"] is not None
            and abs(row["execution_bias"]) >= 0.15
            else "execution_watch"
        )
        bakery_rows.append(row)
    bakery_ranking = pd.DataFrame(bakery_rows).sort_values(
        ["contribution_to_total_abs_error", "complete_demand_wape"], ascending=False
    )

    def priorities(group_columns: list[str]) -> pd.DataFrame:
        rows = []
        for keys, group in eligible.groupby(group_columns, dropna=False):
            key_values = keys if isinstance(keys, tuple) else (keys,)
            error = group["error_qty"]
            abs_error = group["abs_error_qty"].sum()
            under = (-error).clip(lower=0).sum()
            over = error.clip(lower=0).sum()
            dominant = abs(error.sum()) / abs_error if abs_error else 0.0
            repeated_days = int(
                group.assign(sign=error.ge(0))
                .groupby("sign")["business_date"]
                .nunique()
                .max()
            )
            contribution = abs_error / total_abs_error if total_abs_error else 0.0
            flags = []
            if contribution >= 0.01:
                flags.append("material_abs_error")
            if dominant >= 0.70 and repeated_days >= 3:
                flags.append("systematic_direction")
            if under > over and dominant >= 0.70:
                flags.append("systematic_underforecast")
            if over > under and dominant >= 0.70:
                flags.append("systematic_overforecast")
            tier = (
                "high"
                if len(flags) >= 2 and contribution >= 0.01
                else ("medium" if flags else "low")
            )
            row = dict(zip(group_columns, key_values, strict=True))
            row.update(
                rows=int(len(group)),
                demand_qty=float(group["demand_qty"].sum()),
                error_qty=float(error.sum()),
                abs_error_qty=float(abs_error),
                underforecast_qty=float(under),
                overforecast_qty=float(over),
                directionality=float(dominant),
                repeated_direction_days=repeated_days,
                contribution_to_total_abs_error=float(contribution),
                recognized_lost_qty=float(group["lost_demand_recognized_qty"].sum()),
                priority_tier=tier,
                priority_flags="|".join(flags),
            )
            rows.append(row)
        return pd.DataFrame(rows).sort_values(
            ["priority_tier", "abs_error_qty"],
            ascending=[True, False],
            key=lambda series: (
                series.map({"high": 0, "medium": 1, "low": 2})
                if series.name == "priority_tier"
                else series
            ),
        )

    sku_columns = ["product_id"] + (
        ["product_name"] if "product_name" in detail else []
    )
    sku_priority = priorities(sku_columns)
    bakery_sku_priority = priorities(["bakery_id", *sku_columns])
    daily_rows = []
    for business_date, group in detail.groupby("business_date"):
        daily_rows.append(
            {"business_date": business_date, "partial_period": False, **_kpi(group)}
        )
    return bakery_ranking, sku_priority, bakery_sku_priority, pd.DataFrame(daily_rows)


def build_priority_queues(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build independently gated model and execution queues."""

    model_rows = []
    total_model_abs = 0.0
    model_intermediate = []
    for product_id, group in detail.groupby("product_id"):
        eligible = group[group["eligible_forecast_summary"]]
        error = eligible["forecast_qty"] - eligible["demand_qty"]
        abs_error = float(error.abs().sum())
        total_model_abs += abs_error
        demand = float(eligible["demand_qty"].sum())
        all_volume = float(group["sold_qty"].clip(lower=0).sum())
        coverage = (
            float(eligible["sold_qty"].clip(lower=0).sum() / all_volume)
            if all_volume
            else 0.0
        )
        bias = float(error.sum() / demand) if demand else None
        directionality = float(abs(error.sum()) / abs_error) if abs_error else 0.0
        model_intermediate.append(
            {
                "product_id": int(product_id),
                "product_name": group["product_name"].dropna().iloc[0]
                if "product_name" in group and group["product_name"].notna().any()
                else None,
                "cohort": "cold_start"
                if int(product_id) in COLD_START_PRODUCT_IDS
                else "mature",
                "eligible_days": int(eligible["business_date"].nunique()),
                "demand_qty": demand,
                "volume_coverage": coverage,
                "bias": bias,
                "directionality": directionality,
                "error_qty": float(error.sum()),
                "abs_error_qty": abs_error,
                "gates_passed": bool(
                    int(product_id) not in COLD_START_PRODUCT_IDS
                    and eligible["business_date"].nunique() >= 7
                    and demand >= 100
                    and coverage >= 0.60
                    and bias is not None
                    and abs(bias) >= 0.15
                    and directionality >= 0.40
                ),
            }
        )
    ordered = sorted(
        model_intermediate, key=lambda row: row["abs_error_qty"], reverse=True
    )
    cumulative = 0.0
    for row in ordered:
        contribution = (
            row["abs_error_qty"] / total_model_abs if total_model_abs else 0.0
        )
        cumulative += contribution
        row["contribution_to_total_abs_error"] = contribution
        row["cumulative_contribution"] = cumulative
        if not row["gates_passed"]:
            row["priority_tier"] = (
                "M3"
                if row["cohort"] == "cold_start" or row["eligible_days"] < 7
                else "M0"
            )
        elif contribution >= 0.01 or cumulative <= 0.50:
            row["priority_tier"] = "M1"
        elif contribution >= 0.0025 or cumulative <= 0.80:
            row["priority_tier"] = "M2"
        else:
            row["priority_tier"] = "M3"
        model_rows.append(row)

    execution_rows, total_execution_abs = [], 0.0
    intermediate = []
    stock_aware = detail[
        detail.get(
            "plan_format_version", pd.Series(index=detail.index, dtype=object)
        ).eq("stock_aware_v2")
    ]
    for product_id, group in stock_aware.groupby("product_id"):
        eligible = group[group["eligible_execution"]]
        error = eligible["produced_qty"] - eligible["plan_qty"]
        abs_error = float(error.abs().sum())
        total_execution_abs += abs_error
        plan = float(eligible["plan_qty"].sum())
        bias = float(error.sum() / plan) if plan else None
        directionality = float(abs(error.sum()) / abs_error) if abs_error else 0.0
        tolerance = eligible["plan_qty"] * 0.20
        normal_mask = error.abs().le(tolerance)
        under_mask = error.lt(-tolerance)
        over_mask = error.gt(tolerance)
        forecast_intersection = eligible[eligible["eligible_forecast_summary"]].copy()
        forecast_error = (
            forecast_intersection["forecast_qty"] - forecast_intersection["demand_qty"]
        )
        forecast_demand = float(forecast_intersection["demand_qty"].sum())
        forecast_wape = (
            float(forecast_error.abs().sum() / forecast_demand)
            if forecast_demand > 0
            else None
        )
        forecast_bias = (
            float(forecast_error.sum() / forecast_demand)
            if forecast_demand > 0
            else None
        )
        forecast_tolerance = forecast_intersection["demand_qty"] * 0.20
        forecast_normal = forecast_error.abs().le(forecast_tolerance)
        forecast_coverage = (
            float(
                forecast_intersection["sold_qty"].clip(lower=0).sum()
                / eligible["sold_qty"].clip(lower=0).sum()
            )
            if eligible["sold_qty"].clip(lower=0).sum() > 0
            else 0.0
        )
        forecast_days = int(forecast_intersection["business_date"].nunique())
        sufficient = (
            forecast_days >= 7 and forecast_demand >= 100 and forecast_coverage >= 0.60
        )
        reliable = bool(
            sufficient
            and forecast_wape is not None
            and forecast_wape <= 0.20
            and forecast_bias is not None
            and abs(forecast_bias) <= 0.10
        )
        reliability = (
            "reliable"
            if reliable
            else ("unreliable" if sufficient else "insufficient_data")
        )
        closer = (
            forecast_intersection["produced_qty"] - forecast_intersection["demand_qty"]
        ).abs() < (
            forecast_intersection["plan_qty"] - forecast_intersection["demand_qty"]
        ).abs()
        execution_fails = bool(abs(bias) >= 0.15) if bias is not None else False
        if reliability == "insufficient_data":
            responsibility = "insufficient_data"
        elif reliable and execution_fails:
            responsibility = "bakery_execution"
        elif not reliable and not execution_fails:
            responsibility = "model_issue"
        elif not reliable and execution_fails:
            responsibility = "mixed_issue"
        else:
            responsibility = "normal"
        intermediate.append(
            {
                "product_id": int(product_id),
                "product_name": group["product_name"].dropna().iloc[0]
                if "product_name" in group and group["product_name"].notna().any()
                else None,
                "plan_days": int(eligible["business_date"].nunique()),
                "plan_qty": plan,
                "bias": bias,
                "execution_bias": bias,
                "plan_fulfillment_ratio": (
                    float(eligible["produced_qty"].sum() / plan) if plan > 0 else None
                ),
                "execution_wape": (
                    float(error.abs().sum() / plan) if plan > 0 else None
                ),
                "execution_mae_qty": float(error.abs().mean()) if len(error) else None,
                "normal_cases": int(normal_mask.sum()),
                "under_cases": int(under_mask.sum()),
                "over_cases": int(over_mask.sum()),
                "normal_case_share": float(normal_mask.mean()) if len(error) else None,
                "under_case_share": float(under_mask.mean()) if len(error) else None,
                "over_case_share": float(over_mask.mean()) if len(error) else None,
                "directionality": directionality,
                "error_qty": float(error.sum()),
                "abs_error_qty": abs_error,
                "forecast_complete_demand_wape": forecast_wape,
                "forecast_complete_demand_bias": forecast_bias,
                "forecast_mae_qty": (
                    float(forecast_error.abs().mean()) if len(forecast_error) else None
                ),
                "forecast_normal_case_share": (
                    float(forecast_normal.mean()) if len(forecast_normal) else None
                ),
                "forecast_demand_coverage": forecast_coverage,
                "forecast_days": forecast_days,
                "forecast_demand_qty": forecast_demand,
                "forecast_reliability_version": FORECAST_RELIABILITY_VERSION,
                "forecast_reliability": reliability,
                "deprecated_responsibility_diagnostic": responsibility,
                "closer_to_demand_than_plan_count": int(closer.sum()),
                "closer_to_demand_than_plan_share": (
                    float(closer.mean()) if len(closer) else None
                ),
                "gates_passed": bool(
                    eligible["business_date"].nunique() >= 7
                    and plan >= 50
                    and bias is not None
                    and abs(bias) >= 0.15
                    and directionality >= 0.40
                ),
            }
        )
    cumulative = 0.0
    for row in sorted(
        intermediate, key=lambda value: value["abs_error_qty"], reverse=True
    ):
        contribution = (
            row["abs_error_qty"] / total_execution_abs if total_execution_abs else 0.0
        )
        cumulative += contribution
        row["contribution_to_total_abs_error"] = contribution
        row["cumulative_contribution"] = cumulative
        if not row["gates_passed"]:
            row["priority_tier"] = "E3" if row["plan_days"] < 7 else "E0"
        elif contribution >= 0.01 or cumulative <= 0.50:
            row["priority_tier"] = "E1"
        elif contribution >= 0.0025 or cumulative <= 0.80:
            row["priority_tier"] = "E2"
        else:
            row["priority_tier"] = "E3"
        execution_rows.append(row)
    return pd.DataFrame(model_rows), pd.DataFrame(execution_rows)


def build_execution_triage(detail: pd.DataFrame) -> pd.DataFrame:
    """Conservative evidence triage at bakery/SKU grain on comparable stock basis."""

    rows = []
    stock = detail[
        detail.get(
            "plan_format_version", pd.Series(index=detail.index, dtype=object)
        ).eq("stock_aware_v2")
    ]
    for (bakery_id, product_id), group in stock.groupby(["bakery_id", "product_id"]):
        execution = group[group["eligible_execution"]]
        forecast = execution[execution["eligible_forecast_summary"]]
        days = int(execution["business_date"].nunique())
        plan = float(execution["plan_qty"].sum())
        produced = float(execution["produced_qty"].sum())
        demand = float(forecast["demand_qty"].sum())
        coverage = (
            float(forecast["sold_qty"].sum() / execution["sold_qty"].sum())
            if execution["sold_qty"].sum() > 0
            else 0.0
        )
        forecast_error = forecast["forecast_qty"] - forecast["demand_qty"]
        forecast_wape = float(forecast_error.abs().sum() / demand) if demand else None
        forecast_bias = float(forecast_error.sum() / demand) if demand else None
        forecast_mae = float(forecast_error.abs().mean()) if len(forecast) else None
        band = (
            "green"
            if forecast_wape is not None and forecast_wape <= 0.20
            else (
                "amber"
                if forecast_wape is not None and forecast_wape <= 0.30
                else "red"
            )
        )
        execution_error = execution["produced_qty"] - execution["plan_qty"]
        execution_bias = float(execution_error.sum() / plan) if plan else None
        execution_wape = float(execution_error.abs().sum() / plan) if plan else None
        signed_directionality = (
            float(abs(execution_error.sum()) / execution_error.abs().sum())
            if execution_error.abs().sum()
            else 0.0
        )
        tolerance = execution["plan_qty"] * 0.20
        normal = execution_error.abs().le(tolerance)
        under, over = execution_error.lt(-tolerance), execution_error.gt(tolerance)

        comparable = (
            forecast[
                forecast["issued_total_for_sale"].notna()
                & forecast["issued_yesterday_stock"].notna()
            ].copy()
            if "issued_total_for_sale" in forecast
            else forecast.iloc[0:0]
        )
        if len(comparable):
            actual_total = (
                comparable["produced_qty"] + comparable["issued_yesterday_stock"]
            )
            plan_distance = (
                comparable["issued_total_for_sale"] - comparable["demand_qty"]
            ).abs()
            actual_distance = (actual_total - comparable["demand_qty"]).abs()
            improvement = plan_distance - actual_distance
            mitigation, harmful = improvement.gt(0), improvement.lt(0)
            mitigation_available = True
        else:
            improvement = pd.Series(dtype=float)
            mitigation = harmful = pd.Series(dtype=bool)
            mitigation_available = False

        sufficient = days >= 7 and demand >= 100 and coverage >= 0.80
        model_signal = bool(
            sufficient
            and forecast_bias is not None
            and (band == "red" or abs(forecast_bias) >= 0.15)
        )
        execution_signal = bool(
            days >= 7
            and plan >= 50
            and execution_bias is not None
            and abs(execution_bias) >= 0.15
            and signed_directionality >= 0.40
        )
        mitigation_signal = "unavailable"
        if mitigation_available:
            mitigation_signal = "helpful" if improvement.sum() > 0 else "not_helpful"
        forecast_good = bool(
            sufficient
            and band in {"green", "amber"}
            and forecast_bias is not None
            and abs(forecast_bias) <= 0.10
        )
        harmful_dominates = bool(len(harmful) and harmful.sum() > mitigation.sum())
        if not sufficient:
            triage = "insufficient_data"
        elif (
            forecast_good
            and execution_signal
            and mitigation_available
            and (improvement.sum() <= 0 or harmful_dominates)
        ):
            triage = "likely_execution"
        elif model_signal and not execution_signal:
            triage = "likely_model"
        else:
            triage = "needs_joint_review"
        rows.append(
            {
                "bakery_id": int(bakery_id),
                "product_id": int(product_id),
                "product_name": group["product_name"].dropna().iloc[0]
                if "product_name" in group and group["product_name"].notna().any()
                else None,
                "days": days,
                "plan_qty": plan,
                "produced_qty": produced,
                "demand_qty": demand,
                "forecast_demand_coverage": coverage,
                "forecast_wape": forecast_wape,
                "forecast_bias": forecast_bias,
                "forecast_mae_qty": forecast_mae,
                "forecast_wape_band": band,
                "execution_bias": execution_bias,
                "execution_wape": execution_wape,
                "normal_case_share": float(normal.mean()) if len(normal) else None,
                "under_case_share": float(under.mean()) if len(under) else None,
                "over_case_share": float(over.mean()) if len(over) else None,
                "signed_directionality": signed_directionality,
                "mitigation_basis": (
                    "issued_yesterday_stock+produced_vs_issued_total_for_sale"
                )
                if mitigation_available
                else "unavailable_actual_carryover",
                "mitigation_count": int(mitigation.sum()),
                "mitigation_share": float(mitigation.mean())
                if len(mitigation)
                else None,
                "mitigation_improvement_qty": float(improvement.sum())
                if len(improvement)
                else None,
                "harmful_count": int(harmful.sum()),
                "harmful_share": float(harmful.mean()) if len(harmful) else None,
                "harmful_qty": float((-improvement.clip(upper=0)).sum())
                if len(improvement)
                else None,
                "model_signal": model_signal,
                "execution_signal": execution_signal,
                "mitigation_signal": mitigation_signal,
                "triage": triage,
            }
        )
    return pd.DataFrame(rows)


_DP_FLAG_TYPES = [
    "invalid_plan",
    "invalid_forecast",
    "demand_incomplete",
    "negative_stock",
    "sales_exceed_available",
    "invalid_production",
    "invalid_sales",
    "available_basis_missing",
    "run_mismatch",
]

_DP_SCOPE_REASONS = frozenset({"forecast_outside_fact_universe"})


def build_data_process_queue(
    detail: pd.DataFrame,
    exclusions: pd.DataFrame,
) -> pd.DataFrame:
    """Build DATA/PROCESS priority queue at (bakery_id, product_id) × issue_type grain.

    Identifies recurring DQ flags, lineage gaps, and scope issues. Assigns
    priority tiers D1 (persistent/blocking) → D2 (recurring) → D3 (occasional).

    Flags are aggregated from the ``dq_flags`` and ``blocking_flags`` columns of
    the in-memory detail DataFrame (tuples of DataQualityFlag enum values).
    Scope issues are drawn from exclusions with bakery_id and product_id set.
    """
    if detail.empty:
        return pd.DataFrame()

    def _flag_in(flags_tuple, flag_name: str) -> bool:
        return any(str(f) == flag_name for f in (flags_tuple or ()))

    rows: list[dict] = []

    for (bakery_id, product_id), group in detail.groupby(["bakery_id", "product_id"]):
        total_days = int(group["business_date"].nunique())
        total_sales = float(group["sold_qty"].clip(lower=0).sum())
        product_name = (
            group["product_name"].dropna().iloc[0]
            if "product_name" in group.columns and group["product_name"].notna().any()
            else None
        )

        # Lineage gap: rows with no forecast_run_id
        if "forecast_run_id" in group.columns:
            gap_mask = group["forecast_run_id"].isna() | group["forecast_run_id"].eq("")
            gap_days = int(gap_mask.sum())
            if gap_days:
                rows.append({
                    "bakery_id": int(bakery_id),
                    "product_id": int(product_id),
                    "product_name": product_name,
                    "issue_type": "lineage_gap",
                    "affected_days": gap_days,
                    "total_days": total_days,
                    "issue_rate": gap_days / total_days if total_days else 0.0,
                    "affected_sales_qty": float(
                        group.loc[gap_mask, "sold_qty"].clip(lower=0).sum()
                    ),
                    "total_sales_qty": total_sales,
                    "blocks_metric": True,
                })

        dq_series = group["dq_flags"] if "dq_flags" in group.columns else None
        blocking_series = (
            group["blocking_flags"] if "blocking_flags" in group.columns else None
        )

        for flag_name in _DP_FLAG_TYPES:
            if dq_series is None:
                continue
            affected_mask = dq_series.apply(lambda t: _flag_in(t, flag_name))
            affected_days = int(affected_mask.sum())
            if not affected_days:
                continue
            blocks = (
                bool(
                    blocking_series[affected_mask]
                    .apply(lambda t: _flag_in(t, flag_name))
                    .any()
                )
                if blocking_series is not None
                else False
            )
            rows.append({
                "bakery_id": int(bakery_id),
                "product_id": int(product_id),
                "product_name": product_name,
                "issue_type": flag_name,
                "affected_days": affected_days,
                "total_days": total_days,
                "issue_rate": affected_days / total_days if total_days else 0.0,
                "affected_sales_qty": float(
                    group.loc[affected_mask, "sold_qty"].clip(lower=0).sum()
                ),
                "total_sales_qty": total_sales,
                "blocks_metric": blocks,
            })

    # Scope issues from exclusions (only where bakery_id and product_id are known)
    scope_excls = exclusions[
        exclusions["reason"].isin(_DP_SCOPE_REASONS)
        & exclusions["bakery_id"].notna()
        & exclusions["product_id"].notna()
    ]
    for (bakery_id, product_id), grp in scope_excls.groupby(
        ["bakery_id", "product_id"]
    ):
        rows.append({
            "bakery_id": int(bakery_id),
            "product_id": int(product_id),
            "product_name": None,
            "issue_type": str(grp["reason"].iloc[0]),
            "affected_days": int(grp["date"].nunique()),
            "total_days": None,
            "issue_rate": None,
            "affected_sales_qty": None,
            "total_sales_qty": None,
            "blocks_metric": False,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    def _tier(row: dict) -> str:
        total = row["total_days"]
        if total is None or total < 5:
            return "D3"
        rate = row["issue_rate"] or 0.0
        blocking = bool(row["blocks_metric"])
        if (rate >= 0.50 and row["affected_days"] >= 5) or (
            blocking and rate >= 0.30
        ):
            return "D1"
        if rate >= 0.25 and row["affected_days"] >= 3:
            return "D2"
        return "D3"

    df["priority_tier"] = df.apply(_tier, axis=1)

    tier_order = {"D1": 0, "D2": 1, "D3": 2}
    df["_sort"] = df["priority_tier"].map(tier_order).fillna(3)
    df = (
        df.sort_values(["_sort", "affected_sales_qty"], ascending=[True, False])
        .drop(columns=["_sort"])
        .reset_index(drop=True)
    )
    return df


def build_coverage_diagnostics(detail: pd.DataFrame) -> pd.DataFrame:
    """Explain missing publication rows and forecast-score exclusions by volume."""

    rows = []
    missing = detail[detail["forecast_qty"].isna()]
    for dimensions in (
        ["business_date"],
        ["bakery_id", "bakery_name"],
        ["fact_category_name"],
        ["product_id", "product_name"],
    ):
        dimensions = [column for column in dimensions if column in missing.columns]
        if not dimensions:
            continue
        for keys, group in missing.groupby(dimensions, dropna=False):
            values = keys if isinstance(keys, tuple) else (keys,)
            rows.append(
                {
                    "diagnostic": "missing_published_forecast",
                    "dimension": "+".join(dimensions),
                    "value": " | ".join(map(str, values)),
                    "rows": len(group),
                    "sales_qty": group["sold_qty"].sum(),
                    "produced_qty": group["produced_qty"].sum(),
                    "stock_qty": group["closing_stock_qty"].sum(),
                }
            )
    ineligible = detail[~detail["eligible_forecast_summary"]]
    for reason, group in ineligible.groupby(
        "lost_demand_rejection_reason", dropna=False
    ):
        rows.append(
            {
                "diagnostic": "forecast_ineligible",
                "dimension": "lost_rejection_reason",
                "value": str(reason),
                "rows": len(group),
                "sales_qty": group["sold_qty"].sum(),
                "produced_qty": group["produced_qty"].sum(),
                "stock_qty": group["closing_stock_qty"].sum(),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "diagnostic",
            "dimension",
            "value",
            "rows",
            "sales_qty",
            "produced_qty",
            "stock_qty",
        ],
    )


SALES_EVENT_HEX = "D09FD180D0BED0B4D0B0D0B6D0B0"


def build_weekly_analytics_all(client, date_from: date, date_to: date) -> pd.DataFrame:
    """Build weekly analytics for ALL bakeries and ALL categories.

    Used by the embedded app's main forecast page for per-bakery weekly dynamics.
    Written to weekly_analytics_all.csv regardless of --scope.
    """
    params = {
        "date_from": date_from,
        "date_to": date_to,
        "sales_event_hex": SALES_EVENT_HEX,
    }
    df = client.query_df(
        """
        with
        check_lines_dedup as (
            select
                check_date,
                toInt64OrZero(toString(bakery_id))                          as bakery_id,
                argMax(toFloat64(quantity),         _updated_at)            as quantity,
                argMax(ifNull(toFloat64(line_amount), 0), _updated_at)     as line_amount
            from Svezhar.fct_check_lines
            where check_date between %(date_from)s and %(date_to)s
              and hex(cash_event_type) = %(sales_event_hex)s
            group by check_date, check_id, line_id, bakery_id
        ),
        sales as (
            select
                toMonday(check_date)   as week_start,
                bakery_id,
                sum(quantity)          as actual_qty,
                sum(line_amount)       as actual_revenue
            from check_lines_dedup
            group by week_start, bakery_id
        ),
        produced_dedup as (
            select
                argMax(bakery_id,   _updated_at) as bid,
                argMax(release_date,_updated_at) as rd,
                toFloat64(argMax(quantity, _updated_at)) as qty
            from Svezhar.fct_production_release
            where toDate(release_date) between %(date_from)s and %(date_to)s
            group by release_id, line_id
            having argMax(is_deleted, _updated_at) not in ('1', 'true', 'Да')
        ),
        produced as (
            select
                toMonday(toDate(rd))                as week_start,
                toInt64OrZero(toString(bid))        as bakery_id,
                sum(qty)                            as total_produced
            from produced_dedup
            group by week_start, bakery_id
        ),
        fcst as (
            select
                toMonday(forecast_date) as week_start,
                toInt64(bakery_id)      as bakery_id,
                sum(forecast_qty)       as total_forecast
            from Svezhar.sku_forecast_day_snapshots
            where lead_days = 1
              and forecast_date between %(date_from)s and %(date_to)s
            group by week_start, bakery_id
        )
        select
            s.week_start     as week_start,
            s.bakery_id      as bakery_id,
            s.actual_qty     as actual_qty,
            s.actual_revenue as actual_revenue,
            ifNull(p.total_produced, 0)  as total_produced,
            s.actual_qty                 as total_sold_mart,
            ifNull(f.total_forecast, 0)  as total_forecast
        from sales s
        left join produced p on p.week_start = s.week_start and p.bakery_id = s.bakery_id
        left join fcst f on f.week_start = s.week_start and f.bakery_id = s.bakery_id
        order by s.bakery_id, s.week_start
        """,
        parameters=params,
    )
    if df.empty:
        return df
    for col in ("total_produced", "total_sold_mart", "total_forecast"):
        if col not in df.columns:
            df[col] = 0.0
    produced = df["total_produced"].fillna(0).astype(float)
    sold_mart = df["total_sold_mart"].fillna(0).astype(float)
    forecast = df["total_forecast"].fillna(0).astype(float)
    df["execution_rate"] = produced.where(forecast > 0) / forecast.where(forecast > 0)
    denom = pd.concat([sold_mart, forecast], axis=1).max(axis=1)
    df["demand_coverage"] = sold_mart.where(denom > 0) / denom.where(denom > 0)
    actual_qty = df["actual_qty"].fillna(0).astype(float)
    actual_revenue = df["actual_revenue"].fillna(0).astype(float)
    avg_price = (actual_revenue / actual_qty).where(actual_qty > 0)
    lost_qty = (forecast - sold_mart).clip(lower=0)
    df["lost_revenue"] = (lost_qty * avg_price).where(avg_price.notna())
    result = df[["bakery_id", "week_start", "actual_qty", "actual_revenue", "execution_rate", "demand_coverage", "lost_revenue"]].copy()
    result["week_start"] = result["week_start"].astype(str).str[:10]
    return result


def _compute_kratnost_plan(
    forecast: pd.DataFrame,
    facts: pd.DataFrame,
    client,
) -> pd.DataFrame:
    """Add issued_total_for_sale / issued_plan_qty plan columns to forecast rows.

    Replicates the nightly publish_pilot_forecast.py logic using ClickHouse
    data only — no Bitrix plan Excel archive required:
      yesterday_stock  = stock_balance[D-1]  (from mart_zero_sales_60d)
      net_need         = max(forecast - yesterday_stock, 0)
      issued_plan_qty  = ceil(net_need / kratnost) * kratnost
      issued_total_for_sale = issued_plan_qty + yesterday_stock
    """
    all_pids_int = [int(p) for p in forecast["product_id"].dropna().unique()]
    if not all_pids_int:
        return forecast
    # product_id in baking_sku_meta is a zero-padded String ('000000034')
    all_pids_str = [f"{p:09d}" for p in all_pids_int]

    try:
        # product_id is String (zero-padded) in baking_sku_meta; bakery_id is Int64 Nullable
        meta_df = client.query_df(
            """
            select product_id, bakery_id, kratnost, scope, dough_group
            from Svezhar.baking_sku_meta final
            where is_active = 1 and product_id in %(pids)s
            """,
            parameters={"pids": all_pids_str},
        )
    except Exception as exc:
        print(f"  WARNING: baking_sku_meta query failed, skipping kratnost plan: {exc}")
        return forecast

    if meta_df.empty:
        print("  WARNING: baking_sku_meta returned no rows, skipping kratnost plan")
        return forecast

    # Convert String zero-padded product_id → Int64 and fill nullable bakery_id
    meta_df["product_id"] = pd.to_numeric(meta_df["product_id"], errors="coerce").astype("Int64")
    meta_df["bakery_id"] = pd.to_numeric(meta_df["bakery_id"], errors="coerce").astype("Int64")
    meta_df["bakery_id_val"] = meta_df.apply(
        lambda r: r["bakery_id"] if r["scope"] == "bakery" else pd.NA, axis=1
    ).astype("Int64")

    # Drop frozen semi-finished products — not baked, no kratnost needed
    frozen_mask = meta_df["dough_group"].str.lower().str.contains(
        "замороженные полуфабрикаты", na=False
    )
    meta_df = meta_df[~frozen_mask].copy()

    meta_df["kratnost"] = pd.to_numeric(meta_df["kratnost"], errors="coerce").clip(lower=1).fillna(1).astype(int)

    bakery_kr = (
        meta_df[meta_df["scope"] == "bakery"][["product_id", "bakery_id_val", "kratnost"]]
        .rename(columns={"bakery_id_val": "bakery_id", "kratnost": "kr_bakery"})
    )
    base_kr = (
        meta_df[meta_df["scope"] != "bakery"][["product_id", "kratnost"]]
        .drop_duplicates("product_id")
        .rename(columns={"kratnost": "kr_base"})
    )

    # Yesterday's closing stock: shift facts.stock_balance dates by +1 day
    facts_stock = facts[["date", "bakery_id", "product_id", "stock_balance"]].copy()
    facts_stock["date"] = pd.to_datetime(facts_stock["date"]).dt.date
    facts_stock["_date_key"] = facts_stock["date"].map(
        lambda d: d + pd.Timedelta(days=1)
    )
    facts_stock = facts_stock.rename(columns={"stock_balance": "yesterday_stock"})
    facts_stock = facts_stock[["_date_key", "bakery_id", "product_id", "yesterday_stock"]]

    work = forecast.copy()
    work["_date_key"] = pd.to_datetime(work["date"]).dt.date
    work["bakery_id"] = pd.to_numeric(work["bakery_id"], errors="coerce").astype("Int64")
    work["product_id"] = pd.to_numeric(work["product_id"], errors="coerce").astype("Int64")

    work = work.merge(facts_stock, on=["_date_key", "bakery_id", "product_id"], how="left")
    work["yesterday_stock"] = work["yesterday_stock"].fillna(0.0)

    # bakery_kr / base_kr product_ids already Int64 from meta_df conversion above
    bakery_kr = bakery_kr.copy()
    bakery_kr["product_id"] = bakery_kr["product_id"].astype("Int64")
    bakery_kr["bakery_id"] = bakery_kr["bakery_id"].astype("Int64")
    base_kr = base_kr.copy()
    base_kr["product_id"] = base_kr["product_id"].astype("Int64")

    work = work.merge(bakery_kr, on=["product_id", "bakery_id"], how="left")
    work = work.merge(base_kr, on="product_id", how="left")

    work["_kratnost"] = work["kr_bakery"].combine_first(work["kr_base"])
    has_kr = work["_kratnost"].notna()

    for col in ("issued_yesterday_stock", "issued_plan_qty", "issued_total_for_sale",
                "plan_format_version", "source_quality", "published_at"):
        work[col] = None

    if has_kr.any():
        kr_vals = work.loc[has_kr, "_kratnost"].astype(float)
        stock_vals = work.loc[has_kr, "yesterday_stock"].astype(float)
        fc_vals = work.loc[has_kr, "forecast_qty"].fillna(0.0).astype(float)
        net_need = (fc_vals - stock_vals).clip(lower=0.0)
        plan_arr = [
            math.ceil(n / k - 1e-9) * k if k > 0 else 0.0
            for n, k in zip(net_need.to_numpy(), kr_vals.to_numpy())
        ]
        work.loc[has_kr, "issued_yesterday_stock"] = stock_vals.to_numpy()
        work.loc[has_kr, "issued_plan_qty"] = plan_arr
        work.loc[has_kr, "issued_total_for_sale"] = (
            [p + s for p, s in zip(plan_arr, stock_vals.to_numpy())]
        )
        work.loc[has_kr, "plan_format_version"] = "stock_aware_v2"
        work.loc[has_kr, "source_quality"] = "clickhouse_derived"

    print(f"  kratnost plan: {int(has_kr.sum())}/{len(work)} forecast rows → stock_aware_v2")
    return work.drop(columns=["_date_key", "kr_bakery", "kr_base", "_kratnost", "yesterday_stock"],
                     errors="ignore")


def extract_clickhouse(
    client,
    date_from: date,
    date_to: date,
    *,
    bakery_ids: tuple[int, ...] | None = PILOT_IDS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract forecast, facts, prices and store metadata from ClickHouse.

    Pass ``bakery_ids=None`` to include all bakeries without an ID filter.
    """
    params: dict = {
        "date_from": date_from,
        "date_to": date_to,
        "categories": SCOPE_CATEGORIES,
        "sales_event_hex": SALES_EVENT_HEX,
    }
    if bakery_ids is not None:
        params["bakery_ids"] = bakery_ids
        bakery_filter = "and toInt64(bakery_id) in %(bakery_ids)s"
        stores_filter = "where toInt64(bakery_id) in %(bakery_ids)s"
        stores_params = {"bakery_ids": bakery_ids}
    else:
        bakery_filter = ""
        stores_filter = ""
        stores_params = {}

    forecast = client.query_df(
        f"""
        select forecast_date date, toInt64(bakery_id) bakery_id,
               toInt64(product_id) product_id, source_run_id, generated_at, forecast_qty
        from Svezhar.sku_forecast_day_snapshots
        where lead_days=1 and forecast_date between %(date_from)s and %(date_to)s
          {bakery_filter}
          and category_name in %(categories)s
        """,
        parameters=params,
    )
    # mart_zero_sales_60d ETL stopped updating ~2026-08-10; use fct tables instead.
    # sold: fct_check_lines with DISTINCT dedup (matches nightly pipeline)
    _fct_sold = client.query_df(
        f"""
        select
            check_date as date,
            toInt64OrZero(toString(bakery_id)) as bakery_id,
            toInt64OrZero(toString(product_id)) as product_id,
            sum(toFloat64(quantity)) as qty_sold,
            max(check_datetime) as last_sale_time
        from (
            select distinct
                check_datetime, check_date, bakery_id, product_id, quantity
            from Svezhar.fct_check_lines
            where hex(cash_event_type) = %(sales_event_hex)s
              and check_date between %(date_from)s and %(date_to)s
              {bakery_filter}
        )
        group by date, bakery_id, product_id
        """,
        parameters=params,
    )
    # produced: fct_production_release with argMax(_updated_at) dedup per (release_id, line_id)
    _fct_bakery_filter = (
        f"and toInt64OrZero(toString(bakery_id)) in %(bakery_ids)s"
        if bakery_ids is not None
        else ""
    )
    _fct_produced = client.query_df(
        f"""
        select
            toDate(rd) as date,
            toInt64OrZero(toString(bid)) as bakery_id,
            toInt64OrZero(toString(pid)) as product_id,
            sum(qty) as qty_produced
        from (
            select
                argMax(bakery_id,   _updated_at) as bid,
                argMax(product_id,  _updated_at) as pid,
                argMax(release_date,_updated_at) as rd,
                toFloat64(argMax(quantity, _updated_at)) as qty,
                argMax(is_deleted,  _updated_at) as deleted
            from Svezhar.fct_production_release
            where toDate(release_date) between %(date_from)s and %(date_to)s
              {_fct_bakery_filter}
            group by release_id, line_id
            having deleted not in ('1', 'true', 'Да')
        )
        group by date, bakery_id, product_id
        """,
        parameters=params,
    )
    # product_name / fact_category_name from forecast snapshots (mart had these denormalized)
    _fct_names = client.query_df(
        f"""
        select
            toInt64(product_id) as product_id,
            any(product_name)   as product_name,
            any(category_name)  as fact_category_name
        from Svezhar.sku_forecast_day_snapshots
        where forecast_date between %(date_from)s and %(date_to)s
          and category_name in %(categories)s
        group by product_id
        """,
        parameters=params,
    )
    # Merge sold + produced; compute stock_balance as closing fresh stock (produced - sold, ≥ 0)
    _merge_keys = ["date", "bakery_id", "product_id"]
    facts = _fct_sold.merge(_fct_produced, on=_merge_keys, how="outer")
    facts["qty_sold"] = pd.to_numeric(facts.get("qty_sold", 0), errors="coerce").fillna(0.0)
    facts["qty_produced"] = pd.to_numeric(facts.get("qty_produced", 0), errors="coerce").fillna(0.0)
    facts["stock_balance"] = (facts["qty_produced"] - facts["qty_sold"]).clip(lower=0.0)
    facts["qty_received"] = 0.0
    facts["qty_sent"] = 0.0
    facts = facts.merge(_fct_names, on="product_id", how="left")
    # Filter to pilot categories (fct tables have no category column; use product_id allowlist)
    if not _fct_names.empty:
        pilot_product_ids = set(_fct_names["product_id"].dropna().astype(int))
        facts = facts[facts["product_id"].isin(pilot_product_ids)]
    prices = client.query_df(
        f"""
        select check_date as date,
               toInt64(bakery_id) as bakery_id,
               toInt64(product_id) as product_id,
               sumIf(toFloat64(line_amount), toFloat64(quantity) > 0) as revenue
        from Svezhar.fct_check_lines
        where check_date between %(date_from)s and %(date_to)s
          {bakery_filter}
          and hex(cash_event_type) = %(sales_event_hex)s
        group by date, bakery_id, product_id
        having revenue > 0
        """,
        parameters=params,
    )
    stores = client.query_df(
        f"""
        select toInt64(bakery_id) bakery_id,
               regional_director, partner, operational_director, city
        from Svezhar.dim_stores
        {stores_filter}
        """,
        parameters=stores_params,
    )
    return forecast, facts, prices, stores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date-from", required=True, type=date.fromisoformat)
    parser.add_argument("--date-to", required=True, type=date.fromisoformat)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--plan-manifest", type=Path, default=DEFAULT_PLAN_MANIFEST)
    parser.add_argument(
        "--scope",
        choices=["pilot_38", "all"],
        default="pilot_38",
        help="pilot_38 = 38 pilot bakeries (default); all = all active bakeries",
    )
    args = parser.parse_args()

    if args.scope == "all":
        bakery_ids: tuple[int, ...] | None = None
        scope_version = "all_bakeries_v1"
    else:
        bakery_ids = PILOT_IDS
        scope_version = SCOPE_VERSION

    client = create_client(ROOT / ".env")
    weekly_analytics = build_weekly_analytics_all(client, args.date_from, args.date_to)
    forecast, facts, prices, stores = extract_clickhouse(
        client, args.date_from, args.date_to, bakery_ids=bakery_ids
    )
    # derive actual bakery scope from facts so select_coherent_lead1 checks real coverage
    actual_scope = (
        tuple(int(x) for x in sorted(facts["bakery_id"].unique()))
        if not facts.empty
        else (bakery_ids or ())
    )
    if args.plan_manifest.exists():
        archived, archive_exclusions = load_effective_plan_archive(
            args.plan_manifest, facts, date_to=args.date_to
        )
        detail, exclusions = build_detail(
            archived, facts, scope_bakery_ids=actual_scope, plans_are_confirmed=True
        )
        exclusions = pd.concat([exclusions, archive_exclusions], ignore_index=True)
        forecast_source = "effective_bitrix_attachment"
    else:
        forecast = _compute_kratnost_plan(forecast, facts, client)
        detail, exclusions = build_detail(
            forecast, facts, scope_bakery_ids=actual_scope
        )
        forecast_source = "snapshot_fallback"
    if not stores.empty and not detail.empty:
        stores_join = stores[["bakery_id", "regional_director", "partner", "operational_director", "city"]].drop_duplicates("bakery_id")
        detail = detail.merge(stores_join, on="bakery_id", how="left")
    if not prices.empty and not detail.empty:
        prices_join = prices[["date", "bakery_id", "product_id", "revenue"]].copy()
        prices_join["date"] = prices_join["date"].astype(str).str[:10]
        detail["business_date"] = detail["business_date"].astype(str).str[:10]
        detail = detail.merge(
            prices_join.rename(columns={"date": "business_date"}),
            on=["business_date", "bakery_id", "product_id"],
            how="left",
        )
        # price = revenue / sold_qty so that sold_qty * price = actual revenue from check_lines
        sold = detail["sold_qty"].clip(lower=0)
        detail["price"] = detail["revenue"] / sold.where(sold > 0)
    company, bakery, week = build_kpis(detail, scope_version=scope_version)
    bakery_ranking, sku_priority, bakery_sku_priority, daily_trend = (
        build_rankings(detail)
        if not detail.empty
        else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    )
    model_priority, execution_priority = (
        build_priority_queues(detail)
        if not detail.empty
        else (pd.DataFrame(), pd.DataFrame())
    )
    execution_triage = (
        build_execution_triage(detail) if not detail.empty else pd.DataFrame()
    )
    data_process_queue = (
        build_data_process_queue(detail, exclusions)
        if not detail.empty
        else pd.DataFrame()
    )
    diagnostics = (
        build_coverage_diagnostics(detail) if not detail.empty else pd.DataFrame()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("weekly_analytics_all", weekly_analytics),
        ("detail", detail),
        ("company_kpi", company),
        ("bakery_kpi", bakery),
        ("week_kpi", week),
        ("exclusions", exclusions),
        ("coverage_diagnostics", diagnostics),
        ("bakery_ranking", bakery_ranking),
        ("sku_priority", sku_priority),
        ("bakery_sku_priority", bakery_sku_priority),
        ("daily_trend", daily_trend),
        ("model_priority", model_priority),
        ("execution_priority", execution_priority),
        ("execution_triage", execution_triage),
        ("data_process_queue", data_process_queue),
    ):
        frame.to_csv(args.output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    payload = {
        "date_from": args.date_from.isoformat(),
        "date_to": args.date_to.isoformat(),
        "scope_version": scope_version,
        "metric_version": METRIC_VERSION,
        "forecast_source": forecast_source,
        "company": company.to_dict("records"),
        "rankings": {
            "bakery_top": bakery_ranking.head(10).to_dict("records"),
            "sku_high_priority": sku_priority[
                sku_priority.get("priority_tier", pd.Series(dtype=object)).eq("high")
            ]
            .head(20)
            .to_dict("records"),
            "bakery_sku_high_priority": bakery_sku_priority[
                bakery_sku_priority.get("priority_tier", pd.Series(dtype=object)).eq(
                    "high"
                )
            ]
            .head(30)
            .to_dict("records"),
            "priority_rule": (
                "high: contribution>=1% and at least two flags; flags are "
                "material_abs_error "
                "and systematic direction (>=70% error in one direction on >=3 days)"
            ),
            "model_m1": model_priority[
                model_priority.get("priority_tier", pd.Series(dtype=object)).eq("M1")
            ].to_dict("records"),
            "execution_e1": execution_priority[
                execution_priority.get("priority_tier", pd.Series(dtype=object)).eq(
                    "E1"
                )
            ].to_dict("records"),
            "execution_responsibility_rule": (
                "forecast_reliability_v1: reliable when forecast days>=7, demand>=100, "
                "coverage>=60%, WAPE<=20%, |bias|<=10%; responsibility combines "
                "forecast reliability with execution |bias|>=15%, while "
                "closer_to_demand_than_plan records reasonable deviations"
            ),
            "authoritative_execution_output": "execution_triage.csv",
            "legacy_priority_outputs": "diagnostic_legacy",
        },
        "coverage": {
            "source_fact_rows": len(facts),
            "source_fact_volume": float(facts["qty_sold"].sum()),
            "expected_fact_rows": len(detail),
            "missing_forecast_rows": (
                int(detail["forecast_qty"].isna().sum()) if not detail.empty else 0
            ),
            "extra_forecast_rows": int(
                exclusions["reason"].eq("forecast_outside_fact_universe").sum()
            ),
            "excluded_dates": int(
                exclusions["reason"]
                .isin(
                    (
                        "no_pre_cutoff_run",
                        "no_full_scope_run",
                        "ambiguous_full_run",
                        "no_forecast_snapshot",
                    )
                )
                .sum()
            ),
            "published_run_proof": forecast_source == "effective_bitrix_attachment",
            "lineage_note": (
                "effective Bitrix attachment with message/disk/hash provenance"
                if forecast_source == "effective_bitrix_attachment"
                else "pre-06:00 coherent snapshot; publication is not proven"
            ),
            "stock_aware_execution_rows": (
                int(
                    detail.get("plan_format_version", pd.Series(dtype=object))
                    .eq("stock_aware_v2")
                    .sum()
                )
                if not detail.empty
                else 0
            ),
        },
        "exclusions": exclusions.to_dict("records"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
