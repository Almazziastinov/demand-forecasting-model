"""Service layer for pilot management analytics (PM-05).

Reads from the CSV report directory produced by build_pilot_management_summary.py.
Public API returns plain dicts/lists — no pandas DataFrames cross the boundary.

Designed to be replaceable with a ClickHouse-backed implementation once the
canonical mart (PM-04, Svezhar.pilot_performance_sku_day) is deployed to
production.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

_FLAG_RE = re.compile(r"'([^']+)'")

_TIER_ORDER = {"M1": 0, "M2": 1, "E1": 0, "E2": 1, "D1": 0, "D2": 1}


def _parse_flag_str(value: Any) -> list[str]:
    """Extract flag slugs from a CSV tuple repr like '(<X.A: 'a'>,)'."""
    if not value or str(value) in ("()", "nan"):
        return []
    return _FLAG_RE.findall(str(value))


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def _pct(value: float | None, ndigits: int = 1) -> str | None:
    """Format as percentage string; returns None when value is None."""
    if value is None:
        return None
    return f"{value * 100:.{ndigits}f}%"


class PilotManagementService:
    """Read pilot management reports from a local CSV directory.

    All query methods are stateless: they reload from disk each call.
    The OS page cache keeps this fast for the small files used here.
    """

    def __init__(self, report_dir: Path) -> None:
        self._dir = Path(report_dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, name: str) -> pd.DataFrame:
        path = self._dir / f"{name}.csv"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    def _load_json(self, name: str) -> dict:
        path = self._dir / f"{name}.json"
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def _bakery_names(self) -> dict[int, str]:
        detail = self._load("detail")
        if detail.empty or "bakery_name" not in detail.columns:
            return {}
        return {
            int(row["bakery_id"]): str(row["bakery_name"])
            for _, row in detail[["bakery_id", "bakery_name"]]
            .drop_duplicates("bakery_id")
            .iterrows()
        }

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    def get_pilot_summary(self) -> dict | None:
        """Pilot-level KPIs with DQ status and period metadata."""
        summary_json = self._load_json("summary")
        if not summary_json:
            return None
        company_kpi = self._load("company_kpi")
        kpi: dict = company_kpi.iloc[0].to_dict() if not company_kpi.empty else {}

        week_kpi = self._load("week_kpi")
        has_partial = (
            bool(week_kpi["partial_period"].any())
            if not week_kpi.empty and "partial_period" in week_kpi.columns
            else False
        )
        dp_queue = self._load("data_process_queue")
        d1_count = (
            int((dp_queue["priority_tier"] == "D1").sum())
            if not dp_queue.empty and "priority_tier" in dp_queue.columns
            else 0
        )
        has_exec = bool(kpi.get("execution_kpi_included", False))
        n_eligible = int(kpi.get("rows_forecast_eligible") or 0)
        abs_error = _maybe_float(kpi.get("absolute_error_qty")) or 0.0
        error_qty_val = _maybe_float(kpi.get("error_qty")) or 0.0
        mae_qty = abs_error / n_eligible if n_eligible > 0 else None
        bias_qty = error_qty_val / n_eligible if n_eligible > 0 else None

        execution_rate = None
        kratnost_wape = None
        kratnost_bias = None
        kratnost_bias_qty = None
        kratnost_mae_qty = None
        kratnost_lost_qty = None
        kratnost_n_eligible = 0
        kratnost_execution_rate = None
        coverage_sku_eligible = 0
        coverage_sku_total = 0

        detail = self._load("detail")
        if not detail.empty:
            if has_exec and "eligible_execution" in detail.columns:
                exec_rows = detail[detail["eligible_execution"].astype(bool)]
                plan_sum = float(exec_rows["plan_qty"].sum()) if "plan_qty" in exec_rows.columns else 0.0
                produced_sum = float(exec_rows["produced_qty"].sum()) if "produced_qty" in exec_rows.columns else 0.0
                execution_rate = produced_sum / plan_sum if plan_sum > 0 else None

            # кратность block: issued_total_for_sale vs demand on forecast-eligible rows
            if "issued_total_for_sale" in detail.columns and "eligible_forecast_summary" in detail.columns:
                krat = detail[
                    detail["issued_total_for_sale"].notna()
                    & detail["eligible_forecast_summary"].astype(bool)
                ]
                if not krat.empty:
                    krat_demand = float(krat["demand_qty"].sum())
                    krat_err = krat["issued_total_for_sale"] - krat["demand_qty"]
                    kratnost_wape = float(krat_err.abs().sum() / krat_demand) if krat_demand > 0 else None
                    kratnost_bias = float(krat_err.sum() / krat_demand) if krat_demand > 0 else None
                    kratnost_mae_qty = float(krat_err.abs().sum() / len(krat))
                    kratnost_bias_qty = float(krat_err.sum() / len(krat))
                    kratnost_n_eligible = len(krat)
                    unmet = (krat["demand_qty"] - krat["issued_total_for_sale"]).clip(lower=0)
                    kratnost_lost_qty = float(unmet.sum())
                if "produced_qty" in detail.columns:
                    both = detail[
                        detail["eligible_forecast_summary"].astype(bool)
                        & detail["produced_qty"].notna()
                    ]
                    both_demand = float(both["demand_qty"].sum())
                    both_produced = float(both["produced_qty"].sum())
                    kratnost_execution_rate = both_produced / both_demand if both_demand > 0 else None

            # unique SKU coverage
            if "product_id" in detail.columns and "eligible_forecast_summary" in detail.columns:
                coverage_sku_total = int(detail["product_id"].nunique())
                coverage_sku_eligible = int(
                    detail.loc[detail["eligible_forecast_summary"].astype(bool), "product_id"].nunique()
                )

        return {
            "date_from": summary_json.get("date_from"),
            "date_to": summary_json.get("date_to"),
            "scope_version": summary_json.get("scope_version"),
            "metric_version": summary_json.get("metric_version"),
            "forecast_source": summary_json.get("forecast_source"),
            "is_partial_period": has_partial,
            "wape": _maybe_float(kpi.get("wape")),
            "mae_qty": mae_qty,
            "bias": _maybe_float(kpi.get("bias")),
            "bias_qty": bias_qty,
            "demand_qty": _maybe_float(kpi.get("demand_qty")),
            "recognized_lost_qty": _maybe_float(kpi.get("recognized_lost_qty")),
            "forecast_coverage": _maybe_float(kpi.get("forecast_coverage")),
            "coverage_eligible": n_eligible,
            "coverage_total": int(kpi.get("rows_total") or 0),
            "execution_wape": _maybe_float(kpi.get("execution_wape")) if has_exec else None,
            "execution_bias": _maybe_float(kpi.get("execution_bias")) if has_exec else None,
            "execution_coverage": _maybe_float(kpi.get("execution_coverage")) if has_exec else None,
            "execution_rate": execution_rate,
            "kratnost_wape": kratnost_wape,
            "kratnost_bias": kratnost_bias,
            "kratnost_bias_qty": kratnost_bias_qty,
            "kratnost_mae_qty": kratnost_mae_qty,
            "kratnost_lost_qty": kratnost_lost_qty,
            "kratnost_n_eligible": kratnost_n_eligible,
            "kratnost_execution_rate": kratnost_execution_rate,
            "coverage_sku_eligible": coverage_sku_eligible,
            "coverage_sku_total": coverage_sku_total,
            "d1_issue_count": d1_count,
            "has_dq_issues": d1_count > 0,
        }

    def get_bakery_list(self) -> list[dict]:
        """Bakery-level KPIs sorted by WAPE descending (worst first)."""
        bakery_kpi = self._load("bakery_kpi")
        if bakery_kpi.empty:
            return []
        names = self._bakery_names()
        rows = []
        for _, row in bakery_kpi.iterrows():
            bid = int(row["bakery_id"])
            has_exec = bool(row.get("execution_kpi_included", False))
            rows.append({
                "bakery_id": bid,
                "bakery_name": names.get(bid, f"Пекарня {bid}"),
                "wape": _maybe_float(row.get("wape")),
                "bias": _maybe_float(row.get("bias")),
                "forecast_coverage": _maybe_float(row.get("forecast_coverage")),
                "demand_qty": _maybe_float(row.get("demand_qty")),
                "recognized_lost_qty": _maybe_float(row.get("recognized_lost_qty")),
                "execution_wape": (
                    _maybe_float(row.get("execution_wape")) if has_exec else None
                ),
                "execution_bias": (
                    _maybe_float(row.get("execution_bias")) if has_exec else None
                ),
            })
        return sorted(rows, key=lambda r: r["wape"] or 0, reverse=True)

    def get_bakery_detail(self, bakery_id: int) -> dict | None:
        """Single bakery KPI row, or None if not found."""
        return next(
            (r for r in self.get_bakery_list() if r["bakery_id"] == bakery_id),
            None,
        )

    def get_sku_list(self, bakery_id: int) -> list[dict]:
        """Per-SKU KPIs for one bakery, aggregated from detail rows."""
        detail = self._load("detail")
        if detail.empty:
            return []
        group = detail[detail["bakery_id"] == bakery_id]
        if group.empty:
            return []
        rows = []
        for product_id, sku_group in group.groupby("product_id"):
            eligible = sku_group[sku_group["eligible_forecast_summary"].astype(bool)]
            error = eligible["forecast_qty"] - eligible["demand_qty"]
            demand = float(eligible["demand_qty"].sum())
            wape = float(error.abs().sum() / demand) if demand > 0 else None
            bias = float(error.sum() / demand) if demand > 0 else None
            exec_eligible = sku_group[sku_group["eligible_execution"].astype(bool)]
            plan = float(exec_eligible["plan_qty"].sum())
            exec_error = exec_eligible["produced_qty"] - exec_eligible["plan_qty"]
            exec_wape = float(exec_error.abs().sum() / plan) if plan > 0 else None
            exec_bias = float(exec_error.sum() / plan) if plan > 0 else None
            has_blocking = (
                sku_group["blocking_flags"].ne("()").any()
                if "blocking_flags" in sku_group.columns
                else False
            )
            pname = (
                sku_group["product_name"].dropna().iloc[0]
                if "product_name" in sku_group.columns
                and sku_group["product_name"].notna().any()
                else None
            )
            rows.append({
                "product_id": int(product_id),
                "product_name": pname,
                "eligible_days": int(eligible["business_date"].nunique()),
                "total_days": int(sku_group["business_date"].nunique()),
                "demand_qty": demand,
                "sold_qty": float(sku_group["sold_qty"].clip(lower=0).sum()),
                "wape": wape,
                "bias": bias,
                "exec_wape": exec_wape,
                "exec_bias": exec_bias,
                "has_blocking": bool(has_blocking),
            })
        return sorted(rows, key=lambda r: r["demand_qty"], reverse=True)

    def get_day_list(self, bakery_id: int, product_id: int) -> list[dict]:
        """Day-level rows for a single bakery × SKU, sorted by date."""
        detail = self._load("detail")
        if detail.empty:
            return []
        mask = (detail["bakery_id"] == bakery_id) & (detail["product_id"] == product_id)
        group = detail[mask].sort_values("business_date")
        rows = []
        for _, row in group.iterrows():
            rows.append({
                "business_date": str(row.get("business_date", "")),
                "forecast_qty": _maybe_float(row.get("forecast_qty")),
                "plan_qty": _maybe_float(row.get("plan_qty")),
                "produced_qty": _maybe_float(row.get("produced_qty")),
                "sold_qty": _maybe_float(row.get("sold_qty")),
                "demand_qty": _maybe_float(row.get("demand_qty")),
                "eligible_forecast": bool(row.get("eligible_forecast_summary", False)),
                "eligible_execution": bool(row.get("eligible_execution", False)),
                "forecast_status": str(row.get("forecast_status", "")),
                "execution_status": str(row.get("execution_status", "")),
                "blocking_flags": _parse_flag_str(row.get("blocking_flags")),
                "dq_flags": _parse_flag_str(row.get("dq_flags")),
            })
        return rows

    def get_week_trend(self) -> list[dict]:
        """Weekly KPI rows for trend view, oldest first."""
        week_kpi = self._load("week_kpi")
        if week_kpi.empty:
            return []
        rows = []
        for _, row in week_kpi.iterrows():
            has_exec = bool(row.get("execution_kpi_included", False))
            rows.append({
                "week_start": str(row.get("week_start", "")),
                "partial_period": bool(row.get("partial_period", False)),
                "wape": _maybe_float(row.get("wape")),
                "bias": _maybe_float(row.get("bias")),
                "demand_qty": _maybe_float(row.get("demand_qty")),
                "forecast_coverage": _maybe_float(row.get("forecast_coverage")),
                "execution_wape": (
                    _maybe_float(row.get("execution_wape")) if has_exec else None
                ),
            })
        return rows

    def get_model_queue(
        self, tiers: tuple[str, ...] = ("M1",)
    ) -> list[dict]:
        """MODEL priority queue rows filtered to given tiers."""
        df = self._load("model_priority")
        if df.empty:
            return []
        if "priority_tier" in df.columns and tiers:
            df = df[df["priority_tier"].isin(tiers)]
        return df.to_dict("records")

    def get_execution_queue(
        self,
        triage_filter: tuple[str, ...] | None = ("likely_execution", "needs_joint_review"),
    ) -> list[dict]:
        """EXECUTION TRIAGE queue rows filtered by triage status."""
        df = self._load("execution_triage")
        if df.empty:
            return []
        if triage_filter is not None and "triage" in df.columns:
            df = df[df["triage"].isin(triage_filter)]
        return df.to_dict("records")

    def get_data_process_queue(
        self, tiers: tuple[str, ...] = ("D1", "D2")
    ) -> list[dict]:
        """DATA/PROCESS queue rows filtered to given tiers."""
        df = self._load("data_process_queue")
        if df.empty:
            return []
        if "priority_tier" in df.columns and tiers:
            df = df[df["priority_tier"].isin(tiers)]
        return df.to_dict("records")

    def get_dq_summary(self) -> dict:
        """Aggregate DQ issue counts for the warning banner."""
        df = self._load("data_process_queue")
        if df.empty:
            return {
                "d1_count": 0,
                "d2_count": 0,
                "blocking_count": 0,
                "top_issue_types": [],
            }
        tier_col = df.get("priority_tier", pd.Series(dtype=object))
        d1_count = int((tier_col == "D1").sum())
        d2_count = int((tier_col == "D2").sum())
        blocking_count = (
            int(df["blocks_metric"].astype(bool).sum())
            if "blocks_metric" in df.columns
            else 0
        )
        top_issues = (
            df.groupby("issue_type")["affected_days"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .reset_index()
            .to_dict("records")
        ) if "issue_type" in df.columns and "affected_days" in df.columns else []
        return {
            "d1_count": d1_count,
            "d2_count": d2_count,
            "blocking_count": blocking_count,
            "top_issue_types": top_issues,
        }
