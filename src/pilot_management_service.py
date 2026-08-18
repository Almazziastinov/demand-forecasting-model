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


def _demand_coverage(sold: float, recognized_lost: float | None) -> float | None:
    """sold / (sold + recognized_lost) — fraction of confirmed demand that was served."""
    rl = recognized_lost or 0.0
    total = sold + rl
    return sold / total if total > 0 else None


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

    def _cat_col(self, df: pd.DataFrame) -> str | None:
        for col in ("category_name", "fact_category_name"):
            if col in df.columns:
                return col
        return None

    def _filter_cat(self, df: pd.DataFrame, category: str | None) -> pd.DataFrame:
        if not category or df.empty:
            return df
        col = self._cat_col(df)
        return df[df[col] == category] if col else df

    def _apply_filters(
        self,
        df: pd.DataFrame,
        *,
        category: str | None = None,
        regional_director: str | None = None,
        partner: str | None = None,
        bakery_id: int | str | None = None,
        product_id: int | str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> pd.DataFrame:
        if df.empty:
            return df
        if category:
            col = self._cat_col(df)
            if col:
                df = df[df[col] == category]
        if regional_director and "regional_director" in df.columns:
            df = df[df["regional_director"] == regional_director]
        if partner and "partner" in df.columns:
            df = df[df["partner"] == partner]
        if bakery_id is not None and "bakery_id" in df.columns:
            df = df[df["bakery_id"] == int(bakery_id)]
        if product_id is not None and "product_id" in df.columns:
            df = df[df["product_id"] == int(product_id)]
        if (date_from or date_to) and "business_date" in df.columns:
            dates = pd.to_datetime(df["business_date"], errors="coerce")
            if date_from:
                df = df[dates >= pd.Timestamp(date_from)]
            if date_to:
                df = df[dates <= pd.Timestamp(date_to)]
        return df

    def _kratnost_metrics(self, df: pd.DataFrame) -> dict:
        """Compute block-2 (kratnost) metrics from a detail slice.

        When issued_total_for_sale (Bitrix plan) is present, uses plan-based
        execution and recognized-lost calculation.  Falls back to Block-1
        forecast-based metrics so the UI always shows meaningful values even
        when no plan archive is available.
        """
        out: dict = {
            "kratnost_recognized_lost_qty": None,
            "kratnost_recognized_lost_revenue": None,
            "kratnost_execution_rate": None,
        }
        if df.empty:
            return out

        # --- plan-based (kratnost) path ---
        if "issued_total_for_sale" in df.columns and df["issued_total_for_sale"].notna().any():
            krat = df[
                df["issued_total_for_sale"].notna()
                & df["eligible_forecast_summary"].astype(bool)
                & df["produced_qty"].notna()
            ]
            if not krat.empty:
                issued = float(krat["issued_total_for_sale"].sum())
                produced = float(krat["produced_qty"].sum())
                out["kratnost_execution_rate"] = produced / issued if issued > 0 else None
                ld = krat[
                    krat["eligible_lost_demand"].astype(bool)
                    & (krat["issued_total_for_sale"] > krat["sold_qty"])
                ]
                if not ld.empty:
                    capped = (ld["issued_total_for_sale"] - ld["sold_qty"]).clip(
                        upper=ld["lost_demand_recognized_qty"]
                    )
                    out["kratnost_recognized_lost_qty"] = float(capped.sum())
                    if "price" in ld.columns:
                        out["kratnost_recognized_lost_revenue"] = float(
                            (capped * ld["price"].fillna(0)).sum()
                        )
            return out

        # --- fallback: Block-1 forecast-based metrics ---
        if "eligible_forecast_summary" not in df.columns or "forecast_qty" not in df.columns:
            return out
        b1 = df[df["eligible_forecast_summary"].astype(bool) & df["produced_qty"].notna()]
        if not b1.empty:
            b1_forecast = float(b1["forecast_qty"].sum())
            b1_produced = float(b1["produced_qty"].sum())
            out["kratnost_execution_rate"] = b1_produced / b1_forecast if b1_forecast > 0 else None
        if (
            "eligible_lost_demand" in df.columns
            and "lost_demand_recognized_qty" in df.columns
            and "sold_qty" in df.columns
        ):
            ld = df[
                df["eligible_lost_demand"].astype(bool)
                & (df["forecast_qty"] > df["sold_qty"])
            ]
            if not ld.empty:
                capped = (ld["forecast_qty"] - ld["sold_qty"]).clip(
                    upper=ld["lost_demand_recognized_qty"]
                )
                out["kratnost_recognized_lost_qty"] = float(capped.sum())
                if "price" in ld.columns:
                    out["kratnost_recognized_lost_revenue"] = float(
                        (capped * ld["price"].fillna(0)).sum()
                    )
        return out

    def _kpi_block(self, df: pd.DataFrame) -> dict:
        """Compute 3-value KPI block: Plan AI / Production / Sales + derived metrics.

        Plan AI  = issued_total_for_sale (kratnost-adjusted baking plan)
        Produced = produced_qty (actual baked)
        Sold     = sold_qty (actual sold)
        """
        out: dict = {
            "plan_qty": None,
            "produced_qty": None,
            "sold_qty": None,
            "execution_rate": None,
            "sellthrough_rate": None,
            "recognized_lost_qty": None,
            "recognized_lost_revenue": None,
        }
        if df.empty:
            return out

        has_issued = (
            "issued_total_for_sale" in df.columns
            and df["issued_total_for_sale"].notna().any()
        )
        if not has_issued or "eligible_forecast_summary" not in df.columns or "produced_qty" not in df.columns:
            return out

        krat = df[
            df["issued_total_for_sale"].notna()
            & df["eligible_forecast_summary"].astype(bool)
            & df["produced_qty"].notna()
        ]
        if not krat.empty:
            plan = float(krat["issued_total_for_sale"].sum())
            produced = float(krat["produced_qty"].sum())
            # Total sold = fresh produced + yesterday stock - unsold closing stock
            # (sold_qty covers only same-day fresh sales; this includes discounted carryover)
            ys = krat["issued_yesterday_stock"].fillna(0) if "issued_yesterday_stock" in krat.columns else 0
            cs = krat["closing_stock_qty"].fillna(0) if "closing_stock_qty" in krat.columns else 0
            sold = float((krat["produced_qty"] + ys - cs).clip(lower=0).sum())
            out["plan_qty"] = plan
            out["produced_qty"] = produced
            out["sold_qty"] = sold
            out["execution_rate"] = produced / plan if plan > 0 else None
            out["sellthrough_rate"] = sold / plan if plan > 0 else None

        krat_elig = df[
            df["issued_total_for_sale"].notna()
            & df["eligible_forecast_summary"].astype(bool)
        ]
        if (
            "eligible_lost_demand" in krat_elig.columns
            and "lost_demand_recognized_qty" in krat_elig.columns
            and "sold_qty" in krat_elig.columns
        ):
            ld = krat_elig[
                krat_elig["eligible_lost_demand"].astype(bool)
                & (krat_elig["issued_total_for_sale"] > krat_elig["sold_qty"])
            ]
            if not ld.empty:
                capped = (ld["issued_total_for_sale"] - ld["sold_qty"]).clip(
                    upper=ld["lost_demand_recognized_qty"]
                )
                out["recognized_lost_qty"] = float(capped.sum())
                if "price" in ld.columns:
                    out["recognized_lost_revenue"] = float(
                        (capped * ld["price"].fillna(0)).sum()
                    )
        return out

    def get_available_categories(self) -> list[str]:
        detail = self._load("detail")
        if detail.empty:
            return []
        col = self._cat_col(detail)
        if col is None:
            return []
        return sorted(detail[col].dropna().unique().tolist())

    def get_available_regional_directors(self) -> list[str]:
        detail = self._load("detail")
        if detail.empty or "regional_director" not in detail.columns:
            return []
        return sorted(detail["regional_director"].dropna().unique().tolist())

    def get_available_partners(self) -> list[str]:
        detail = self._load("detail")
        if detail.empty or "partner" not in detail.columns:
            return []
        return sorted(detail["partner"].dropna().unique().tolist())

    def get_available_bakery_names(self) -> list[dict]:
        detail = self._load("detail")
        if detail.empty or "bakery_name" not in detail.columns:
            return []
        return (
            detail[["bakery_id", "bakery_name"]]
            .drop_duplicates("bakery_id")
            .sort_values("bakery_name")
            .rename(columns={"bakery_id": "id", "bakery_name": "name"})
            .to_dict("records")
        )

    def get_available_products(self) -> list[dict]:
        detail = self._load("detail")
        if detail.empty or "product_name" not in detail.columns:
            return []
        return (
            detail[["product_id", "product_name"]]
            .drop_duplicates("product_id")
            .dropna(subset=["product_name"])
            .sort_values("product_name")
            .rename(columns={"product_id": "id", "product_name": "name"})
            .to_dict("records")
        )

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    def get_pilot_summary(
        self,
        category: str | None = None,
        regional_director: str | None = None,
        partner: str | None = None,
        bakery_id: int | str | None = None,
        product_id: int | str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict | None:
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
        block1_lost_qty = None
        block1_lost_revenue = None
        block1_recognized_lost_qty = None
        block1_recognized_lost_revenue = None
        kratnost_wape = None
        kratnost_bias = None
        kratnost_bias_qty = None
        kratnost_mae_qty = None
        kratnost_lost_qty = None
        kratnost_n_eligible = 0
        kratnost_lost_revenue = None
        kratnost_recognized_lost_qty = None
        kratnost_recognized_lost_revenue = None
        kratnost_execution_rate = None
        coverage_sku_eligible = 0
        coverage_sku_total = 0

        actual_revenue = None
        detail = self._load("detail")
        detail = self._apply_filters(
            detail,
            category=category,
            regional_director=regional_director,
            partner=partner,
            bakery_id=bakery_id,
            product_id=product_id,
            date_from=date_from,
            date_to=date_to,
        )
        if not detail.empty:
            if "sold_qty" in detail.columns and "price" in detail.columns:
                actual_revenue = float(
                    (detail["sold_qty"].clip(lower=0) * detail["price"].fillna(0)).sum()
                )
            # Block 1: execution = produced / forecast
            # lost = max(0, forecast - sold); recognized_lost = sum(lost_demand_recognized_qty)
            if "eligible_forecast_summary" in detail.columns:
                b1 = detail[detail["eligible_forecast_summary"].astype(bool)]
                if not b1.empty and "forecast_qty" in b1.columns:
                    if "sold_qty" in b1.columns:
                        b1_lost = (b1["forecast_qty"] - b1["sold_qty"]).clip(lower=0)
                        block1_lost_qty = float(b1_lost.sum())
                        if "price" in b1.columns:
                            block1_lost_revenue = float((b1_lost * b1["price"].fillna(0)).sum())
                    if "produced_qty" in b1.columns:
                        b1_exec = b1[b1["produced_qty"].notna()]
                        if not b1_exec.empty:
                            b1_forecast = float(b1_exec["forecast_qty"].sum())
                            b1_produced = float(b1_exec["produced_qty"].sum())
                            execution_rate = b1_produced / b1_forecast if b1_forecast > 0 else None
            if (
                "eligible_lost_demand" in detail.columns
                and "lost_demand_recognized_qty" in detail.columns
                and "forecast_qty" in detail.columns
                and "sold_qty" in detail.columns
            ):
                ld = detail[
                    detail["eligible_lost_demand"].astype(bool)
                    & (detail["forecast_qty"] > detail["sold_qty"])
                ]
                if not ld.empty:
                    ld_capped = (ld["forecast_qty"] - ld["sold_qty"]).clip(upper=ld["lost_demand_recognized_qty"])
                    block1_recognized_lost_qty = float(ld_capped.sum())
                    if "price" in ld.columns:
                        block1_recognized_lost_revenue = float((ld_capped * ld["price"].fillna(0)).sum())

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
                    # Block 2: lost = max(0, issued - produced); execution = produced / issued
                    if "produced_qty" in krat.columns:
                        krat_exec = krat[krat["produced_qty"].notna()]
                        if not krat_exec.empty:
                            issued_sum = float(krat_exec["issued_total_for_sale"].sum())
                            b2_produced = float(krat_exec["produced_qty"].sum())
                            kratnost_execution_rate = b2_produced / issued_sum if issued_sum > 0 else None
                            krat_lost = (krat_exec["issued_total_for_sale"] - krat_exec["produced_qty"]).clip(lower=0)
                            kratnost_lost_qty = float(krat_lost.sum())
                            if "price" in krat_exec.columns:
                                kratnost_lost_revenue = float((krat_lost * krat_exec["price"].fillna(0)).sum())
                # Block 2 recognized: min(issued - sold, lost_demand_recognized) where issued > sold
                if (
                    "eligible_lost_demand" in krat.columns
                    and "lost_demand_recognized_qty" in krat.columns
                    and "sold_qty" in krat.columns
                ):
                    ld2 = krat[
                        krat["eligible_lost_demand"].astype(bool)
                        & (krat["issued_total_for_sale"] > krat["sold_qty"])
                    ]
                    if not ld2.empty:
                        ld2_capped = (
                            (ld2["issued_total_for_sale"] - ld2["sold_qty"])
                            .clip(upper=ld2["lost_demand_recognized_qty"])
                        )
                        kratnost_recognized_lost_qty = float(ld2_capped.sum())
                        if "price" in ld2.columns:
                            kratnost_recognized_lost_revenue = float(
                                (ld2_capped * ld2["price"].fillna(0)).sum()
                            )

            # Fallback to block1 metrics when no Bitrix plan data
            if kratnost_execution_rate is None:
                kratnost_execution_rate = execution_rate
            if kratnost_recognized_lost_qty is None and block1_recognized_lost_qty is not None:
                kratnost_recognized_lost_qty = block1_recognized_lost_qty
                kratnost_recognized_lost_revenue = block1_recognized_lost_revenue
            if kratnost_lost_qty is None and block1_lost_qty is not None:
                kratnost_lost_qty = block1_lost_qty
                kratnost_lost_revenue = block1_lost_revenue

            # unique SKU coverage (computed from filtered detail, not from precomputed KPI)
            if "product_id" in detail.columns and "eligible_forecast_summary" in detail.columns:
                coverage_sku_total = int(detail["product_id"].nunique())
                coverage_sku_eligible = int(
                    detail.loc[detail["eligible_forecast_summary"].astype(bool), "product_id"].nunique()
                )

        # 3-value KPI block (Plan AI / Produced / Sold)
        kpi3 = self._kpi_block(detail) if not detail.empty else self._kpi_block(pd.DataFrame())

        sold_sum = float(detail["sold_qty"].clip(lower=0).sum()) if not detail.empty and "sold_qty" in detail.columns else 0.0
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
            "actual_revenue": actual_revenue,
            "block1_lost_qty": block1_lost_qty,
            "block1_lost_revenue": block1_lost_revenue,
            "block1_recognized_lost_qty": block1_recognized_lost_qty,
            "block1_recognized_lost_revenue": block1_recognized_lost_revenue,
            "block1_demand_coverage": _demand_coverage(sold_sum, block1_recognized_lost_qty),
            "demand_coverage": _demand_coverage(sold_sum, kratnost_recognized_lost_qty),
            "forecast_coverage": coverage_sku_eligible / coverage_sku_total if coverage_sku_total > 0 else _maybe_float(kpi.get("forecast_coverage")),
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
            "kratnost_lost_revenue": kratnost_lost_revenue,
            "kratnost_recognized_lost_qty": kratnost_recognized_lost_qty,
            "kratnost_recognized_lost_revenue": kratnost_recognized_lost_revenue,
            "kratnost_n_eligible": kratnost_n_eligible,
            "kratnost_execution_rate": kratnost_execution_rate,
            "coverage_sku_eligible": coverage_sku_eligible,
            "coverage_sku_total": coverage_sku_total,
            "d1_issue_count": d1_count,
            "has_dq_issues": d1_count > 0,
            # 3-value KPI block (new primary display)
            "plan_qty": kpi3["plan_qty"],
            "kpi_produced_qty": kpi3["produced_qty"],
            "kpi_sold_qty": kpi3["sold_qty"],
            "kpi_execution_rate": kpi3["execution_rate"],
            "kpi_sellthrough_rate": kpi3["sellthrough_rate"],
            "kpi_recognized_lost_qty": kpi3["recognized_lost_qty"],
            "kpi_recognized_lost_revenue": kpi3["recognized_lost_revenue"],
        }

    def get_bakery_list(
        self,
        category: str | None = None,
        regional_director: str | None = None,
        partner: str | None = None,
        bakery_id: int | str | None = None,
        product_id: int | str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict]:
        """Bakery-level KPIs sorted by demand coverage ascending (worst first)."""
        detail = self._load("detail")
        detail = self._apply_filters(
            detail,
            category=category,
            regional_director=regional_director,
            partner=partner,
            bakery_id=bakery_id,
            product_id=product_id,
            date_from=date_from,
            date_to=date_to,
        )
        if detail.empty:
            return []
        rows = []
        for bid, grp in detail.groupby("bakery_id"):
            km = self._kpi_block(grp)
            actual_rev = float((grp["sold_qty"].clip(lower=0) * grp["price"].fillna(0)).sum()) if "price" in grp.columns else None
            name = (
                grp["bakery_name"].dropna().iloc[0]
                if "bakery_name" in grp.columns and grp["bakery_name"].notna().any()
                else f"Пекарня {int(bid)}"
            )
            rows.append({
                "bakery_id": int(bid),
                "bakery_name": str(name),
                "plan_qty": km["plan_qty"],
                "produced_qty": km["produced_qty"],
                "sold_qty": km["sold_qty"],
                "execution_rate": km["execution_rate"],
                "sellthrough_rate": km["sellthrough_rate"],
                "recognized_lost_qty": km["recognized_lost_qty"],
                "recognized_lost_revenue": km["recognized_lost_revenue"],
                "actual_revenue": actual_rev,
            })
        return sorted(rows, key=lambda r: r["execution_rate"] if r["execution_rate"] is not None else 2.0)

    def get_bakery_detail(self, bakery_id: int, **filter_kwargs) -> dict | None:
        """Single bakery KPI row, or None if not found."""
        return next(
            (r for r in self.get_bakery_list(**filter_kwargs) if r["bakery_id"] == bakery_id),
            None,
        )

    def get_bakery_kpi(self, bakery_id: int, category: str | None = None) -> dict | None:
        """Full KPI set for one bakery (mirrors get_pilot_summary structure)."""
        detail = self._load("detail")
        if detail.empty:
            return None
        detail = detail[detail["bakery_id"] == bakery_id].copy()
        if detail.empty:
            return None
        name = (
            detail["bakery_name"].dropna().iloc[0]
            if "bakery_name" in detail.columns and detail["bakery_name"].notna().any()
            else f"Пекарня {bakery_id}"
        )
        # store full list of categories before filtering
        cat_col = self._cat_col(detail)
        bakery_categories = sorted(detail[cat_col].dropna().unique().tolist()) if cat_col else []
        detail = self._filter_cat(detail, category)

        actual_revenue = None
        if "price" in detail.columns:
            actual_revenue = float((detail["sold_qty"].clip(lower=0) * detail["price"].fillna(0)).sum())

        wape = bias = execution_rate = None
        block1_lost_qty = block1_lost_revenue = None
        block1_recognized_lost_qty = block1_recognized_lost_revenue = None
        if "eligible_forecast_summary" in detail.columns:
            b1 = detail[detail["eligible_forecast_summary"].astype(bool)]
            if not b1.empty and "forecast_qty" in b1.columns and "demand_qty" in b1.columns:
                demand_sum = float(b1["demand_qty"].sum())
                if demand_sum > 0:
                    wape = float((b1["forecast_qty"] - b1["demand_qty"]).abs().sum() / demand_sum)
                    bias = float((b1["forecast_qty"] - b1["demand_qty"]).sum() / demand_sum)
                if "sold_qty" in b1.columns:
                    b1_lost = (b1["forecast_qty"] - b1["sold_qty"]).clip(lower=0)
                    block1_lost_qty = float(b1_lost.sum())
                    if "price" in b1.columns:
                        block1_lost_revenue = float((b1_lost * b1["price"].fillna(0)).sum())
                if "produced_qty" in b1.columns:
                    b1e = b1[b1["produced_qty"].notna()]
                    if not b1e.empty:
                        fc = float(b1e["forecast_qty"].sum())
                        execution_rate = float(b1e["produced_qty"].sum()) / fc if fc > 0 else None
        if (
            "eligible_lost_demand" in detail.columns
            and "lost_demand_recognized_qty" in detail.columns
            and "forecast_qty" in detail.columns
            and "sold_qty" in detail.columns
        ):
            ld = detail[
                detail["eligible_lost_demand"].astype(bool)
                & (detail["forecast_qty"] > detail["sold_qty"])
            ]
            if not ld.empty:
                capped = (ld["forecast_qty"] - ld["sold_qty"]).clip(upper=ld["lost_demand_recognized_qty"])
                block1_recognized_lost_qty = float(capped.sum())
                if "price" in ld.columns:
                    block1_recognized_lost_revenue = float((capped * ld["price"].fillna(0)).sum())

        kpi3 = self._kpi_block(detail)
        return {
            "bakery_id": bakery_id,
            "bakery_name": str(name),
            "bakery_categories": bakery_categories,
            "actual_revenue": actual_revenue,
            "plan_qty": kpi3["plan_qty"],
            "produced_qty": kpi3["produced_qty"],
            "sold_qty": kpi3["sold_qty"],
            "execution_rate": kpi3["execution_rate"],
            "sellthrough_rate": kpi3["sellthrough_rate"],
            "recognized_lost_qty": kpi3["recognized_lost_qty"],
            "recognized_lost_revenue": kpi3["recognized_lost_revenue"],
        }

    def get_bakery_week_trend(self, bakery_id: int, category: str | None = None) -> list[dict]:
        """Weekly KPI rows for one bakery, oldest first, with trend vs first week."""
        detail = self._load("detail")
        if detail.empty:
            return []
        detail = detail[detail["bakery_id"] == bakery_id].copy()
        detail = self._filter_cat(detail, category)
        if detail.empty:
            return []
        detail["business_date"] = pd.to_datetime(detail["business_date"], errors="coerce")
        detail["week_start"] = detail["business_date"].dt.to_period("W-MON").apply(
            lambda p: str(p.start_time.date())
        )
        rows = []
        for week_start, grp in detail.groupby("week_start"):
            kpi = self._kpi_block(grp)
            rows.append({
                "week_start": str(week_start),
                "plan_qty": kpi["plan_qty"],
                "produced_qty": kpi["produced_qty"],
                "sold_qty": kpi["sold_qty"],
                "execution_rate": kpi["execution_rate"],
                "sellthrough_rate": kpi["sellthrough_rate"],
                "recognized_lost_qty": kpi["recognized_lost_qty"],
                "exec_delta": None,
            })
        rows.sort(key=lambda r: r["week_start"])
        if rows:
            base_exec = rows[0]["execution_rate"]
            for r in rows:
                r["exec_delta"] = (r["execution_rate"] - base_exec) if (r["execution_rate"] is not None and base_exec is not None) else None
        return rows

    def get_sku_list(self, bakery_id: int, category: str | None = None) -> list[dict]:
        """Per-SKU KPIs for one bakery, aggregated from detail rows."""
        detail = self._load("detail")
        if detail.empty:
            return []
        group = detail[detail["bakery_id"] == bakery_id]
        group = self._filter_cat(group, category)
        if group.empty:
            return []
        rows = []
        for product_id, sku_group in group.groupby("product_id"):
            kpi = self._kpi_block(sku_group)
            pname = (
                sku_group["product_name"].dropna().iloc[0]
                if "product_name" in sku_group.columns
                and sku_group["product_name"].notna().any()
                else None
            )
            rows.append({
                "product_id": int(product_id),
                "product_name": pname,
                "plan_qty": kpi["plan_qty"],
                "produced_qty": kpi["produced_qty"],
                "sold_qty": kpi["sold_qty"],
                "execution_rate": kpi["execution_rate"],
                "sellthrough_rate": kpi["sellthrough_rate"],
                "recognized_lost_qty": kpi["recognized_lost_qty"],
                "recognized_lost_revenue": kpi["recognized_lost_revenue"],
            })
        return sorted(rows, key=lambda r: r["execution_rate"] if r["execution_rate"] is not None else 0)

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

    def get_bakery_week_days(
        self, bakery_id: int, week_start: str, category: str | None = None
    ) -> list[dict]:
        """Per-day aggregated rows for one bakery in one ISO week."""
        _RU_DOW = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
        detail = self._load("detail")
        if detail.empty:
            return []
        detail = detail[detail["bakery_id"] == bakery_id].copy()
        detail = self._filter_cat(detail, category)
        if detail.empty:
            return []
        detail["business_date"] = pd.to_datetime(detail["business_date"], errors="coerce")
        ws = pd.Timestamp(week_start)
        we = ws + pd.Timedelta(days=6)
        week = detail[(detail["business_date"] >= ws) & (detail["business_date"] <= we)]
        if week.empty:
            return []
        rows = []
        for date, grp in week.groupby("business_date"):
            kpi = self._kpi_block(grp)
            rows.append({
                "date": str(date.date()),
                "weekday": _RU_DOW[date.weekday()],
                "n_skus": int(grp["product_id"].nunique()),
                "plan_qty": kpi["plan_qty"],
                "produced_qty": kpi["produced_qty"],
                "sold_qty": kpi["sold_qty"],
                "execution_rate": kpi["execution_rate"],
                "sellthrough_rate": kpi["sellthrough_rate"],
                "recognized_lost_qty": kpi["recognized_lost_qty"],
            })
        rows.sort(key=lambda r: r["date"])
        return rows

    def get_bakery_week_sku_summary(
        self, bakery_id: int, week_start: str, category: str | None = None
    ) -> list[dict]:
        """Per-SKU summary for one bakery in one ISO week, sorted by execution rate asc."""
        detail = self._load("detail")
        if detail.empty:
            return []
        detail = detail[detail["bakery_id"] == bakery_id].copy()
        detail = self._filter_cat(detail, category)
        if detail.empty:
            return []
        detail["business_date"] = pd.to_datetime(detail["business_date"], errors="coerce")
        ws = pd.Timestamp(week_start)
        we = ws + pd.Timedelta(days=6)
        week = detail[(detail["business_date"] >= ws) & (detail["business_date"] <= we)]
        if week.empty:
            return []
        rows = []
        for product_id, grp in week.groupby("product_id"):
            kpi = self._kpi_block(grp)
            name = (
                grp["product_name"].dropna().iloc[0]
                if "product_name" in grp.columns and grp["product_name"].notna().any()
                else None
            )
            rows.append({
                "product_id": int(product_id),
                "product_name": str(name) if name else f"SKU {product_id}",
                "days": int(grp["business_date"].nunique()),
                "plan_qty": kpi["plan_qty"],
                "produced_qty": kpi["produced_qty"],
                "sold_qty": kpi["sold_qty"],
                "execution_rate": kpi["execution_rate"],
                "sellthrough_rate": kpi["sellthrough_rate"],
                "recognized_lost_qty": kpi["recognized_lost_qty"],
            })
        return sorted(rows, key=lambda r: r["execution_rate"] or 0)

    def get_bakery_day_detail(
        self, bakery_id: int, date: str, category: str | None = None
    ) -> list[dict]:
        """Raw per-SKU rows for one bakery on one date — used for Excel export."""
        detail = self._load("detail")
        if detail.empty:
            return []
        detail = detail[detail["bakery_id"] == bakery_id].copy()
        detail = self._filter_cat(detail, category)
        detail["business_date"] = pd.to_datetime(detail["business_date"], errors="coerce").dt.date.astype(str)
        day = detail[detail["business_date"] == date]
        if day.empty:
            return []
        cols = [
            "bakery_name", "product_name", "fact_category_name",
            "forecast_qty", "issued_total_for_sale", "produced_qty",
            "sold_qty", "demand_qty", "price", "revenue",
            "lost_demand_recognized_qty", "eligible_forecast_summary",
            "eligible_lost_demand", "execution_status",
        ]
        present = [c for c in cols if c in day.columns]
        return day[present].to_dict("records")

    def get_week_trend(
        self,
        category: str | None = None,
        regional_director: str | None = None,
        partner: str | None = None,
        bakery_id: int | str | None = None,
        product_id: int | str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        granularity: str = "auto",
    ) -> list[dict]:
        """Period KPI rows: Plan AI / Production / Sales per week or day.

        granularity: 'auto' = day when period < 7 days else week; 'day'; 'week'.
        """
        from datetime import date as _date

        detail = self._load("detail")
        detail = self._apply_filters(
            detail,
            category=category,
            regional_director=regional_director,
            partner=partner,
            bakery_id=bakery_id,
            product_id=product_id,
            date_from=date_from,
            date_to=date_to,
        )
        if detail.empty:
            return []
        detail = detail.copy()
        detail["business_date"] = pd.to_datetime(detail["business_date"], errors="coerce")

        # Determine granularity
        use_days = False
        if granularity == "day":
            use_days = True
        elif granularity == "week":
            use_days = False
        else:  # auto
            if date_from and date_to:
                try:
                    d1 = _date.fromisoformat(date_from)
                    d2 = _date.fromisoformat(date_to)
                    use_days = (d2 - d1).days < 7
                except ValueError:
                    pass

        if use_days:
            detail["period_key"] = detail["business_date"].dt.strftime("%Y-%m-%d")
        else:
            detail["period_key"] = detail["business_date"].dt.to_period("W-MON").apply(
                lambda p: str(p.start_time.date())
            )

        rows = []
        for period_key, grp in detail.groupby("period_key"):
            km = self._kpi_block(grp)
            rows.append({
                "week_start": str(period_key),
                "plan_qty": km["plan_qty"],
                "produced_qty": km["produced_qty"],
                "sold_qty": km["sold_qty"],
                "execution_rate": km["execution_rate"],
                "sellthrough_rate": km["sellthrough_rate"],
                "recognized_lost_qty": km["recognized_lost_qty"],
                "recognized_lost_revenue": km["recognized_lost_revenue"],
                "exec_delta": None,
            })
        rows.sort(key=lambda r: r["week_start"])
        if rows:
            base_exec = rows[0]["execution_rate"]
            for r in rows:
                r["exec_delta"] = (
                    r["execution_rate"] - base_exec
                    if r["execution_rate"] is not None and base_exec is not None
                    else None
                )
        return rows

    def get_sku_summary(
        self,
        category: str | None = None,
        regional_director: str | None = None,
        partner: str | None = None,
        bakery_id: int | str | None = None,
        product_id: int | str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict]:
        """Per-SKU KPIs across all pilot bakeries, sorted by demand coverage ascending."""
        detail = self._load("detail")
        detail = self._apply_filters(
            detail,
            category=category,
            regional_director=regional_director,
            partner=partner,
            bakery_id=bakery_id,
            product_id=product_id,
            date_from=date_from,
            date_to=date_to,
        )
        if detail.empty:
            return []
        rows = []
        for pid, grp in detail.groupby("product_id"):
            km = self._kpi_block(grp)
            name = (
                grp["product_name"].dropna().iloc[0]
                if "product_name" in grp.columns and grp["product_name"].notna().any()
                else None
            )
            actual_rev = (
                float((grp["sold_qty"].clip(lower=0) * grp["price"].fillna(0)).sum())
                if "price" in grp.columns else None
            )
            rows.append({
                "product_id": int(pid),
                "product_name": str(name) if name else f"SKU {pid}",
                "plan_qty": km["plan_qty"],
                "produced_qty": km["produced_qty"],
                "sold_qty": km["sold_qty"],
                "execution_rate": km["execution_rate"],
                "sellthrough_rate": km["sellthrough_rate"],
                "recognized_lost_qty": km["recognized_lost_qty"],
                "recognized_lost_revenue": km["recognized_lost_revenue"],
                "actual_revenue": actual_rev,
            })
        return sorted(rows, key=lambda r: r["execution_rate"] if r["execution_rate"] is not None else 2.0)

    def get_regional_director_summary(
        self,
        category: str | None = None,
        regional_director: str | None = None,
        partner: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict]:
        """KPIs grouped by regional_director, sorted by execution_rate ascending."""
        detail = self._load("detail")
        if detail.empty or "regional_director" not in detail.columns:
            return []
        detail = self._apply_filters(
            detail,
            category=category,
            regional_director=regional_director,
            partner=partner,
            date_from=date_from,
            date_to=date_to,
        )
        if detail.empty:
            return []
        rows = []
        for director, grp in detail.groupby("regional_director", dropna=False):
            if pd.isna(director):
                continue
            km = self._kpi_block(grp)
            actual_rev = (
                float((grp["sold_qty"].clip(lower=0) * grp["price"].fillna(0)).sum())
                if "price" in grp.columns
                else None
            )
            bakery_count = int(grp["bakery_id"].nunique())
            rows.append({
                "regional_director": str(director),
                "bakery_count": bakery_count,
                "plan_qty": km["plan_qty"],
                "produced_qty": km["produced_qty"],
                "sold_qty": km["sold_qty"],
                "execution_rate": km["execution_rate"],
                "sellthrough_rate": km["sellthrough_rate"],
                "recognized_lost_qty": km["recognized_lost_qty"],
                "recognized_lost_revenue": km["recognized_lost_revenue"],
                "actual_revenue": actual_rev,
            })
        return sorted(rows, key=lambda r: r["execution_rate"] if r["execution_rate"] is not None else 2.0)

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
