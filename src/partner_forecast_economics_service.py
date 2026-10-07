"""Read-only partner economics view for the main forecast page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class PartnerForecastEconomicsService:
    def __init__(self, report_dir: str | Path) -> None:
        self.report_dir = Path(report_dir)

    def _load(self, name: str) -> pd.DataFrame:
        path = self.report_dir / f"{name}.csv"
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def get_scope_summary(
        self,
        bakery_ids: list[int] | set[int],
        category_group: str | None = None,
    ) -> dict | None:
        scope = {int(value) for value in bakery_ids}
        if not scope:
            return None
        bakery_week = self._load("bakery_week")
        sku_week = self._load("bakery_sku_week")
        if bakery_week.empty or sku_week.empty:
            return None
        bakery_week = bakery_week[bakery_week["bakery_id"].isin(scope)].copy()
        sku_week = sku_week[sku_week["bakery_id"].isin(scope)].copy()
        if category_group:
            sku_week = sku_week[sku_week["category_group"].eq(category_group)].copy()
        if bakery_week.empty or sku_week.empty:
            return None
        bakery_count = int(bakery_week["bakery_id"].nunique())

        fold_meta = (
            bakery_week.groupby("fold", as_index=False)
            .agg(
                date_min=("date_min", "min"),
                date_max=("date_max", "max"),
                observed_days=("observed_days", "max"),
            )
        )
        bakery_week = (
            sku_week.groupby("fold", as_index=False)
            .agg(
                actual_profit=("actual_profit", "sum"),
                model_profit=("model_profit", "sum"),
                profit_delta=("profit_delta", "sum"),
            )
            .merge(fold_meta, on="fold", how="left", validate="one_to_one")
            .sort_values("fold")
        )
        weeks = []
        for row in bakery_week.itertuples(index=False):
            weeks.append(
                {
                    "fold": str(row.fold),
                    "period": f"{str(row.date_min)[5:]}–{str(row.date_max)[5:]}",
                    "days": int(row.observed_days),
                    "actual_profit": float(row.actual_profit),
                    "model_profit": float(row.model_profit),
                    "profit_delta": float(row.profit_delta),
                }
            )

        sku_total = (
            sku_week.groupby(["product_id", "product_name"], as_index=False)
            .agg(
                actual_volume=("actual_volume", "sum"),
                profit_delta=("profit_delta", "sum"),
            )
            .sort_values(["actual_volume", "profit_delta"], ascending=[False, False])
        )
        week_order = bakery_week["fold"].astype(str).tolist()
        sku_rows = []
        for row in sku_total.itertuples(index=False):
            product_history = (
                sku_week[sku_week["product_id"].eq(row.product_id)]
                .groupby("fold", as_index=False)["profit_delta"]
                .sum()
            )
            product_history["fold"] = product_history["fold"].astype(str)
            history = product_history.set_index("fold")
            sku_rows.append(
                {
                    "product_id": int(row.product_id),
                    "product_name": str(row.product_name),
                    "actual_volume": float(row.actual_volume),
                    "profit_delta": float(row.profit_delta),
                    "weekly_delta": [
                        float(history.at[fold, "profit_delta"])
                        if fold in history.index
                        else 0.0
                        for fold in week_order
                    ],
                }
            )

        actual_profit = float(bakery_week["actual_profit"].sum())
        model_profit = float(bakery_week["model_profit"].sum())
        return {
            "bakery_count": bakery_count,
            "weeks": weeks,
            "observed_days": int(bakery_week["observed_days"].sum()),
            "actual_profit": actual_profit,
            "model_profit": model_profit,
            "profit_delta": model_profit - actual_profit,
            "profit_delta_pct": (
                (model_profit - actual_profit) / actual_profit
                if actual_profit
                else None
            ),
            "top_sku": sku_rows,
        }

    def get_bakery_summary(
        self,
        bakery_id: int | None,
        category_group: str | None = None,
    ) -> dict | None:
        return (
            self.get_scope_summary({bakery_id}, category_group)
            if bakery_id is not None
            else None
        )
