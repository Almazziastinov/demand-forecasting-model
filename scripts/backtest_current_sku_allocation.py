"""Backtest current SKU allocation with fixed production category totals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / ".codex_tmp" / "current_sku_allocation_20260825"
FACT = (
    ROOT
    / "reports"
    / "base_norm_recent_vs_mean7_20260824"
    / "sku_day_comparison.parquet"
)
OUTPUT = ROOT / "reports" / "current_sku_allocation_backtest_20260825"
KEYS = ["date", "bakery_id", "product_id"]
GROUP = ["date", "bakery_id", "category"]


def normalized_share(
    raw: pd.Series, rows: pd.DataFrame, fallback: pd.Series
) -> pd.Series:
    raw = pd.to_numeric(raw, errors="coerce").where(lambda x: x.ge(0))
    raw = raw.fillna(fallback)
    denominator = raw.groupby([rows[key] for key in GROUP]).transform("sum")
    return (raw / denominator.replace(0, np.nan)).fillna(fallback)


def historical_quantities(
    history: pd.DataFrame,
    universe: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    values = (
        history[history["date"].between(start, end)]
        .groupby(["bakery_id", "product_id"])["sold"]
        .sum()
    )
    index = pd.MultiIndex.from_frame(universe[["bakery_id", "product_id"]])
    return pd.Series(
        values.reindex(index, fill_value=0).to_numpy(), index=universe.index
    )


def allocate_day(day: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    date = day["date"].iloc[0]
    category_total = day.groupby(GROUP)["incumbent_sku_forecast"].transform("sum")

    recent = historical_quantities(
        history, day, date - pd.Timedelta(days=7), date - pd.Timedelta(days=1)
    )
    prior = historical_quantities(
        history, day, date - pd.Timedelta(days=14), date - pd.Timedelta(days=8)
    )
    recent_total = recent.groupby([day[key] for key in GROUP]).transform("sum")
    prior_total = prior.groupby([day[key] for key in GROUP]).transform("sum")
    recent_share = recent / recent_total.replace(0, np.nan)
    prior_share = prior / prior_total.replace(0, np.nan)

    broad = historical_quantities(
        history, day, date - pd.Timedelta(days=56), date - pd.Timedelta(days=1)
    )
    broad_total = broad.groupby([day[key] for key in GROUP]).transform("sum")
    broad_share = broad / broad_total.replace(0, np.nan)
    count = day.groupby(GROUP)["product_id"].transform("count")
    uniform = 1.0 / count
    fallback = broad_share.fillna(uniform)

    weekday_dates = [date - pd.Timedelta(days=7 * step) for step in range(1, 5)]
    weekday = (
        history[history["date"].isin(weekday_dates)]
        .groupby(["bakery_id", "product_id"])["sold"]
        .sum()
    )
    index = pd.MultiIndex.from_frame(day[["bakery_id", "product_id"]])
    weekday_qty = pd.Series(
        weekday.reindex(index, fill_value=0).to_numpy(), index=day.index
    )
    weekday_total = weekday_qty.groupby([day[key] for key in GROUP]).transform("sum")
    weekday_share = normalized_share(
        weekday_qty / weekday_total.replace(0, np.nan), day, fallback
    )

    trend_ratio = (recent_share / prior_share.replace(0, np.nan)).clip(0.70, 1.30)
    trend_raw = fallback * trend_ratio.pow(0.5).fillna(1.0)
    trend_share = normalized_share(trend_raw, day, fallback)

    result = day.copy()
    result["same_weekday_forecast"] = weekday_share * category_total
    result["causal_trend_forecast"] = trend_share * category_total
    for alpha in (0.25, 0.50, 0.75):
        result[f"blend_{int(alpha * 100)}"] = (1 - alpha) * result[
            "incumbent_sku_forecast"
        ] + alpha * result["causal_trend_forecast"]
    return result


def metric(rows: pd.DataFrame, forecast: str, target: str) -> dict[str, float]:
    actual = rows[target].sum()
    error = rows[forecast] - rows[target]
    return {
        "wape_pct": float(100 * error.abs().sum() / actual),
        "bias_pct": float(100 * error.sum() / actual),
        "forecast": float(rows[forecast].sum()),
        "actual": float(actual),
    }


def allocation_metric(rows: pd.DataFrame, forecast: str, target: str) -> float:
    forecast_total = rows.groupby(GROUP)[forecast].transform("sum")
    actual_total = rows.groupby(GROUP)[target].transform("sum")
    oracle = (rows[forecast] / forecast_total.replace(0, np.nan) * actual_total).fillna(
        0
    )
    return float(100 * (oracle - rows[target]).abs().sum() / rows[target].sum())


def concentration(rows: pd.DataFrame, forecast: str) -> dict[str, float | int]:
    total = rows.groupby(["date", "bakery_id"])[forecast].transform("sum")
    top = (
        (rows[forecast] / total.replace(0, np.nan))
        .groupby([rows["date"], rows["bakery_id"]])
        .max()
    )
    return {
        "p95": float(top.quantile(0.95)),
        "max": float(top.max()),
        "ge20": int(top.ge(0.20).sum()),
        "ge30": int(top.ge(0.30).sum()),
        "ge40": int(top.ge(0.40).sum()),
    }


def main() -> None:
    snapshot = pd.read_parquet(INPUT / "snapshot.parquet")
    history = pd.read_parquet(INPUT / "sales_history.parquet")
    facts = pd.read_parquet(FACT)[KEYS + ["sold", "strict_demand"]]
    for frame in (snapshot, history, facts):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["product_id"] = pd.to_numeric(frame["product_id"]).astype("int64")

    product_category = snapshot.groupby("product_id")["category"].agg(
        lambda values: values.mode().iloc[0] if not values.mode().empty else "unknown"
    )
    history = history.merge(
        product_category.rename("category"), on="product_id", how="left"
    )

    outputs = []
    for _, day in snapshot.groupby("date", sort=True):
        outputs.append(allocate_day(day.reset_index(drop=True), history))
    predicted = pd.concat(outputs, ignore_index=True).merge(facts, on=KEYS, how="left")
    predicted[["sold", "strict_demand"]] = predicted[["sold", "strict_demand"]].fillna(
        0
    )

    bakery_actual = predicted.groupby(["date", "bakery_id"])["sold"].transform("sum")
    active = predicted[bakery_actual.gt(0)].copy()
    excluded = predicted[bakery_actual.le(0)].copy()
    methods = [
        "incumbent_sku_forecast",
        "same_weekday_forecast",
        "causal_trend_forecast",
        "blend_25",
        "blend_50",
        "blend_75",
    ]
    summary = {
        "scope": {
            "dates": int(active["date"].nunique()),
            "bakeries": int(active["bakery_id"].nunique()),
            "bakery_days": int(active.groupby(["date", "bakery_id"]).ngroups),
            "sku_rows": int(len(active)),
        },
        "dq_excluded": {
            "bakery_days": int(excluded.groupby(["date", "bakery_id"]).ngroups),
            "forecast": float(excluded["incumbent_sku_forecast"].sum()),
        },
        "metrics": {
            target: {
                method: {
                    **metric(active, method, target),
                    "allocation_wape_pct": allocation_metric(active, method, target),
                }
                for method in methods
            }
            for target in ("sold", "strict_demand")
        },
        "concentration": {method: concentration(active, method) for method in methods},
        "conservation": {
            method: float(
                (
                    active.groupby(GROUP)[method].sum()
                    - active.groupby(GROUP)["incumbent_sku_forecast"].sum()
                )
                .abs()
                .max()
            )
            for method in methods[1:]
        },
        "production_write": False,
    }

    incumbent_error = (
        (active["incumbent_sku_forecast"] - active["strict_demand"])
        .abs()
        .groupby([active["date"], active["bakery_id"]])
        .sum()
    )
    summary["stability"] = {}
    for method in methods[1:]:
        challenger_error = (
            (active[method] - active["strict_demand"])
            .abs()
            .groupby([active["date"], active["bakery_id"]])
            .sum()
        )
        summary["stability"][method] = {
            "better_bakery_days": int(challenger_error.lt(incumbent_error).sum()),
            "bakery_days": int(len(challenger_error)),
        }

    date_rows = []
    for date, rows in active.groupby("date", sort=True):
        for method in methods:
            date_rows.append(
                {
                    "date": date,
                    "method": method,
                    **metric(rows, method, "strict_demand"),
                }
            )
    sku_1071 = active[active["product_id"].eq(1071)]
    summary["sku_1071"] = {
        method: metric(sku_1071, method, "strict_demand") for method in methods
    }
    bakery_29 = active[
        active["bakery_id"].eq(29) & active["date"].eq(pd.Timestamp("2026-08-23"))
    ]
    summary["bakery_29_2026_08_23"] = {
        method: metric(bakery_29, method, "strict_demand") for method in methods
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    predicted.to_parquet(OUTPUT / "predictions.parquet", index=False)
    pd.DataFrame(date_rows).to_csv(OUTPUT / "metrics_by_date.csv", index=False)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
