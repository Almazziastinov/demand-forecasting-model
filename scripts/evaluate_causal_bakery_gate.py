"""Evaluate a causal bakery-level selector between Direct and scoped P50."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ECON_ROOT = ROOT / "reports/retrained_p50_checkout_price_economics_20260910"
AUGUST = ECON_ROOT / "august_versioned_scope_loss075_no_extra_cap"
SEPTEMBER = ECON_ROOT / "september_versioned_scope_loss075_no_extra_cap"
OUTPUT = ROOT / "reports/causal_bakery_gate_loss075_20260910"

KEYS = ["date", "bakery_id"]
METRICS = ["production", "served", "lost", "gross_profit"]
VARIANTS = ["actual_state", "prod_direct_forecast", "p50_loss_075"]


def load_daily(path: Path, period: str) -> pd.DataFrame:
    rows = pd.read_csv(path / "by_bakery_sku_date.csv")
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    aggregations = {
        f"{metric}_{variant}": "sum"
        for metric in METRICS
        for variant in VARIANTS
    }
    daily = rows.groupby(KEYS, as_index=False).agg(aggregations)
    daily["period"] = period
    daily["candidate_vs_direct"] = (
        daily["gross_profit_p50_loss_075"]
        - daily["gross_profit_prod_direct_forecast"]
    )
    return daily


def apply_gate(
    evaluation: pd.DataFrame,
    history: pd.DataFrame,
    *,
    window_days: int,
    min_days: int,
    mean_threshold: float,
    min_win_rate: float,
    eligible_bakeries: set[int] | None = None,
) -> pd.DataFrame:
    decisions = []
    history = history.sort_values(KEYS)
    for row in evaluation.sort_values(KEYS).itertuples(index=False):
        cutoff = row.date - pd.Timedelta(days=1)
        start = row.date - pd.Timedelta(days=window_days)
        prior = history[
            history["bakery_id"].eq(row.bakery_id)
            & history["date"].between(start, cutoff)
        ]
        evidence_days = len(prior)
        mean_delta = prior["candidate_vs_direct"].mean()
        win_rate = prior["candidate_vs_direct"].gt(0).mean()
        use_candidate = bool(
            (eligible_bakeries is None or row.bakery_id in eligible_bakeries)
            and
            evidence_days >= min_days
            and mean_delta > mean_threshold
            and win_rate >= min_win_rate
        )
        result = {
            "date": row.date,
            "bakery_id": row.bakery_id,
            "period": row.period,
            "use_candidate": use_candidate,
            "evidence_days": evidence_days,
            "prior_mean_delta": mean_delta,
            "prior_win_rate": win_rate,
        }
        selected = "p50_loss_075" if use_candidate else "prod_direct_forecast"
        for metric in METRICS:
            result[f"{metric}_selected"] = getattr(row, f"{metric}_{selected}")
            result[f"{metric}_actual"] = getattr(row, f"{metric}_actual_state")
            result[f"{metric}_direct"] = getattr(
                row, f"{metric}_prod_direct_forecast"
            )
            result[f"{metric}_candidate"] = getattr(
                row, f"{metric}_p50_loss_075"
            )
        decisions.append(result)
    return pd.DataFrame(decisions)


def summarize(rows: pd.DataFrame) -> dict[str, float]:
    actual = rows["gross_profit_actual"].sum()
    selected = rows["gross_profit_selected"].sum()
    direct = rows["gross_profit_direct"].sum()
    candidate = rows["gross_profit_candidate"].sum()
    demand = rows["served_actual"].sum() + rows["lost_actual"].sum()
    return {
        "rows": len(rows),
        "candidate_decisions": int(rows["use_candidate"].sum()),
        "candidate_share_pct": 100 * rows["use_candidate"].mean(),
        "gross_profit_actual": actual,
        "gross_profit_direct": direct,
        "gross_profit_candidate": candidate,
        "gross_profit_gate": selected,
        "gate_vs_actual": selected - actual,
        "gate_vs_actual_pct": 100 * (selected - actual) / actual,
        "gate_vs_direct": selected - direct,
        "gate_vs_candidate": selected - candidate,
        "service_gate_pct": 100 * rows["served_selected"].sum() / demand,
    }


def main() -> None:
    august = load_daily(AUGUST, "august")
    september = load_daily(SEPTEMBER, "september")
    grid_rows = []
    for window_days, min_days, mean_threshold, min_win_rate in product(
        [14, 28, 60], [1, 2, 3, 4], [0.0, 500.0, 1_500.0, 3_000.0], [0.0, 0.5, 0.6]
    ):
        selected = apply_gate(
            august,
            august,
            window_days=window_days,
            min_days=min_days,
            mean_threshold=mean_threshold,
            min_win_rate=min_win_rate,
        )
        summary = summarize(selected)
        grid_rows.append(
            {
                "window_days": window_days,
                "min_days": min_days,
                "mean_threshold": mean_threshold,
                "min_win_rate": min_win_rate,
                **summary,
            }
        )
    grid = pd.DataFrame(grid_rows).sort_values(
        ["gross_profit_gate", "candidate_share_pct"], ascending=[False, True]
    )
    best = grid.iloc[0]
    history = pd.concat([august, september], ignore_index=True)
    established_bakeries = set(august["bakery_id"].astype(int).unique())
    september_gate = apply_gate(
        september,
        history,
        window_days=int(best["window_days"]),
        min_days=int(best["min_days"]),
        mean_threshold=float(best["mean_threshold"]),
        min_win_rate=float(best["min_win_rate"]),
        eligible_bakeries=established_bakeries,
    )
    august_gate = apply_gate(
        august,
        august,
        window_days=int(best["window_days"]),
        min_days=int(best["min_days"]),
        mean_threshold=float(best["mean_threshold"]),
        min_win_rate=float(best["min_win_rate"]),
    )
    summary = pd.DataFrame(
        [
            {"period": "august", **summarize(august_gate)},
            {"period": "september", **summarize(september_gate)},
        ]
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUTPUT / "august_parameter_grid.csv", index=False)
    august_gate.to_csv(OUTPUT / "august_decisions.csv", index=False)
    september_gate.to_csv(OUTPUT / "september_decisions.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    print(best[["window_days", "min_days", "mean_threshold", "min_win_rate"]])
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
