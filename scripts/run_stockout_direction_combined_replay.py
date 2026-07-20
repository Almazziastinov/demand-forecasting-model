"""Combine allocation and demand-restoration candidates in an offline replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipelines.forecast_publish.load_forecast_run import create_client  # noqa: E402

DEFAULT_CASES = ROOT / "reports/stockout_mechanism_classification/classified_cases.csv"
DEFAULT_ADJUSTMENTS = ROOT / "reports/demand_adjusted_stockout_history/case_adjustments.csv"
DEFAULT_DYNAMIC = ROOT / "reports/dynamic_sku_allocation_experiment/best_scenario_rows.csv"
DEFAULT_OUTPUT = ROOT / "reports/stockout_direction_combined_replay"


def load_current_allocation_shares(client, bakery_ids: list[int]) -> pd.DataFrame:
    run_id = client.query_df(
        """
        select run_id
        from forecast_runs_embedded
        where status = 'active'
        order by generated_at desc
        limit 1
        """
    ).iloc[0, 0]
    return client.query_df(
        """
        with daily as (
            select forecast_date, bakery_id, sum(forecast_qty) as total_qty
            from sku_forecast_day_snapshots final
            where source_run_id = %(run_id)s
            group by forecast_date, bakery_id
        )
        select
            s.bakery_id,
            s.product_id,
            toDayOfWeek(s.forecast_date) as replay_dow,
            sum(s.forecast_qty) / sum(d.total_qty) as current_share
        from sku_forecast_day_snapshots as s final
        inner join daily as d using (forecast_date, bakery_id)
        where s.source_run_id = %(run_id)s
          and s.bakery_id in %(bakery_ids)s
        group by s.bakery_id, s.product_id, replay_dow
        """,
        parameters={"run_id": run_id, "bakery_ids": bakery_ids},
    )


def build_replay(
    cases: pd.DataFrame,
    adjustments: pd.DataFrame,
    dynamic: pd.DataFrame,
    current_shares: pd.DataFrame,
) -> pd.DataFrame:
    work = cases.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    work["replay_dow"] = work["date"].dt.dayofweek + 1
    work = work.merge(
        current_shares,
        on=["bakery_id", "product_id", "replay_dow"],
        how="left",
        validate="many_to_one",
    )
    work["current_profile_forecast"] = (
        work["current_share"] * work["bakery_forecast_qty"]
    )
    adjustment_columns = ["date", "bakery_id", "product_id", "imputed_demand"]
    adjustment = adjustments[adjustment_columns].copy()
    adjustment["date"] = pd.to_datetime(adjustment["date"]).dt.normalize()
    work = work.merge(
        adjustment,
        on=["date", "bakery_id", "product_id"],
        how="left",
        validate="one_to_one",
    )
    work["imputed_demand"] = work["imputed_demand"].fillna(0.0)
    dynamic_columns = ["date", "bakery_id", "product_id", "adjusted_forecast_qty"]
    dynamic_work = dynamic[dynamic_columns].copy()
    dynamic_work["date"] = pd.to_datetime(dynamic_work["date"]).dt.normalize()
    work = work.merge(
        dynamic_work,
        on=["date", "bakery_id", "product_id"],
        how="left",
        validate="one_to_one",
    ).rename(columns={"adjusted_forecast_qty": "dynamic_forecast"})
    work["dynamic_forecast"] = work["dynamic_forecast"].fillna(work["forecast_qty"])
    work["current_profile_forecast"] = work["current_profile_forecast"].fillna(
        work["forecast_qty"]
    )
    work["demand_only_forecast"] = work["forecast_qty"] + work["imputed_demand"]
    work["current_profile_plus_demand"] = (
        work["current_profile_forecast"] + work["imputed_demand"]
    )
    work["dynamic_plus_demand"] = work["dynamic_forecast"] + work["imputed_demand"]
    return work


def summarize_scenario(
    frame: pd.DataFrame, *, scenario: str, forecast_column: str
) -> dict[str, object]:
    shortfall = (frame["daily_sold"] - frame[forecast_column]).clip(lower=0.0)
    baseline = (frame["daily_sold"] - frame["forecast_qty"]).clip(lower=0.0)
    return {
        "scenario": scenario,
        "forecast_column": forecast_column,
        "cases": int(len(frame)),
        "shortfall_qty": float(shortfall.sum()),
        "cases_fixed": int(shortfall.le(0.5).sum()),
        "cases_improved": int(shortfall.lt(baseline - 0.01).sum()),
        "cases_worsened": int(shortfall.gt(baseline + 0.01).sum()),
        "added_bakery_demand": float(
            frame["imputed_demand"].sum() if "demand" in scenario else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--adjustments", default=str(DEFAULT_ADJUSTMENTS))
    parser.add_argument("--dynamic", default=str(DEFAULT_DYNAMIC))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    cases = pd.read_csv(args.cases, encoding="utf-8-sig")
    adjustments = pd.read_csv(args.adjustments, encoding="utf-8-sig")
    dynamic = pd.read_csv(args.dynamic, encoding="utf-8-sig")
    client = create_client(args.env_file)
    shares = load_current_allocation_shares(
        client, sorted(cases["bakery_id"].unique().tolist())
    )
    replay = build_replay(cases, adjustments, dynamic, shares)
    scenario_columns = {
        "historical_baseline": "forecast_qty",
        "demand_only": "demand_only_forecast",
        "current_profile_diagnostic": "current_profile_forecast",
        "current_profile_plus_demand_diagnostic": "current_profile_plus_demand",
        "dynamic_walk_forward": "dynamic_forecast",
        "dynamic_walk_forward_plus_demand": "dynamic_plus_demand",
    }
    rows = []
    for scenario, column in scenario_columns.items():
        rows.append(summarize_scenario(replay, scenario=scenario, forecast_column=column))
        for case_type in ["allocation", "demand_loss", "uncertain"]:
            subset = replay[replay["robust_case_type"].eq(case_type)]
            row = summarize_scenario(
                subset,
                scenario=f"{scenario}__{case_type}",
                forecast_column=column,
            )
            row["segment"] = case_type
            rows.append(row)
    comparison = pd.DataFrame(rows)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    replay.to_csv(output / "case_replay.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output / "scenario_comparison.csv", index=False)
    top_level = comparison[comparison["scenario"].isin(scenario_columns)]
    summary = {
        "scenarios": top_level.to_dict(orient="records"),
        "recommended_shadow_components": ["robust_demand_loss_preprocessing"],
        "rejected_for_shadow": ["dynamic_walk_forward_allocation"],
        "diagnostic_only": ["current_profile_replay_due_to_profile_lookahead"],
        "production_write": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
