"""Fixed-origin screen of past-only volatility-gated weekly regularization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_chronos2_fixed_origin import _sha256, _summary  # noqa: E402
from src.experiments_v2.bakery_day_forecast import build_event_calendar  # noqa: E402
from src.model_tournament.fixed_origin import (  # noqa: E402
    KEYS,
    _same_weekday_formula,
    build_fixed_origin_panel,
    validate_flows,
)
from src.model_tournament.series_diagnostics import stage_diagnostics  # noqa: E402
from src.model_tournament.unexplained_weekly_dips import (  # noqa: E402
    annotate_unexplained_dips,
)
from src.model_tournament.volatility_guard import (  # noqa: E402
    attach_supply_context,
    attach_volatility_guard,
)
from src.model_tournament.weekly_bridge import weekly_bridge_dips  # noqa: E402


EARLY_ORIGINS = pd.to_datetime(
    ["2026-03-30", "2026-04-27", "2026-05-25", "2026-06-21"]
).tolist()
LATER_ORIGINS = pd.to_datetime(
    ["2026-07-06", "2026-07-20", "2026-08-03", "2026-08-17"]
).tolist()
KEY_COLUMNS = ["date", *KEYS]
UPLIFT_COLUMNS = [
    "proposed_uplift",
    "guarded_total_uplift",
    "guarded_supply_uplift",
    "guarded_no_supply_signal_uplift",
]


def _event_dates(start: pd.Timestamp, end: pd.Timestamp) -> set[pd.Timestamp]:
    calendar = build_event_calendar(
        start - pd.Timedelta(days=1), end + pd.Timedelta(days=1)
    )
    return {
        day + pd.Timedelta(days=offset)
        for day in calendar["date"]
        for offset in (-1, 0, 1)
    }


def _forecast_uplift(
    rows: pd.DataFrame, history: pd.DataFrame, column: str
) -> pd.Series:
    source = history[[*KEY_COLUMNS, column]].rename(
        columns={column: "observed_sales_qty"}
    )
    return pd.Series(_same_weekday_formula(rows, source), index=rows.index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "reports/reconstructed_demand_full_20261005/panel.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports/volatility_guarded_dips_20261005",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    panel = validate_flows(
        pd.read_parquet(
            args.input,
            columns=[
                *KEY_COLUMNS,
                "observed_sales_qty",
                "release_qty",
                "incoming_move_qty",
                "outgoing_move_qty",
                "broad_stockout_signal",
            ],
        )
    )
    if panel["date"].max() < pd.Timestamp("2026-08-31"):
        raise ValueError("Panel lacks full later-fold outcomes")
    origins = [*EARLY_ORIGINS, *LATER_ORIGINS]
    history = panel.loc[panel["date"].le(origins[-1])]
    bridge = weekly_bridge_dips(history)
    policy = annotate_unexplained_dips(
        bridge,
        _event_dates(history["date"].min(), history["date"].max()),
        history,
    )
    guarded = attach_supply_context(attach_volatility_guard(policy), history)
    if not (guarded["guarded_total_uplift"] <= guarded["proposed_uplift"] + 1e-9).all():
        raise AssertionError("Guard increased a proposed target adjustment")
    details: list[pd.DataFrame] = []
    adjustments = []
    for origin in origins:
        origin_history = guarded.loc[
            guarded["date"].between(origin - pd.Timedelta(days=55), origin)
        ].copy()
        available = origin_history["date"].le(origin - pd.Timedelta(days=7))
        origin_history.loc[~available, UPLIFT_COLUMNS] = 0.0
        rows = build_fixed_origin_panel(panel, origin)
        baseline = rows["weighted_weekday_sales"]
        components = {
            column: _forecast_uplift(rows, origin_history, column)
            for column in UPLIFT_COLUMNS
        }
        predictions = {
            "raw_sales": baseline,
            "naive_half": baseline + 0.5 * components["proposed_uplift"],
            "guarded_half": baseline + 0.5 * components["guarded_total_uplift"],
            "guarded_full": baseline + components["guarded_total_uplift"],
            "guarded_supply_half": baseline + 0.5 * components["guarded_supply_uplift"],
            "guarded_no_supply_signal_half": baseline
            + 0.5 * components["guarded_no_supply_signal_uplift"],
        }
        for model, prediction in predictions.items():
            detail = rows[
                ["forecast_origin", "date", "lead_days", *KEYS, "actual"]
            ].copy()
            detail["phase"] = "early" if origin in EARLY_ORIGINS else "later"
            detail["model"] = model
            detail["prediction"] = prediction.to_numpy(dtype=float)
            details.append(detail)
        adjustments.append(
            {
                "origin": str(origin.date()),
                "phase": "early" if origin in EARLY_ORIGINS else "later",
                "weekly_dip_candidates": int(
                    origin_history.loc[available, "regularization_candidate"].sum()
                ),
                "volatility_passes": int(
                    origin_history.loc[available, "volatility_guard_pass"].sum()
                ),
                "supply_passes": int(
                    origin_history["guarded_supply_uplift"].gt(0).sum()
                ),
                "no_supply_signal_passes": int(
                    origin_history["guarded_no_supply_signal_uplift"].gt(0).sum()
                ),
                "naive_full_uplift": float(origin_history["proposed_uplift"].sum()),
                "guarded_full_uplift": float(
                    origin_history["guarded_total_uplift"].sum()
                ),
            }
        )
    detail = pd.concat(details, ignore_index=True).merge(
        panel[[*KEY_COLUMNS, "broad_stockout_signal"]],
        on=KEY_COLUMNS,
        how="left",
        validate="many_to_one",
    )
    if detail["broad_stockout_signal"].isna().any():
        raise ValueError("Missing outcome SKU-days; do not impute them as zero")
    leaderboard = _summary(detail, ["phase", "model"])
    clean = _summary(detail.loc[~detail["broad_stockout_signal"]], ["phase", "model"])
    leaderboard = leaderboard.merge(
        clean[["phase", "model", "wmape_pct"]].rename(
            columns={"wmape_pct": "heuristic_clean_wmape_pct"}
        ),
        on=["phase", "model"],
        validate="one_to_one",
    )
    by_origin = _summary(detail, ["forecast_origin", "phase", "model"])
    diagnostic_input = guarded[
        [*KEY_COLUMNS, "observed_sales_qty", "proposed_uplift", "guarded_total_uplift"]
    ].copy()
    diagnostic_input["demand_after_restoration"] = (
        diagnostic_input["observed_sales_qty"]
        + 0.5 * diagnostic_input["proposed_uplift"]
    )
    diagnostic_input["reconstructed_demand_qty"] = (
        diagnostic_input["observed_sales_qty"]
        + 0.5 * diagnostic_input["guarded_total_uplift"]
    )
    diagnostics = stage_diagnostics(diagnostic_input, cutoff=origins[-1])
    diagnostic_summary = (
        diagnostics.groupby("stage")
        .agg(
            series=("bakery_id", "size"),
            median_cv=("cv", "median"),
            median_acf7=("acf7", "median"),
            median_pacf7=("pacf7", "median"),
            median_weekly_strength=("weekly_seasonality_strength", "median"),
            median_prior_weekday_r2=("prior_weekday_r2", "median"),
        )
        .reset_index()
    )
    args.output_dir.mkdir(parents=True)
    detail.to_parquet(args.output_dir / "detail.parquet", index=False)
    leaderboard.to_csv(args.output_dir / "leaderboard_by_phase.csv", index=False)
    by_origin.to_csv(args.output_dir / "by_origin.csv", index=False)
    diagnostic_summary.to_csv(
        args.output_dir / "series_diagnostics_summary.csv", index=False
    )
    guarded.loc[guarded["regularization_candidate"]].to_parquet(
        args.output_dir / "candidate_registry.parquet", index=False
    )
    metadata = {
        "production_write": False,
        "target": "observed sales for risk scoring; latent demand unknown",
        "rule_frozen_before_later_fold_scoring": True,
        "prior_weeks": 8,
        "minimum_prior_weeks": 6,
        "z_threshold": 2.5,
        "minimum_reference": 30.0,
        "regime_tolerance": 0.25,
        "supply_proxy": "release + incoming move - outgoing move; not shelf stock",
        "origins": [str(origin.date()) for origin in origins],
        "input_sha256": _sha256(args.input),
        "adjustments": adjustments,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        leaderboard[
            [
                "phase",
                "model",
                "rows",
                "wmape_pct",
                "heuristic_clean_wmape_pct",
                "bias_pct",
                "under_qty",
                "over_qty",
            ]
        ].to_string(index=False)
    )
    print(
        by_origin[["forecast_origin", "model", "wmape_pct", "bias_pct"]].to_string(
            index=False
        )
    )
    print(diagnostic_summary.to_string(index=False))
    print(json.dumps(adjustments, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
