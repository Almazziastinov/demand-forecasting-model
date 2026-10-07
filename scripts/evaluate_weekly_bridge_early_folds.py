"""Offline screen of two-sided same-weekday dip interpolation on early folds.

Every target is rebuilt from history ending at its forecast origin. Future
outcome sales are used only for scoring, never to smooth earlier histories.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_chronos2_fixed_origin import _sha256, _summary  # noqa: E402
from src.model_tournament.fixed_origin import (  # noqa: E402
    KEYS,
    _same_weekday_formula,
    build_fixed_origin_panel,
    validate_flows,
)
from src.model_tournament.series_diagnostics import stage_diagnostics  # noqa: E402
from src.model_tournament.weekly_bridge import weekly_bridge_dips  # noqa: E402


ORIGINS = pd.to_datetime(
    ["2026-03-30", "2026-04-27", "2026-05-25", "2026-06-21"]
).tolist()
KEY_COLUMNS = ["date", *KEYS]


def _uplift_forecast(
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
        default=ROOT / "reports/weekly_bridge_early_folds_20261005",
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
                "broad_stockout_signal",
            ],
        )
    )
    if panel["date"].max() < pd.Timestamp("2026-07-05"):
        raise ValueError("Panel does not contain all early-fold outcomes")
    details: list[pd.DataFrame] = []
    adjustments: list[dict[str, object]] = []
    latest_bridge: pd.DataFrame | None = None
    for origin in ORIGINS:
        source_history = panel[panel["date"].le(origin)]
        bridge = weekly_bridge_dips(source_history)
        if origin == ORIGINS[-1]:
            latest_bridge = bridge
        origin_history = bridge[
            bridge["date"].between(origin - pd.Timedelta(days=55), origin)
        ].copy()
        origin_history["pure_high_volume_uplift"] = origin_history[
            "pure_weekly_bridge_uplift"
        ].where(origin_history["bridge_reference"].ge(50), 0.0)
        origin_history["context_high_volume_uplift"] = origin_history[
            "context_weekly_bridge_uplift"
        ].where(origin_history["bridge_reference"].ge(50), 0.0)
        rows = build_fixed_origin_panel(panel, origin)
        raw = rows["weighted_weekday_sales"]
        pure = _uplift_forecast(rows, origin_history, "pure_weekly_bridge_uplift")
        contextual = _uplift_forecast(
            rows, origin_history, "context_weekly_bridge_uplift"
        )
        pure_high_volume = _uplift_forecast(
            rows, origin_history, "pure_high_volume_uplift"
        )
        context_high_volume = _uplift_forecast(
            rows, origin_history, "context_high_volume_uplift"
        )
        for model, prediction in {
            "raw_sales": raw,
            "weekly_bridge_full": raw + pure,
            "weekly_bridge_context": raw + contextual,
            "weekly_bridge_high_volume": raw + pure_high_volume,
            "weekly_bridge_context_high_volume": raw + context_high_volume,
        }.items():
            detail = rows[
                ["forecast_origin", "date", "lead_days", *KEYS, "actual"]
            ].copy()
            detail["model"] = model
            detail["prediction"] = prediction.to_numpy(dtype=float)
            details.append(detail)
        adjustments.append(
            {
                "origin": str(origin.date()),
                "history_rows": int(len(origin_history)),
                "pure_events": int(origin_history["isolated_weekly_dip"].sum()),
                "context_events": int(origin_history["context_guard_pass"].sum()),
                "pure_uplift": float(origin_history["pure_weekly_bridge_uplift"].sum()),
                "context_uplift": float(
                    origin_history["context_weekly_bridge_uplift"].sum()
                ),
                "pure_high_volume_events": int(
                    origin_history["pure_high_volume_uplift"].gt(0).sum()
                ),
                "context_high_volume_events": int(
                    origin_history["context_high_volume_uplift"].gt(0).sum()
                ),
            }
        )
    detail = pd.concat(details, ignore_index=True)
    detail = detail.merge(
        panel[[*KEY_COLUMNS, "broad_stockout_signal"]],
        on=KEY_COLUMNS,
        how="left",
        validate="many_to_one",
    )
    if detail["broad_stockout_signal"].isna().any():
        raise ValueError("Evaluation rows lack a historical source-status flag")
    leaderboard = _summary(detail, ["model"])
    clean = _summary(detail.loc[~detail["broad_stockout_signal"]], ["model"])
    leaderboard = leaderboard.merge(
        clean[["model", "wmape_pct"]].rename(
            columns={"wmape_pct": "heuristic_clean_wmape_pct"}
        ),
        on="model",
        validate="one_to_one",
    )
    by_origin = _summary(detail, ["forecast_origin", "model"])
    assert latest_bridge is not None
    diagnostic = latest_bridge[[*KEY_COLUMNS, "observed_sales_qty"]].copy()
    diagnostic["demand_after_restoration"] = (
        latest_bridge["observed_sales_qty"] + latest_bridge["pure_weekly_bridge_uplift"]
    )
    diagnostic["reconstructed_demand_qty"] = (
        latest_bridge["observed_sales_qty"]
        + latest_bridge["context_weekly_bridge_uplift"]
    )
    diagnostics = stage_diagnostics(diagnostic, cutoff=ORIGINS[-1])
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
    raw = leaderboard.set_index("model").loc["raw_sales"]
    leaderboard["passes_sales_screen"] = (
        leaderboard["model"].ne("raw_sales")
        & leaderboard["wmape_pct"].le(raw["wmape_pct"])
        & leaderboard["heuristic_clean_wmape_pct"].le(
            raw["heuristic_clean_wmape_pct"] + 0.1
        )
        & leaderboard["under_qty"].le(raw["under_qty"])
    )
    args.output_dir.mkdir(parents=True)
    detail.to_parquet(args.output_dir / "detail.parquet", index=False)
    leaderboard.to_csv(args.output_dir / "leaderboard.csv", index=False)
    clean.to_csv(args.output_dir / "heuristic_clean_subset.csv", index=False)
    by_origin.to_csv(args.output_dir / "by_origin.csv", index=False)
    diagnostics.to_parquet(args.output_dir / "series_diagnostics.parquet", index=False)
    diagnostic_summary.to_csv(
        args.output_dir / "series_diagnostics_summary.csv", index=False
    )
    latest_bridge.loc[
        latest_bridge["date"].between(ORIGINS[-1] - pd.Timedelta(days=55), ORIGINS[-1])
        & latest_bridge["isolated_weekly_dip"]
    ].to_parquet(args.output_dir / "latest_origin_cases.parquet", index=False)
    metadata = {
        "production_write": False,
        "source": (
            "historical deduplicated-fct projection; stg parity verified "
            "only on local pilot overlap"
        ),
        "evaluation_target": "unchanged observed sales, not latent demand",
        "origins": [str(value.date()) for value in ORIGINS],
        "last_outcome_date": "2026-07-05",
        "two_sided_bridge_available_only_after_next_week": True,
        "bridge_rule": (
            "exact adjacent weekdays; neighbors within 20%; both >=10; "
            "current >0 and <=75% of mean neighbors"
        ),
        "context_rule": (
            "bakery and leave-one-out peer-product daily ratios >=85% "
            "of adjacent-week mean"
        ),
        "high_volume_reference_minimum": 50,
        "input_sha256": _sha256(args.input),
        "adjustments": adjustments,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(leaderboard.to_string(index=False))
    print(diagnostic_summary.to_string(index=False))
    print(json.dumps(adjustments, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
