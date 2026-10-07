"""Offline audit of explainability-first, partially regularized SKU targets."""

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
from src.model_tournament.weekly_bridge import weekly_bridge_dips  # noqa: E402


ORIGINS = pd.to_datetime(
    ["2026-03-30", "2026-04-27", "2026-05-25", "2026-06-21"]
).tolist()
KEY_COLUMNS = ["date", *KEYS]


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


def _calendar_incidence(
    bridge: pd.DataFrame, events: set[pd.Timestamp]
) -> pd.DataFrame:
    reference = bridge["bridge_reference"]
    eligible = (
        bridge["previous_week_sales"].ge(10)
        & bridge["next_week_sales"].ge(10)
        & bridge["previous_week_sales"]
        .sub(bridge["next_week_sales"])
        .abs()
        .le(0.2 * reference)
        & bridge["observed_sales_qty"].gt(0)
    )
    sample = bridge.loc[eligible, ["date", "product_id", "isolated_weekly_dip"]].copy()
    sample["calendar_window"] = sample["date"].isin(events)
    incidence = sample.groupby("calendar_window", as_index=False).agg(
        eligible_sku_days=("product_id", "size"),
        low_dip_sku_days=("isolated_weekly_dip", "sum"),
    )
    incidence["low_dip_rate_pct"] = (
        100 * incidence["low_dip_sku_days"] / incidence["eligible_sku_days"]
    )
    return incidence


def _weather_incidence(
    bridge: pd.DataFrame,
    metadata_path: Path,
    weather_path: Path,
    events: set[pd.Timestamp],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Describe historical weather association; do not use it as a future feature."""
    metadata = pd.read_parquet(metadata_path, columns=["date", "bakery_id", "city"])
    metadata["date"] = pd.to_datetime(metadata["date"]).dt.normalize()
    metadata = metadata.loc[metadata["date"].le(ORIGINS[0])]
    city_counts = metadata.groupby("bakery_id")["city"].nunique(dropna=True)
    if city_counts.gt(1).any():
        raise ValueError("Pre-origin bakery-city mapping is ambiguous")
    cities = metadata[["bakery_id", "city"]].dropna().drop_duplicates("bakery_id")
    weather = pd.read_csv(weather_path, usecols=["date", "city", "is_bad_weather"])
    weather["date"] = pd.to_datetime(weather["date"]).dt.normalize()
    if weather.duplicated(["date", "city"]).any():
        raise ValueError("Duplicate historical weather city-day")
    weather = weather.rename(columns={"is_bad_weather": "bad_current"})
    for label, offset in (("bad_previous", 7), ("bad_next", -7)):
        lookup = weather[["date", "city", "bad_current"]].copy()
        lookup["date"] += pd.Timedelta(days=offset)
        weather = weather.merge(
            lookup.rename(columns={"bad_current": label}),
            on=["date", "city"],
            how="left",
            validate="one_to_one",
        )
    reference = bridge["bridge_reference"]
    stable = (
        bridge["previous_week_sales"].ge(10)
        & bridge["next_week_sales"].ge(10)
        & bridge["previous_week_sales"]
        .sub(bridge["next_week_sales"])
        .abs()
        .le(0.2 * reference)
        & bridge["observed_sales_qty"].gt(0)
        & ~bridge["date"].isin(events)
    )
    sample = (
        bridge.loc[stable, [*KEY_COLUMNS, "isolated_weekly_dip"]]
        .merge(cities, on="bakery_id", how="left", validate="many_to_one")
        .merge(weather, on=["date", "city"], how="left", validate="many_to_one")
    )
    complete = sample[["bad_current", "bad_previous", "bad_next"]].notna().all(axis=1)
    sample = sample.loc[complete].copy()
    sample["isolated_bad_weather"] = (
        sample["bad_current"].eq(1)
        & sample["bad_previous"].eq(0)
        & sample["bad_next"].eq(0)
    )
    incidence = sample.groupby("isolated_bad_weather", as_index=False).agg(
        eligible_sku_days=("product_id", "size"),
        low_dip_sku_days=("isolated_weekly_dip", "sum"),
    )
    incidence["low_dip_rate_pct"] = (
        100 * incidence["low_dip_sku_days"] / incidence["eligible_sku_days"]
    )
    bakery = bridge[
        [
            "date",
            "bakery_id",
            "bakery_sales",
            "bakery_previous",
            "bakery_next",
            "bakery_context_ratio",
        ]
    ].drop_duplicates(["date", "bakery_id"])
    bakery_reference = (bakery["bakery_previous"] + bakery["bakery_next"]) / 2
    bakery = (
        bakery.loc[
            bakery["bakery_previous"].ge(100)
            & bakery["bakery_next"].ge(100)
            & bakery["bakery_previous"]
            .sub(bakery["bakery_next"])
            .abs()
            .le(0.2 * bakery_reference)
            & ~bakery["date"].isin(events)
        ]
        .merge(cities, on="bakery_id", how="left", validate="many_to_one")
        .merge(weather, on=["date", "city"], how="left", validate="many_to_one")
    )
    bakery = bakery.loc[
        bakery[["bad_current", "bad_previous", "bad_next"]].notna().all(axis=1)
    ].copy()
    bakery["isolated_bad_weather"] = (
        bakery["bad_current"].eq(1)
        & bakery["bad_previous"].eq(0)
        & bakery["bad_next"].eq(0)
    )
    bakery["bakery_dip"] = bakery["bakery_context_ratio"].le(0.8)
    bakery_incidence = bakery.groupby("isolated_bad_weather", as_index=False).agg(
        eligible_bakery_days=("bakery_id", "size"),
        bakery_dip_days=("bakery_dip", "sum"),
        median_sales_to_weekly_reference=("bakery_context_ratio", "median"),
    )
    bakery_incidence["bakery_dip_rate_pct"] = (
        100
        * bakery_incidence["bakery_dip_days"]
        / bakery_incidence["eligible_bakery_days"]
    )
    summary = {
        "eligible_stable_triplets_before_weather_join": int(stable.sum()),
        "weather_complete_triplets": int(len(sample)),
        "weather_is_historical_observed_not_asof_forecast": True,
    }
    return incidence, bakery_incidence, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "reports/reconstructed_demand_full_20261005/panel.parquet",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / ".codex_tmp/historical_clickhouse_panel/causal_panel.parquet",
    )
    parser.add_argument(
        "--weather",
        type=Path,
        default=ROOT
        / ".codex_tmp/dev_refresh_backup_20260820/bakery_weather_features.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports/unexplained_weekly_dips_early_folds_20261005",
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
        raise ValueError("Missing complete early-fold outcomes")
    history = panel.loc[panel["date"].le(ORIGINS[-1])]
    bridge = weekly_bridge_dips(history)
    events = _event_dates(history["date"].min(), history["date"].max())
    policy = annotate_unexplained_dips(bridge, events, history)
    calendar_incidence = _calendar_incidence(bridge, events)
    weather_incidence, bakery_weather_incidence, weather_summary = _weather_incidence(
        bridge, args.metadata, args.weather, events
    )
    details: list[pd.DataFrame] = []
    adjustments = []
    for origin in ORIGINS:
        origin_history = policy.loc[
            policy["date"].between(origin - pd.Timedelta(days=55), origin)
        ].copy()
        origin_history.loc[
            origin_history["date"].gt(origin - pd.Timedelta(days=7)),
            ["proposed_uplift", "release_uplift", "unverified_uplift"],
        ] = 0.0
        rows = build_fixed_origin_panel(panel, origin)
        uplift = _forecast_uplift(rows, origin_history, "proposed_uplift")
        release_uplift = _forecast_uplift(rows, origin_history, "release_uplift")
        unverified_uplift = _forecast_uplift(rows, origin_history, "unverified_uplift")
        for model, prediction in {
            "raw_sales": rows["weighted_weekday_sales"],
            "release_half": rows["weighted_weekday_sales"] + 0.5 * release_uplift,
            "release_full": rows["weighted_weekday_sales"] + release_uplift,
            "unverified_half": rows["weighted_weekday_sales"] + 0.5 * unverified_uplift,
            "unverified_full": rows["weighted_weekday_sales"] + unverified_uplift,
            "regularized_half": rows["weighted_weekday_sales"] + 0.5 * uplift,
            "regularized_full": rows["weighted_weekday_sales"] + uplift,
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
                "candidate_dips": int(origin_history["isolated_weekly_dip"].sum()),
                "calendar_candidate_dips": int(
                    (
                        origin_history["isolated_weekly_dip"]
                        & origin_history["calendar_candidate"]
                    ).sum()
                ),
                "release_shortfall_candidates": int(
                    (
                        origin_history["release_shortfall_candidate"]
                        & origin_history["proposed_uplift"].gt(0)
                    ).sum()
                ),
                "no_verified_explanation_candidates": int(
                    (
                        origin_history["unexplained_dip"]
                        & origin_history["proposed_uplift"].gt(0)
                    ).sum()
                ),
                "regularized_dips_available": int(
                    origin_history["proposed_uplift"].gt(0).sum()
                ),
                "full_uplift_available": float(origin_history["proposed_uplift"].sum()),
            }
        )
    detail = pd.concat(details, ignore_index=True).merge(
        panel[[*KEY_COLUMNS, "broad_stockout_signal"]],
        on=KEY_COLUMNS,
        how="left",
        validate="many_to_one",
    )
    if detail["broad_stockout_signal"].isna().any():
        raise ValueError("Evaluation contains missing source-status flags")
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
    diagnostic_input = policy[
        [*KEY_COLUMNS, "observed_sales_qty", "target_half", "target_full"]
    ].rename(
        columns={
            "target_half": "demand_after_restoration",
            "target_full": "reconstructed_demand_qty",
        }
    )
    diagnostics = stage_diagnostics(diagnostic_input, cutoff=ORIGINS[-1])
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
    leaderboard.to_csv(args.output_dir / "leaderboard.csv", index=False)
    clean.to_csv(args.output_dir / "heuristic_clean_subset.csv", index=False)
    by_origin.to_csv(args.output_dir / "by_origin.csv", index=False)
    weather_incidence.to_csv(
        args.output_dir / "weather_incidence_descriptive.csv", index=False
    )
    calendar_incidence.to_csv(
        args.output_dir / "calendar_incidence_descriptive.csv", index=False
    )
    bakery_weather_incidence.to_csv(
        args.output_dir / "bakery_weather_incidence_descriptive.csv", index=False
    )
    diagnostic_summary.to_csv(
        args.output_dir / "series_diagnostics_summary.csv", index=False
    )
    policy.loc[policy["isolated_weekly_dip"]].to_parquet(
        args.output_dir / "dip_registry.parquet", index=False
    )
    metadata = {
        "production_write": False,
        "source": (
            "historical deduplicated-fct projection; stg parity only on pilot overlap"
        ),
        "evaluation_target": "unchanged observed sales; latent demand unavailable",
        "origins": [str(origin.date()) for origin in ORIGINS],
        "calendar_source": (
            "project event calendar, +/-1 day; occurrence is not causal proof"
        ),
        "weather": weather_summary,
        "weather_used_as_forecast_feature": False,
        "input_sha256": _sha256(args.input),
        "metadata_sha256": _sha256(args.metadata),
        "weather_sha256": _sha256(args.weather),
        "adjustments": adjustments,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(leaderboard.to_string(index=False))
    print(
        by_origin[["forecast_origin", "model", "wmape_pct", "bias_pct"]].to_string(
            index=False
        )
    )
    print(weather_incidence.to_string(index=False))
    print(calendar_incidence.to_string(index=False))
    print(bakery_weather_incidence.to_string(index=False))
    print(diagnostic_summary.to_string(index=False))
    print(json.dumps(adjustments, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
