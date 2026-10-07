"""Evaluate bakery/category weekly-dip reconstruction on frozen early folds."""

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
from src.model_tournament.hierarchical_weekly_bridge import (  # noqa: E402
    hierarchical_weekly_bridge,
)
from src.model_tournament.series_diagnostics import stage_diagnostics  # noqa: E402


ORIGINS = pd.to_datetime(
    ["2026-03-30", "2026-04-27", "2026-05-25", "2026-06-21"]
).tolist()
KEY_COLUMNS = ["date", *KEYS]


def _load_categories(path: Path) -> pd.DataFrame:
    metadata = pd.read_parquet(path, columns=["date", "product_id", "category_name"])
    metadata["date"] = pd.to_datetime(metadata["date"]).dt.normalize()
    metadata = metadata.loc[
        metadata["date"].le(ORIGINS[0]), ["product_id", "category_name"]
    ]
    counts = metadata.groupby("product_id")["category_name"].nunique(dropna=True)
    if counts.gt(1).any():
        raise ValueError("Product-category mapping changed within pre-origin history")
    return metadata.dropna().drop_duplicates("product_id").reset_index(drop=True)


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
        "--category-source",
        type=Path,
        default=ROOT / ".codex_tmp/historical_clickhouse_panel/causal_panel.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports/hierarchical_weekly_bridge_early_folds_20261005",
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
        raise ValueError("Panel lacks final outcome date")
    categories = _load_categories(args.category_source)
    history = panel.loc[panel["date"].le(ORIGINS[-1])]
    bridged = hierarchical_weekly_bridge(history, categories)
    details = []
    adjustments = []
    for origin in ORIGINS:
        origin_history = bridged.loc[
            bridged["date"].between(origin - pd.Timedelta(days=55), origin)
        ].copy()
        # A proposal is not known until its right-hand adjacent weekday is observed.
        origin_history.loc[
            origin_history["date"].gt(origin - pd.Timedelta(days=7)),
            ["bakery_only_uplift", "category_guarded_uplift"],
        ] = 0.0
        rows = build_fixed_origin_panel(panel, origin)
        raw = rows["weighted_weekday_sales"]
        variants = {
            "raw_sales": raw,
            "hier_bakery": raw
            + _forecast_uplift(rows, origin_history, "bakery_only_uplift"),
            "hier_category": raw
            + _forecast_uplift(rows, origin_history, "category_guarded_uplift"),
        }
        for model, prediction in variants.items():
            detail = rows[
                ["forecast_origin", "date", "lead_days", *KEYS, "actual"]
            ].copy()
            detail["model"] = model
            detail["prediction"] = prediction.to_numpy(dtype=float)
            details.append(detail)
        adjustments.append(
            {
                "origin": str(origin.date()),
                "bakery_dip_days": int(
                    origin_history.loc[
                        origin_history["date"].le(origin - pd.Timedelta(days=7)),
                        ["date", "bakery_id", "bakery_dip_pass"],
                    ]
                    .drop_duplicates(["date", "bakery_id"])["bakery_dip_pass"]
                    .sum()
                ),
                "bakery_only_sku_days": int(
                    origin_history["bakery_only_uplift"].gt(0).sum()
                ),
                "category_guarded_sku_days": int(
                    origin_history["category_guarded_uplift"].gt(0).sum()
                ),
                "bakery_only_uplift": float(origin_history["bakery_only_uplift"].sum()),
                "category_guarded_uplift": float(
                    origin_history["category_guarded_uplift"].sum()
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
        raise ValueError("Evaluation rows lack historical source-status flag")
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
    diagnostics_input = history[[*KEY_COLUMNS, "observed_sales_qty"]].merge(
        bridged[[*KEY_COLUMNS, "bakery_only_uplift", "category_guarded_uplift"]],
        on=KEY_COLUMNS,
        validate="one_to_one",
    )
    diagnostics_input["demand_after_restoration"] = (
        diagnostics_input["observed_sales_qty"]
        + diagnostics_input["bakery_only_uplift"]
    )
    diagnostics_input["reconstructed_demand_qty"] = (
        diagnostics_input["observed_sales_qty"]
        + diagnostics_input["category_guarded_uplift"]
    )
    diagnostics = stage_diagnostics(diagnostics_input, cutoff=ORIGINS[-1])
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
    raw_row = leaderboard.set_index("model").loc["raw_sales"]
    leaderboard["passes_sales_screen"] = (
        leaderboard["model"].ne("raw_sales")
        & leaderboard["wmape_pct"].le(raw_row["wmape_pct"])
        & leaderboard["heuristic_clean_wmape_pct"].le(
            raw_row["heuristic_clean_wmape_pct"] + 0.1
        )
        & leaderboard["under_qty"].le(raw_row["under_qty"])
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
    bridged.loc[
        bridged["date"].between(ORIGINS[-1] - pd.Timedelta(days=55), ORIGINS[-1])
        & bridged["bakery_dip_pass"]
    ].to_parquet(args.output_dir / "latest_origin_cases.parquet", index=False)
    metadata = {
        "production_write": False,
        "source": (
            "historical deduplicated-fct projection; stg parity verified "
            "only on local pilot overlap"
        ),
        "evaluation_target": "unchanged observed sales, not latent demand",
        "category_mapping_cutoff": str(ORIGINS[0].date()),
        "origins": [str(value.date()) for value in ORIGINS],
        "last_outcome_date": "2026-07-05",
        "two_sided_bridge_available_only_after_next_week": True,
        "known_holiday_buffer_days": 1,
        "input_sha256": _sha256(args.input),
        "category_source_sha256": _sha256(args.category_source),
        "adjustments": adjustments,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(leaderboard.to_string(index=False))
    print(by_origin.to_string(index=False))
    print(diagnostic_summary.to_string(index=False))
    print(json.dumps(adjustments, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
