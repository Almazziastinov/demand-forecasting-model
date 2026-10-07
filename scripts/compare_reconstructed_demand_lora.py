"""Compare demand-LoRA, matched zero-shot, sales-LoRA, and weekday baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_chronos2_fixed_origin import _sha256, _summary  # noqa: E402


KEYS = ["forecast_origin", "date", "lead_days", "bakery_id", "product_id"]


def _validate(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if frame.duplicated([*KEYS, "model"]).any():
        raise ValueError(f"{label} has duplicate model-SKU-days")
    if not np.isfinite(frame[["actual", "prediction"]].to_numpy(dtype=float)).all():
        raise ValueError(f"{label} contains invalid values")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demand-lora", type=Path, required=True)
    parser.add_argument("--sales-lora", type=Path, required=True)
    parser.add_argument("--weekday", type=Path, required=True)
    parser.add_argument("--target-panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    demand = _validate(pd.read_parquet(args.demand_lora), "demand LoRA")
    sales = _validate(pd.read_parquet(args.sales_lora), "sales LoRA")
    weekday = _validate(pd.read_parquet(args.weekday), "weekday")
    if demand["model"].nunique() != 2 or sales["model"].nunique() != 1:
        raise ValueError("Expected two demand-context models and one sales LoRA")
    demand_names = set(demand["model"])
    if not any("zero_shot" in name for name in demand_names):
        raise ValueError("Demand LoRA report is missing matched zero-shot")
    if not any("lora" in name for name in demand_names):
        raise ValueError("Demand LoRA report is missing adapted model")
    reference = demand.loc[
        demand["model"].eq(sorted(demand_names)[0]),
        [*KEYS, "actual", "broad_stockout_signal"],
    ]
    if reference.duplicated(KEYS).any():
        raise ValueError("Reference contains duplicate SKU-days")
    for label, frame in (("sales LoRA", sales), ("weekday", weekday)):
        for _, model in frame.groupby("model"):
            aligned = reference.merge(
                model[[*KEYS, "actual"]],
                on=KEYS,
                how="outer",
                validate="one_to_one",
                indicator=True,
                suffixes=("_reference", "_other"),
            )
            if not aligned["_merge"].eq("both").all():
                raise ValueError(f"{label} does not have exactly the same SKU-days")
            if not np.allclose(
                aligned["actual_reference"], aligned["actual_other"], atol=1e-8
            ):
                raise ValueError(f"{label} has different observed sales")
    sales = sales.copy()
    sales["model"] = "chronos2_small_lora_raw_sales_h14"
    all_models = pd.concat([demand, sales, weekday], ignore_index=True)
    flags = reference[[*KEYS, "broad_stockout_signal"]]
    all_models = all_models.drop(columns="broad_stockout_signal", errors="ignore")
    all_models = all_models.merge(flags, on=KEYS, validate="many_to_one")
    target_panel = pd.read_parquet(
        args.target_panel,
        columns=[
            "date", "bakery_id", "product_id",
            "restoration_uplift_qty", "outlier_reduction_qty",
        ],
    )
    historical_flags = []
    for origin in sorted(reference["forecast_origin"].unique()):
        history = target_panel.loc[
            target_panel["date"].between(
                origin - pd.Timedelta(days=55), origin
            )
        ].copy()
        history["historically_adjusted"] = (
            history["restoration_uplift_qty"].gt(0)
            | history["outlier_reduction_qty"].gt(0)
        )
        grouped = history.groupby(["bakery_id", "product_id"], as_index=False)[
            "historically_adjusted"
        ].any()
        grouped["forecast_origin"] = origin
        historical_flags.append(grouped)
    historical = pd.concat(historical_flags, ignore_index=True)
    all_models = all_models.merge(
        historical,
        on=["forecast_origin", "bakery_id", "product_id"],
        how="left",
        validate="many_to_one",
    )
    if all_models["historically_adjusted"].isna().any():
        raise ValueError("Some evaluation pairs lack historical target evidence")
    leaderboard = _summary(all_models, ["model"])
    clean = _summary(all_models.loc[~all_models["broad_stockout_signal"]], ["model"])
    by_day = _summary(all_models, ["date", "model"])
    by_origin = _summary(all_models, ["forecast_origin", "model"])
    by_history_segment = _summary(all_models, ["historically_adjusted", "model"])
    daily_risk = by_day.groupby("model").agg(
        underforecast_days=("bias_qty", lambda values: int((values < 0).sum())),
        worst_daily_bias_qty=("bias_qty", "min"),
        average_daily_bias_qty=("bias_qty", "mean"),
    ).reset_index()
    args.output_dir.mkdir(parents=True)
    leaderboard.to_csv(args.output_dir / "leaderboard.csv", index=False)
    clean.to_csv(args.output_dir / "clean_subset.csv", index=False)
    by_day.to_csv(args.output_dir / "by_day.csv", index=False)
    by_origin.to_csv(args.output_dir / "by_origin.csv", index=False)
    by_history_segment.to_csv(
        args.output_dir / "by_history_segment.csv", index=False
    )
    daily_risk.to_csv(args.output_dir / "daily_risk.csv", index=False)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "production_write": False,
                "primary_target": "original_observed_sales_qty",
                "common_sku_day_rows_per_model": len(reference),
                "inputs": {
                    "demand_lora": _sha256(args.demand_lora),
                    "sales_lora": _sha256(args.sales_lora),
                    "weekday": _sha256(args.weekday),
                    "target_panel": _sha256(args.target_panel),
                },
                "limitation": (
                    "Clean subset excludes heuristic stockout flags; true latent "
                    "demand on censored days is not observed."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    display_columns = ["model", "rows", "wmape_pct", "bias_pct"]
    print(leaderboard[display_columns].to_string(index=False))
    print(daily_risk.to_string(index=False))


if __name__ == "__main__":
    main()
