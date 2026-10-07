"""Audit how much of the frozen LoRA holdout has independent demand evidence.

All outputs are research-only. A plausible unconstrained sale is still not an
observed latent-demand label; daily flows lack intraday shelf availability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
KEYS = ["forecast_origin", "date", "lead_days", "bakery_id", "product_id"]
DAY_KEYS = ["date", "bakery_id", "product_id"]
RAW_LORA = (
    ROOT
    / "reports/model_tournament_chronos2_lora_summer_1000_eval_20261002/detail.parquet"
)
DEMAND_LORA = ROOT / "reports/chronos2_demand_lora_full_1000_20261005/detail.parquet"
DEMAND_PANEL = ROOT / "reports/reconstructed_demand_full_20261005/panel.parquet"
SPARSE_SOURCE = ROOT / ".codex_tmp/historical_clickhouse_panel"
STG_SOURCE = ROOT / "data/raw/pilot_stg_check_lines_2026-04-30_2026-07-19.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sparse_keys() -> pd.DataFrame:
    parts = []
    for month in ("202607", "202608"):
        path = SPARSE_SOURCE / f"{month}.csv.gz"
        part = pd.read_csv(path, usecols=DAY_KEYS, parse_dates=["date"])
        parts.append(part.drop_duplicates(DAY_KEYS))
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(DAY_KEYS).any():
        raise ValueError("Duplicate sparse source keys")
    return result.assign(sparse_source_row=True)


def load_stg_daily() -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    bakery_days = []
    columns = ["check_date", "cash_event_type", "quantity", "bakery_id", "product_id"]
    for chunk in pd.read_csv(STG_SOURCE, usecols=columns, chunksize=250_000):
        chunk = chunk.loc[
            chunk["cash_event_type"].eq("Продажа")
            & chunk["check_date"].between("2026-07-07", "2026-07-19"),
            ["check_date", "bakery_id", "product_id", "quantity"],
        ].copy()
        if chunk.empty:
            continue
        chunk["date"] = pd.to_datetime(chunk.pop("check_date"))
        chunk["quantity"] = pd.to_numeric(chunk["quantity"], errors="raise")
        if chunk["quantity"].lt(0).any():
            raise ValueError("Negative stg sale quantity")
        bakery_days.append(chunk[["date", "bakery_id"]].drop_duplicates())
        parts.append(chunk.groupby(DAY_KEYS, as_index=False)["quantity"].sum())
    if not parts:
        raise ValueError("No stg sales in the holdout period")
    daily = (
        pd.concat(parts, ignore_index=True)
        .groupby(DAY_KEYS, as_index=False)["quantity"]
        .sum()
    )
    daily = daily.rename(columns={"quantity": "stg_sales_qty"})
    coverage = (
        pd.concat(bakery_days, ignore_index=True)
        .drop_duplicates()
        .assign(stg_bakery_day=True)
    )
    return daily, coverage


def score(group: pd.DataFrame, prediction_col: str) -> dict[str, float]:
    actual = group["actual"].to_numpy(float)
    prediction = group[prediction_col].to_numpy(float)
    denominator = actual.sum()
    return {
        "wmape_pct": float(100 * np.abs(prediction - actual).sum() / denominator),
        "bias_pct": float(100 * (prediction - actual).sum() / denominator),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    raw = pd.read_parquet(RAW_LORA, columns=[*KEYS, "actual", "prediction"])
    demand = pd.read_parquet(DEMAND_LORA)
    demand = demand.loc[
        demand["model"].eq("chronos2_small_lora_reconstructed_demand_h14"),
        [*KEYS, "prediction"],
    ].rename(columns={"prediction": "demand_lora_prediction"})
    panel = pd.read_parquet(DEMAND_PANEL)
    for frame, keys in ((raw, KEYS), (demand, KEYS), (panel, DAY_KEYS)):
        if frame.duplicated(keys).any():
            raise ValueError(f"Duplicate keys: {keys}")
    work = raw.merge(demand, on=KEYS, how="left", validate="one_to_one")
    work = work.merge(panel, on=DAY_KEYS, how="left", validate="one_to_one")
    if work[["demand_lora_prediction", "observed_sales_qty"]].isna().any().any():
        raise ValueError("Missing frozen prediction or demand-panel row")
    if not np.allclose(work["actual"], work["observed_sales_qty"], atol=1e-8):
        raise ValueError("Frozen sales truth disagrees with demand panel")

    work = work.merge(
        load_sparse_keys(), on=DAY_KEYS, how="left", validate="one_to_one"
    )
    work["sparse_source_row"] = work["sparse_source_row"].eq(True)
    stg, coverage = load_stg_daily()
    work = work.merge(
        coverage, on=["date", "bakery_id"], how="left", validate="many_to_one"
    )
    work = work.merge(stg, on=DAY_KEYS, how="left", validate="one_to_one")
    work["stg_bakery_day"] = work["stg_bakery_day"].eq(True)
    work["stg_sales_qty"] = work["stg_sales_qty"].fillna(0.0)
    work["stg_reconciled_1unit"] = work["stg_bakery_day"] & (
        work["actual"] - work["stg_sales_qty"]
    ).abs().le(1.0)
    work["stg_reconciled_exact"] = work["stg_bakery_day"] & np.isclose(
        work["actual"], work["stg_sales_qty"], atol=1e-6
    )

    available = (
        work["release_qty"] + work["incoming_move_qty"] - work["outgoing_move_qty"]
    ).clip(lower=0)
    residual = available - work["actual"] - work["written_off_qty"]
    close_gap = work["bakery_last_sale_hour"] - work["last_sale_hour"]
    positive = work["actual"].gt(0)
    late_sale = positive & close_gap.ge(0) & close_gap.le(1)
    surplus = residual.ge(np.maximum(2.0, 0.1 * work["actual"]))
    work["late_sale_candidate"] = work["sparse_source_row"] & late_sale
    work["late_sale_with_surplus_candidate"] = (
        work["late_sale_candidate"]
        & surplus
        & work["written_off_qty"].eq(0)
        & ~work["broad_stockout_signal"]
    )
    work["stg_exact_surplus_candidate"] = (
        work["late_sale_with_surplus_candidate"] & work["stg_reconciled_exact"]
    )
    work["zero_sales_positive_flow"] = work["actual"].eq(0) & available.gt(0)

    masks = {
        "all_holdout": pd.Series(True, index=work.index),
        "sparse_fact_present": work["sparse_source_row"],
        "stg_bakery_day_covered": work["stg_bakery_day"],
        "stg_sales_reconciled_exact": work["stg_reconciled_exact"],
        "stg_sales_reconciled_within_1_unit": work["stg_reconciled_1unit"],
        "positive_sales": positive,
        "zero_sales_positive_daily_flow_ambiguous": work["zero_sales_positive_flow"],
        "broad_stockout_heuristic_review_only": work["broad_stockout_signal"],
        "proxy_restored_not_ground_truth": work["is_restored"],
        "late_sale_candidate": work["late_sale_candidate"],
        "late_sale_with_surplus_candidate": work["late_sale_with_surplus_candidate"],
        "stg_exact_surplus_candidate": work["stg_exact_surplus_candidate"],
    }
    summary_rows = []
    score_rows = []
    for name, mask in masks.items():
        group = work.loc[mask]
        summary_rows.append(
            {
                "tier": name,
                "sku_days": len(group),
                "share_of_holdout_pct": 100 * len(group) / len(work),
                "observed_sales_qty": float(group["actual"].sum()),
                "share_of_holdout_sales_pct": 100
                * group["actual"].sum()
                / work["actual"].sum(),
                "distinct_bakeries": int(group["bakery_id"].nunique()),
            }
        )
        if (
            name
            in {
                "all_holdout",
                "late_sale_candidate",
                "late_sale_with_surplus_candidate",
                "stg_exact_surplus_candidate",
            }
            and group["actual"].sum() > 0
        ):
            for model, column in (
                ("raw_sales_lora", "prediction"),
                ("reconstructed_demand_lora", "demand_lora_prediction"),
            ):
                score_rows.append(
                    {
                        "tier": name,
                        "model": model,
                        "sku_days": len(group),
                        **score(group, column),
                    }
                )
    origin_rows = []
    for origin, group in work.groupby("forecast_origin"):
        origin_rows.append(
            {
                "origin": str(origin.date()),
                "sku_days": len(group),
                "stg_covered": int(group["stg_bakery_day"].sum()),
                "stg_reconciled_exact": int(group["stg_reconciled_exact"].sum()),
                "stg_reconciled": int(group["stg_reconciled_1unit"].sum()),
                "broad_signal": int(group["broad_stockout_signal"].sum()),
                "restored": int(group["is_restored"].sum()),
                "late_sale_candidate": int(group["late_sale_candidate"].sum()),
                "late_sale_with_surplus_candidate": int(
                    group["late_sale_with_surplus_candidate"].sum()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    scores = pd.DataFrame(score_rows)
    origins = pd.DataFrame(origin_rows)
    sensitivity_rows = []
    for close_hours in (0.5, 1.0, 2.0):
        for surplus_units in (0.0, 2.0, 5.0):
            mask = (
                work["sparse_source_row"]
                & positive
                & close_gap.ge(0)
                & close_gap.le(close_hours)
                & residual.ge(surplus_units)
                & work["written_off_qty"].eq(0)
                & ~work["broad_stockout_signal"]
            )
            sensitivity_rows.append(
                {
                    "close_hours": close_hours,
                    "min_daily_surplus_units": surplus_units,
                    "sku_days": int(mask.sum()),
                    "share_of_holdout_pct": float(100 * mask.mean()),
                    "sales_share_pct": float(
                        100 * work.loc[mask, "actual"].sum() / work["actual"].sum()
                    ),
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    metadata = {
        "research_only": True,
        "production_write": False,
        "verified_true_latent_demand_labels": 0,
        "why_zero": (
            "No independent shelf-availability or unmet-request observations; "
            "daily supply is not timed."
        ),
        "late_sale_definition": (
            "Last SKU sale within one hour of last selected-product sale at "
            "the bakery; not a verified closing time."
        ),
        "surplus_definition": (
            "Daily release + inbound - outbound - observed sales - writeoffs "
            ">= max(2, 10% of sales), no writeoff."
        ),
        "candidate_interpretation": (
            "Sales-as-demand compatibility screen, not proven availability; "
            "selection favors active, higher-volume SKU-days."
        ),
        "sparse_source_note": (
            "Historical sales are DISTINCT-projected fct_check_lines, "
            "not network-wide stg_check_lines."
        ),
        "stg_source_note": (
            "Independent stg_check_lines export covers 10 overlapping bakeries "
            "through 2026-07-19 only; zero SKU sales are reconciled only when "
            "the bakery-day is present."
        ),
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [
                RAW_LORA,
                DEMAND_LORA,
                DEMAND_PANEL,
                STG_SOURCE,
                SPARSE_SOURCE / "202607.csv.gz",
                SPARSE_SOURCE / "202608.csv.gz",
            ]
        },
    }
    args.output_dir.mkdir(parents=True)
    summary.to_csv(args.output_dir / "tier_coverage.csv", index=False)
    scores.to_csv(args.output_dir / "candidate_scores.csv", index=False)
    origins.to_csv(args.output_dir / "by_origin.csv", index=False)
    sensitivity.to_csv(args.output_dir / "sensitivity.csv", index=False)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(scores.to_string(index=False))
    print(origins.to_string(index=False))
    print(sensitivity.to_string(index=False))


if __name__ == "__main__":
    main()
