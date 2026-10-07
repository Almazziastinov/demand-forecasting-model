"""Re-score frozen research predictions against sales and a demand proxy.

This is a target-alignment audit, not validation of true latent demand.
All source predictions are read-only and the proxy is not independent of the
full reconstructed-demand training target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["forecast_origin", "date", "lead_days", "bakery_id", "product_id"]
REPORTS = [
    "reconstructed_demand_baseline_eval_20261005",
    "reconstruction_early_folds_20261005",
    "weekly_bridge_early_folds_20261005_v3",
    "unexplained_weekly_dips_early_folds_20261005_v5",
    "hierarchical_weekly_bridge_early_folds_20261005_v1",
    "volatility_guarded_dips_20261005_v2",
    "chronos2_demand_lora_full_1000_20261005",
    "chronos2_volatility_guarded_lora_20261005_v1",
    "chronos2_volatility_guarded_factorial_20261005_v1",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score(frame: pd.DataFrame, label: str) -> tuple[float, float]:
    actual = frame[label].to_numpy(dtype=float)
    prediction = frame["prediction"].to_numpy(dtype=float)
    if not np.isfinite(actual).all() or not np.isfinite(prediction).all():
        raise ValueError("Non-finite score input")
    denominator = actual.sum()
    if denominator <= 0:
        raise ValueError("Non-positive target sum")
    return (
        float(100 * np.abs(prediction - actual).sum() / denominator),
        float(100 * (prediction - actual).sum() / denominator),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    panel = pd.read_parquet(
        args.panel,
        columns=[
            "date",
            "bakery_id",
            "product_id",
            "observed_sales_qty",
            "reconstructed_demand_qty",
        ],
    )
    panel_keys = ["date", "bakery_id", "product_id"]
    if panel.duplicated(panel_keys).any():
        raise ValueError("Demand proxy panel has duplicate SKU-days")
    results = []
    checksums = {}
    for name in REPORTS:
        path = args.reports_dir / name / "detail.parquet"
        detail = pd.read_parquet(path)
        required = set([*KEYS, "actual", "model", "prediction"])
        if not required.issubset(detail):
            raise ValueError(f"{name} lacks {sorted(required - set(detail))}")
        if detail.duplicated([*KEYS, "model"]).any():
            raise ValueError(f"{name} has duplicate model/SKU-days")
        merged = detail[[*KEYS, "actual", "model", "prediction"]].merge(
            panel, on=panel_keys, how="left", validate="many_to_one"
        )
        if merged["observed_sales_qty"].isna().any():
            raise ValueError(f"{name} has SKU-days missing from demand panel")
        if not np.allclose(merged["actual"], merged["observed_sales_qty"], atol=1e-8):
            raise ValueError(f"{name} was not scored against panel sales")
        model_keys = merged.groupby("model").size()
        common_scope = model_keys.nunique() == 1
        if common_scope:
            first = merged.loc[merged["model"].eq(merged["model"].iloc[0]), KEYS]
            for _, group in merged.groupby("model"):
                if len(
                    first.merge(
                        group[KEYS], on=KEYS, how="inner", validate="one_to_one"
                    )
                ) != len(first):
                    common_scope = False
                    break
        for model, group in merged.groupby("model", sort=True):
            sales_wmape, sales_bias = _score(group, "observed_sales_qty")
            proxy_wmape, proxy_bias = _score(group, "reconstructed_demand_qty")
            results.append(
                {
                    "report": name,
                    "model": model,
                    "rows": len(group),
                    "common_scope_within_report": common_scope,
                    "sales_wmape_pct": sales_wmape,
                    "proxy_demand_wmape_pct": proxy_wmape,
                    "sales_bias_pct": sales_bias,
                    "proxy_demand_bias_pct": proxy_bias,
                    "proxy_changed_rows": int(
                        group["reconstructed_demand_qty"]
                        .ne(group["observed_sales_qty"])
                        .sum()
                    ),
                }
            )
        checksums[name] = _sha256(path)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = pd.DataFrame(results)
    result.to_csv(args.output_dir / "dual_target_scores.csv", index=False)
    metadata = {
        "production_write": False,
        "decision_grade_demand_score": False,
        "reason": "Evaluation proxy is not independent of its reconstruction algorithm",
        "panel_sha256": _sha256(args.panel),
        "detail_sha256": checksums,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        result[
            ["report", "model", "sales_wmape_pct", "proxy_demand_wmape_pct"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
