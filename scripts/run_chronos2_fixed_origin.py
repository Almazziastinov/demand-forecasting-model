"""Research-only zero-shot Chronos-2-Small fixed-origin sales benchmark.

Requires the optional ``chronos-forecasting==2.3.2`` environment. The script
reads a frozen local panel and writes local reports; it has no production IO.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_tournament.fixed_origin import validate_flows  # noqa: E402
from src.model_tournament.metrics import canonical_metrics  # noqa: E402
from src.model_tournament.nhits_candidate import (  # noqa: E402
    evaluation_history_and_rows,
)


MODEL_VARIANTS = {
    "small": (
        "new_chronos2_small_zero_shot_h14",
        "autogluon/chronos-2-small",
        "ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a",
    ),
    "base": (
        "new_chronos2_base_zero_shot_h14",
        "amazon/chronos-2",
        "29ec3766d36d6f73f0696f85560a422f50e8498c",
    ),
}


def _origins(value: str) -> list[pd.Timestamp]:
    origins = [pd.Timestamp(part.strip()).normalize() for part in value.split(",")]
    if not origins or len(origins) != len(set(origins)):
        raise ValueError("Evaluation origins must be nonempty and unique")
    ordered = sorted(origins)
    for previous, current in zip(ordered, ordered[1:]):
        if current <= previous + pd.Timedelta(days=13):
            raise ValueError("Evaluation windows overlap")
    return ordered


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(detail: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    rows = []
    for keys, frame in detail.groupby(group, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append({**dict(zip(group, keys, strict=True)), **canonical_metrics(frame)})
    return pd.DataFrame(rows)


def align_chronos_predictions(
    rows: pd.DataFrame,
    predicted: pd.DataFrame,
    *,
    model_id: str = MODEL_VARIANTS["small"][0],
) -> pd.DataFrame:
    """Require a prediction for every frozen SKU-day before scoring."""
    required = {"unique_id", "ds", "0.5"}
    if not required.issubset(predicted.columns):
        missing = sorted(required - set(predicted.columns))
        raise ValueError(f"Chronos output lacks {missing}")
    if predicted.duplicated(["unique_id", "ds"]).any():
        raise ValueError("Chronos output has duplicate series-day keys")
    expected = rows[
        ["forecast_origin", "date", "lead_days", "bakery_id", "product_id", "actual"]
    ].copy()
    expected["unique_id"] = (
        expected["bakery_id"].astype(str) + ":" + expected["product_id"].astype(str)
    )
    merged = expected.merge(
        predicted[["unique_id", "ds", "0.5"]],
        left_on=["unique_id", "date"],
        right_on=["unique_id", "ds"],
        how="left",
        validate="one_to_one",
    )
    values = pd.to_numeric(merged["0.5"], errors="coerce").to_numpy(dtype=float)
    if len(predicted) != len(expected) or not np.isfinite(values).all():
        raise ValueError("Chronos output does not fully cover the frozen scope")
    detail = expected.drop(columns="unique_id")
    detail["model"] = model_id
    detail["prediction"] = np.maximum(values, 0.0)
    return detail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--evaluation-origins", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-variant", choices=MODEL_VARIANTS, default="small")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.max_pairs is not None and args.max_pairs < 1:
        raise ValueError("max-pairs must be positive")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    origins = _origins(args.evaluation_origins)
    model_id, model_repo, model_revision = MODEL_VARIANTS[args.model_variant]

    try:
        import torch
        from chronos import Chronos2Pipeline
    except ImportError as exc:
        raise SystemExit(
            "Install chronos-forecasting==2.3.2 in a research environment"
        ) from exc

    torch.set_num_threads(4)
    pipeline = Chronos2Pipeline.from_pretrained(
        model_repo, revision=model_revision, device_map="cpu"
    )
    flows = validate_flows(pd.read_parquet(args.input))
    detail_parts = []
    origin_counts = []
    for origin in origins:
        context, rows = evaluation_history_and_rows(
            flows, origin, max_pairs=args.max_pairs
        )
        predicted = pipeline.predict_df(
            context,
            id_column="unique_id",
            timestamp_column="ds",
            target="y",
            prediction_length=14,
            quantile_levels=[0.5],
            batch_size=args.batch_size,
        )
        detail_parts.append(
            align_chronos_predictions(rows, predicted, model_id=model_id)
        )
        baseline = rows[
            [
                "forecast_origin", "date", "lead_days", "bakery_id", "product_id",
                "actual",
            ]
        ].copy()
        baseline["model"] = "weighted_weekday_sales_formula"
        baseline["prediction"] = rows["weighted_weekday_sales"].to_numpy(dtype=float)
        detail_parts.append(baseline)
        origin_counts.append(
            {"origin": str(origin.date()), "pairs": len(rows) // 14, "rows": len(rows)}
        )

    detail = pd.concat(detail_parts, ignore_index=True)
    leaderboard = _summary(detail, ["model"]).sort_values("wmape_pct")
    metadata = {
        "production_write": False,
        "comparable_full_scope": args.max_pairs is None,
        "target": "observed_sales_qty",
        "scope": (
            "positive release in origin-55..origin; "
            "all eligible pairs on every future date"
        ),
        "history_cutoff": "forecast_origin for every 56-day inference context",
        "evaluation_origins": [str(origin.date()) for origin in origins],
        "evaluation_counts": origin_counts,
        "horizon_days": 14,
        "model": model_id,
        "model_variant": args.model_variant,
        "model_repo": model_repo,
        "model_revision": model_revision,
        "model_training": "published pretrained checkpoint; zero-shot on this panel",
        "chronos_forecasting_version": version("chronos-forecasting"),
        "torch_version": torch.__version__,
        "max_pairs": args.max_pairs,
        "batch_size": args.batch_size,
        "missing_sparse_fact_rows_as_zero": True,
        "as_of_fact_arrival_verified": False,
        "production_target_and_scope_parity_verified": False,
        "input_path": str(args.input.resolve()),
        "input_sha256": _sha256(args.input),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    detail.to_parquet(args.output_dir / "detail.parquet", index=False)
    leaderboard.to_csv(args.output_dir / "leaderboard.csv", index=False)
    _summary(detail, ["forecast_origin", "model"]).to_csv(
        args.output_dir / "by_origin.csv", index=False
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()
