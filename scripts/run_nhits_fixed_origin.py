"""Research-only N-HiTS evaluation on frozen 14-day sales windows.

Requires optional ``neuralforecast==3.1.2`` and PyTorch dependencies. This
script writes local reports only; it has no production connection or writer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_tournament.fixed_origin import validate_flows  # noqa: E402
from src.model_tournament.metrics import canonical_metrics  # noqa: E402
from src.model_tournament.nhits_candidate import (  # noqa: E402
    INPUT_DAYS,
    MODEL_ID,
    align_predictions,
    evaluation_history_and_rows,
    training_history,
)


BASELINE_ID = "weighted_weekday_sales_formula"


def _dates(value: str) -> list[pd.Timestamp]:
    dates = [pd.Timestamp(piece.strip()).normalize() for piece in value.split(",")]
    if not dates or len(dates) != len(set(dates)):
        raise ValueError("Origins must be nonempty and unique")
    return sorted(dates)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(detail: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    records = []
    for keys, group in detail.groupby(columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        records.append(
            {**dict(zip(columns, keys, strict=True)), **canonical_metrics(group)}
        )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-origins", required=True)
    parser.add_argument("--evaluation-origins", required=True)
    parser.add_argument("--train-start", default="2026-01-01")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Smoke test only; not a comparable score",
    )
    args = parser.parse_args()
    train_origins = _dates(args.train_origins)
    eval_origins = _dates(args.evaluation_origins)
    if max(train_origins) + pd.Timedelta(days=14) >= min(eval_origins):
        raise ValueError("Training labels must precede evaluation origins")
    for previous, current in zip(eval_origins, eval_origins[1:]):
        if current <= previous + pd.Timedelta(days=13):
            raise ValueError("Evaluation windows overlap")
    if args.max_steps < 1:
        raise ValueError("max_steps must be positive")

    try:
        import neuralforecast
        import torch
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NHITS
    except ImportError as exc:
        raise SystemExit(
            "Install the optional neuralforecast==3.1.2 research environment"
        ) from exc

    flows = validate_flows(pd.read_parquet(args.input))
    train, train_counts = training_history(
        flows,
        train_origins,
        start=pd.Timestamp(args.train_start),
        max_pairs=args.max_pairs,
    )
    torch.set_num_threads(4)
    model = NHITS(
        h=14,
        input_size=INPUT_DAYS,
        max_steps=args.max_steps,
        scaler_type="robust",
        random_seed=42,
        enable_progress_bar=False,
        accelerator="cpu",
        devices=1,
    )
    forecast = NeuralForecast(models=[model], freq="D")
    forecast.fit(df=train, val_size=0)

    parts = []
    eval_counts = []
    for origin in eval_origins:
        context, rows = evaluation_history_and_rows(
            flows, origin, max_pairs=args.max_pairs
        )
        predicted = forecast.predict(df=context)
        parts.append(align_predictions(rows, predicted))
        baseline = rows[
            [
                "forecast_origin",
                "date",
                "lead_days",
                "bakery_id",
                "product_id",
                "actual",
            ]
        ].copy()
        baseline["model"] = BASELINE_ID
        baseline["prediction"] = rows["weighted_weekday_sales"].to_numpy(dtype=float)
        parts.append(baseline)
        eval_counts.append(
            {"origin": str(origin.date()), "pairs": len(rows) // 14, "rows": len(rows)}
        )
    detail = pd.concat(parts, ignore_index=True)
    leaderboard = _summary(detail, ["model"]).sort_values("wmape_pct")
    metadata = {
        "production_write": False,
        "comparable_full_scope": args.max_pairs is None,
        "same_training_rows_as_tabular_candidates": False,
        "training_protocol": (
            "Global univariate daily series through train_cutoff, selected from "
            "train-origin scopes; tabular candidates use fixed-origin rows"
        ),
        "target": "observed_sales_qty",
        "scope": (
            "positive release in origin-55..origin; "
            "all eligible pairs on every future date"
        ),
        "train_cutoff": str((max(train_origins) + pd.Timedelta(days=14)).date()),
        "train_start": args.train_start,
        "train_origins": [str(origin.date()) for origin in train_origins],
        "evaluation_origins": [str(origin.date()) for origin in eval_origins],
        "train_counts": train_counts,
        "evaluation_counts": eval_counts,
        "input_days": INPUT_DAYS,
        "horizon_days": 14,
        "max_steps": args.max_steps,
        "random_seed": 42,
        "validation_size": 0,
        "max_pairs": args.max_pairs,
        "model": MODEL_ID,
        "model_notes": (
            "Nixtla N-HiTS, global univariate observed-sales model; "
            "no future exogenous data"
        ),
        "neuralforecast_version": neuralforecast.__version__,
        "torch_version": torch.__version__,
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
