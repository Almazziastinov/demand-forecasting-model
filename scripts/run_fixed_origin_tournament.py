"""Compare sales models on fixed-origin, non-overlapping 14-day windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_tournament.fixed_origin import (  # noqa: E402
    build_fixed_origin_panel,
    validate_flows,
)
from src.model_tournament.fixed_origin_candidates import (  # noqa: E402
    BASELINE_ID,
    DEFAULT_CANDIDATES,
    PROVENANCE,
    fit_predict_candidates,
)
from src.model_tournament.metrics import canonical_metrics  # noqa: E402


FEATURES = [
    "bakery_id",
    "product_id",
    "lead_days",
    "day_of_week",
    "month",
    "day_of_month",
    "origin_mean1",
    "origin_mean7",
    "origin_mean14",
    "weighted_weekday_sales",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_origins(value: str) -> list[pd.Timestamp]:
    origins = [pd.Timestamp(part.strip()).normalize() for part in value.split(",")]
    if not origins or len(origins) != len(set(origins)):
        raise ValueError("Origin dates must be nonempty and unique")
    return sorted(origins)


def _summaries(detail: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    records = []
    for keys, group in detail.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        records.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                **canonical_metrics(group),
            }
        )
    return pd.DataFrame(records)


def run_fixed_origin_tournament(
    flows: pd.DataFrame,
    train_origins: list[pd.Timestamp],
    evaluation_origins: list[pd.Timestamp],
    *,
    horizon_days: int = 14,
    n_estimators: int = 160,
    candidate_models: tuple[str, ...] = DEFAULT_CANDIDATES,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if not train_origins or not evaluation_origins:
        raise ValueError("Training and evaluation origins are both required")
    if horizon_days < 1:
        raise ValueError("horizon_days must be positive")
    sorted_eval = sorted(evaluation_origins)
    for previous, current in zip(sorted_eval, sorted_eval[1:]):
        if current <= previous + pd.Timedelta(days=horizon_days - 1):
            raise ValueError("Evaluation windows must not overlap")
    train = pd.concat(
        [
            build_fixed_origin_panel(flows, origin, horizon_days=horizon_days)
            for origin in train_origins
        ],
        ignore_index=True,
    )
    evaluation = pd.concat(
        [
            build_fixed_origin_panel(flows, origin, horizon_days=horizon_days)
            for origin in evaluation_origins
        ],
        ignore_index=True,
    )
    if train["date"].max() >= evaluation["date"].min():
        raise ValueError("Training labels overlap the evaluation period")
    combined = pd.concat([train[FEATURES], evaluation[FEATURES]], ignore_index=True)
    for column in ("bakery_id", "product_id"):
        combined[column] = combined[column].astype("category")
    X_train = combined.iloc[: len(train)]
    X_eval = combined.iloc[len(train) :]
    y_train = train["actual"].to_numpy(dtype=float)
    predictions = {
        BASELINE_ID: evaluation["weighted_weekday_sales"].to_numpy(dtype=float)
    }
    predictions.update(
        fit_predict_candidates(
            X_train,
            X_eval,
            y_train,
            train["weighted_weekday_sales"].to_numpy(dtype=float),
            evaluation["weighted_weekday_sales"].to_numpy(dtype=float),
            model_ids=candidate_models,
            n_estimators=n_estimators,
        )
    )
    columns = [
        "forecast_origin",
        "date",
        "lead_days",
        "bakery_id",
        "product_id",
        "actual",
    ]
    parts = []
    for model_id, values in predictions.items():
        part = evaluation[columns].copy()
        part["model"] = model_id
        part["prediction"] = values
        parts.append(part)
    detail = pd.concat(parts, ignore_index=True)
    leaderboard = _summaries(detail, ["model"]).sort_values(["wmape_pct", "mae"])
    leaderboard["model_class"] = leaderboard["model"].map(
        {model_id: info["class"] for model_id, info in PROVENANCE.items()}
    )
    metadata = {
        "production_write": False,
        "target": "observed_sales_qty",
        "scope": (
            "positive release in origin-55..origin; all eligible pairs "
            "on every future date"
        ),
        "history_cutoff": "fixed forecast_origin for scope and every feature",
        "train_origins": [str(value.date()) for value in train_origins],
        "evaluation_origins": [str(value.date()) for value in evaluation_origins],
        "last_train_target_date": str(train["date"].max().date()),
        "first_evaluation_target_date": str(evaluation["date"].min().date()),
        "horizon_days": horizon_days,
        "train_rows": len(train),
        "evaluation_rows_per_model": len(evaluation),
        "feature_names": FEATURES,
        "n_estimators": n_estimators,
        "models": [BASELINE_ID, *candidate_models],
        "model_provenance": {
            model_id: PROVENANCE[model_id]
            for model_id in [BASELINE_ID, *candidate_models]
        },
        "as_of_fact_arrival_verified": False,
        "production_target_and_scope_parity_verified": False,
    }
    return detail, leaderboard, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-origins", required=True)
    parser.add_argument("--evaluation-origins", required=True)
    parser.add_argument("--horizon-days", type=int, default=14)
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_CANDIDATES),
        help="Comma-separated trainable candidate IDs; baseline is always included",
    )
    parser.add_argument("--n-estimators", type=int, default=160)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    train_origins = _parse_origins(args.train_origins)
    evaluation_origins = _parse_origins(args.evaluation_origins)
    candidate_models = tuple(
        model.strip() for model in args.models.split(",") if model.strip()
    )
    flows = validate_flows(pd.read_parquet(args.input))
    detail, leaderboard, metadata = run_fixed_origin_tournament(
        flows,
        train_origins,
        evaluation_origins,
        horizon_days=args.horizon_days,
        n_estimators=args.n_estimators,
        candidate_models=candidate_models,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    detail.to_parquet(args.output_dir / "detail.parquet", index=False)
    leaderboard.to_csv(args.output_dir / "leaderboard.csv", index=False)
    _summaries(detail, ["forecast_origin", "model"]).to_csv(
        args.output_dir / "by_origin.csv", index=False
    )
    _summaries(detail, ["lead_days", "model"]).to_csv(
        args.output_dir / "by_lead.csv", index=False
    )
    metadata["input_path"] = str(args.input.resolve())
    metadata["input_sha256"] = _sha256(args.input)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(leaderboard.to_string(index=False))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
