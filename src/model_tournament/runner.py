"""Orchestration for a local, read-only model tournament."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import numpy as np

from src.model_tournament.baselines import predict_baseline
from src.model_tournament.metrics import canonical_metrics
from src.model_tournament.trainable import train_and_predict


DEFAULT_KEYS = ("bakery_id", "product_id")
DEFAULT_BREAKDOWNS = (
    "date",
    "week",
    "month",
    "dow",
    "city",
    "category",
    "category_name",
    "bakery_id",
    "product_id",
)


@dataclass(frozen=True)
class TournamentConfig:
    input_path: Path
    output_dir: Path
    date_from: str
    date_to: str
    target_col: str
    models: tuple[str, ...]
    prediction_columns: tuple[tuple[str, str], ...] = ()
    target_id: str = "unspecified_target"
    scope_id: str = "unspecified_scope"
    reference_model: str | None = None
    sales_col: str | None = None
    key_cols: tuple[str, ...] = DEFAULT_KEYS
    min_coverage: float = 1.0
    lead_days: int = 1
    history_cutoff: str | None = None
    trainable_models: tuple[str, ...] = ()
    train_end: str | None = None


def load_panel(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".gz"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported panel format: {path.suffix}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_panel(frame: pd.DataFrame, config: TournamentConfig) -> pd.DataFrame:
    required = {"date", *config.key_cols, config.target_col}
    required.update(column for _, column in config.prediction_columns)
    if config.sales_col:
        required.add(config.sales_col)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Panel is missing required columns: {missing}")

    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"], errors="raise").dt.normalize()
    keys = ["date", *config.key_cols]
    duplicate_count = int(work.duplicated(keys).sum())
    if duplicate_count:
        raise ValueError(f"Panel contains {duplicate_count} duplicate canonical keys")
    work[config.target_col] = pd.to_numeric(work[config.target_col], errors="coerce")
    if not np.isfinite(work[config.target_col].to_numpy(dtype=float)).all():
        raise ValueError("Target contains missing or non-finite values")
    if (work[config.target_col] < 0).any():
        raise ValueError("Target contains negative values")
    if config.sales_col:
        work[config.sales_col] = pd.to_numeric(
            work[config.sales_col], errors="coerce"
        )
        if not np.isfinite(work[config.sales_col].to_numpy(dtype=float)).all():
            raise ValueError("Sales contain missing or non-finite values")
        if (work[config.sales_col] < 0).any():
            raise ValueError("Sales contain negative values")
        if (work[config.target_col] + 1e-9 < work[config.sales_col]).any():
            raise ValueError("Demand target is below observed sales")
    return work.sort_values(keys).reset_index(drop=True)


def _summarize_breakdown(
    detail: pd.DataFrame, column: str, sales_col: str | None
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = detail.groupby([column, "model"], dropna=False, sort=True)
    for (value, model), group in groups:
        metrics = canonical_metrics(
            group,
            actual_col="actual",
            prediction_col="prediction",
            sales_col=sales_col,
        )
        rows.append({column: value, "model": model, **metrics})
    return pd.DataFrame(rows)


def _pairwise_wins(
    detail: pd.DataFrame,
    *,
    id_columns: list[str],
    reference_model: str,
) -> pd.DataFrame:
    reference = detail[detail["model"].eq(reference_model)][
        [*id_columns, "actual", "prediction"]
    ].rename(columns={"prediction": "reference_prediction"})
    if reference.empty:
        raise ValueError(f"Reference model {reference_model!r} is absent")
    rows: list[dict[str, object]] = []
    for model, candidate in detail.groupby("model", sort=True):
        compared = candidate.merge(
            reference,
            on=[*id_columns, "actual"],
            how="inner",
            validate="one_to_one",
        )
        candidate_error = (compared["prediction"] - compared["actual"]).abs()
        reference_error = (
            compared["reference_prediction"] - compared["actual"]
        ).abs()
        delta = candidate_error - reference_error
        rows.append(
            {
                "model": model,
                "reference_model": reference_model,
                "rows": int(len(compared)),
                "wins": int((delta < -1e-9).sum()),
                "ties": int(delta.abs().le(1e-9).sum()),
                "losses": int((delta > 1e-9).sum()),
                "win_rate_pct": float(100.0 * (delta < -1e-9).mean()),
                "mae_delta_vs_reference": float(delta.mean()),
                "absolute_error_delta_qty": float(delta.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mae_delta_vs_reference", "win_rate_pct"], ascending=[True, False]
    )


def run_tournament(config: TournamentConfig) -> dict[str, Path]:
    if not 0.0 < config.min_coverage <= 1.0:
        raise ValueError("min_coverage must be in (0, 1]")
    if config.lead_days < 1:
        raise ValueError("lead_days must be at least 1")
    panel = validate_panel(load_panel(config.input_path), config)
    start = pd.Timestamp(config.date_from).normalize()
    end = pd.Timestamp(config.date_to).normalize()
    if start > end:
        raise ValueError("date_from must not be after date_to")
    history_cutoff = (
        pd.Timestamp(config.history_cutoff).normalize()
        if config.history_cutoff
        else None
    )
    if history_cutoff is not None and history_cutoff >= start:
        raise ValueError("history_cutoff must precede every evaluation date")
    evaluation_mask = panel["date"].between(start, end)
    evaluation = panel.loc[evaluation_mask].copy()
    if evaluation.empty:
        raise ValueError("Evaluation period is empty")

    id_columns = ["date", *config.key_cols]
    context_columns = [
        column
        for column in ("city", "category", "category_name")
        if column in evaluation.columns
    ]
    base_columns = [*id_columns, *context_columns]
    detail_parts: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []
    artifact_columns = dict(config.prediction_columns)
    if len(artifact_columns) != len(config.prediction_columns):
        raise ValueError("Duplicate artifact model IDs")
    if len(set(config.models)) != len(config.models):
        raise ValueError("Duplicate baseline model IDs")
    all_model_ids = [*config.models, *artifact_columns, *config.trainable_models]
    if len(all_model_ids) != len(set(all_model_ids)):
        raise ValueError(
            "Model IDs must be unique across baselines, artifacts and trainable models"
        )
    model_ids = all_model_ids
    if not model_ids:
        raise ValueError("At least one model is required")
    if config.trainable_models and (
        config.lead_days != 1 or history_cutoff is not None
    ):
        raise ValueError("Trainable MVP supports daily lead-one evaluation only")
    train_end = (
        pd.Timestamp(config.train_end).normalize()
        if config.train_end
        else start - pd.Timedelta(days=1)
    )
    if config.trainable_models and train_end >= start:
        raise ValueError("train_end must precede the evaluation period")
    model_provenance: dict[str, dict[str, object]] = {}
    for model in model_ids:
        if model in artifact_columns:
            raw_predictions = panel[artifact_columns[model]]
            all_predictions = pd.to_numeric(raw_predictions, errors="coerce")
            malformed = raw_predictions.notna() & all_predictions.isna()
            if malformed.any():
                raise ValueError(
                    f"Model {model!r} has {int(malformed.sum())} non-numeric "
                    "artifact predictions"
                )
            model_source = f"panel_column:{artifact_columns[model]}"
            model_class = "historical_frozen_unverified"
            model_provenance[model] = {
                "source": model_source,
                "training_cutoff_verified": False,
            }
        elif model in config.trainable_models:
            all_predictions, provenance = train_and_predict(
                model,
                panel,
                target_col=config.target_col,
                key_cols=list(config.key_cols),
                evaluation_mask=evaluation_mask,
                train_end=train_end,
            )
            model_source = f"fresh_training:{model}"
            model_class = "freshly_trained_on_panel"
            model_provenance[model] = provenance
        else:
            all_predictions = predict_baseline(
                model,
                panel,
                target_col=config.target_col,
                key_cols=list(config.key_cols),
                lead_days=config.lead_days,
                history_cutoff=history_cutoff,
            )
            model_source = f"causal_baseline:{model}"
            model_class = "recomputed_baseline"
            model_provenance[model] = {
                "source": model_source,
                "history_cutoff_rule": (
                    str(history_cutoff.date())
                    if history_cutoff is not None
                    else f"target_date-{config.lead_days}d"
                ),
            }
        predicted = all_predictions.loc[evaluation_mask].reset_index(drop=True)
        part = evaluation[base_columns].reset_index(drop=True)
        part["actual"] = evaluation[config.target_col].to_numpy()
        if config.sales_col:
            part[config.sales_col] = evaluation[config.sales_col].to_numpy()
        part["model"] = model
        part["prediction"] = predicted
        prediction_values = part["prediction"].to_numpy(dtype=float)
        finite = np.isfinite(prediction_values)
        if (np.isinf(prediction_values)).any():
            raise ValueError(f"Model {model!r} produced non-finite predictions")
        if (prediction_values[finite] < 0).any():
            raise ValueError(f"Model {model!r} produced negative predictions")
        covered = pd.Series(finite)
        coverage = float(covered.mean())
        coverage_rows.append(
            {
                "model": model,
                "evaluation_rows": int(len(part)),
                "covered_rows": int(covered.sum()),
                "coverage_pct": 100.0 * coverage,
                "source": model_source,
                "model_class": model_class,
            }
        )
        if coverage + 1e-12 < config.min_coverage:
            raise ValueError(
                f"Model {model!r} coverage {coverage:.2%} is below "
                f"required {config.min_coverage:.2%}; choose a later date_from "
                "or explicitly lower --min-coverage"
            )
        detail_parts.append(part.loc[covered])

    detail = pd.concat(detail_parts, ignore_index=True)
    key_sets = {
        model: set(map(tuple, group[id_columns].to_numpy()))
        for model, group in detail.groupby("model", sort=False)
    }
    common_keys = set.intersection(*key_sets.values())
    if any(keys != common_keys for keys in key_sets.values()):
        detail = detail[
            detail[id_columns].apply(tuple, axis=1).isin(common_keys)
        ].copy()
    if not common_keys:
        raise ValueError("Models have no common evaluation scope")
    common_coverage = len(common_keys) / len(evaluation)
    if common_coverage + 1e-12 < config.min_coverage:
        raise ValueError(
            f"Common-key coverage {common_coverage:.2%} is below "
            f"required {config.min_coverage:.2%}"
        )

    detail["month"] = detail["date"].dt.to_period("M").astype(str)
    detail["week"] = detail["date"].dt.to_period("W").astype(str)
    detail["dow"] = detail["date"].dt.dayofweek
    summary_rows = []
    for model, group in detail.groupby("model", sort=True):
        summary_rows.append(
            {
                "model": model,
                **canonical_metrics(
                    group,
                    actual_col="actual",
                    prediction_col="prediction",
                    sales_col=config.sales_col,
                ),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["wmape_pct", "mae", "rmse"], na_position="last"
    )
    coverage_frame = pd.DataFrame(coverage_rows)
    summary = summary.merge(
        coverage_frame[["model", "model_class"]],
        on="model",
        how="left",
        validate="one_to_one",
    )
    reference_model = config.reference_model or (
        "weighted_weekday_formula_v1"
        if "weighted_weekday_formula_v1" in model_ids
        else model_ids[0]
    )
    pairwise = _pairwise_wins(
        detail,
        id_columns=id_columns,
        reference_model=reference_model,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "detail": config.output_dir / "detail.parquet",
        "summary": config.output_dir / "leaderboard.csv",
        "coverage": config.output_dir / "coverage.csv",
        "metadata": config.output_dir / "metadata.json",
        "wins_losses": config.output_dir / "wins_losses.csv",
    }
    detail.to_parquet(paths["detail"], index=False)
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    coverage_frame.to_csv(paths["coverage"], index=False, encoding="utf-8-sig")
    pairwise.to_csv(paths["wins_losses"], index=False, encoding="utf-8-sig")

    breakdown_dir = config.output_dir / "breakdowns"
    breakdown_dir.mkdir(exist_ok=True)
    for column in DEFAULT_BREAKDOWNS:
        if column not in detail.columns:
            continue
        breakdown = _summarize_breakdown(detail, column, config.sales_col)
        path = breakdown_dir / f"by_{column}.csv"
        breakdown.to_csv(path, index=False, encoding="utf-8-sig")
        paths[f"by_{column}"] = path

    metadata = {
        "production_write": False,
        "scope_contract": "strict common date/bakery/product keys",
        "target_id": config.target_id,
        "scope_id": config.scope_id,
        "input": {
            "path": str(config.input_path),
            "size_bytes": config.input_path.stat().st_size,
            "sha256": _file_sha256(config.input_path),
        },
        "config": {
            **asdict(config),
            "input_path": str(config.input_path),
            "output_dir": str(config.output_dir),
        },
        "evaluation_rows_per_model": len(common_keys),
        "evaluation_rows_before_intersection": len(evaluation),
        "common_coverage_pct": 100.0 * common_coverage,
        "history_cutoff_rule": (
            f"fixed <= {history_cutoff.date()}"
            if history_cutoff is not None
            else f"per target <= target_date - {config.lead_days} days"
        ),
        "models": model_ids,
        "model_provenance": model_provenance,
        "reference_model": reference_model,
        "date_range": [str(start.date()), str(end.date())],
    }
    paths["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return paths
