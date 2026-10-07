"""Run the local read-only model tournament on a canonical panel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_tournament import TournamentConfig, run_tournament  # noqa: E402
from src.model_tournament.baselines import available_baselines  # noqa: E402
from src.model_tournament.trainable import TRAINABLE_MODELS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare causal forecast baselines on one immutable local panel."
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="CSV or Parquet panel"
    )
    parser.add_argument("--date-from", required=True)
    parser.add_argument("--date-to", required=True)
    parser.add_argument(
        "--target", required=True, help="Target column, e.g. demand or sold"
    )
    parser.add_argument("--target-id", default="unspecified_target")
    parser.add_argument("--scope-id", default="unspecified_scope")
    parser.add_argument(
        "--lead-days", type=int, default=1,
        help="Last available fact date is target date minus this many days",
    )
    parser.add_argument(
        "--history-cutoff",
        help="Optional fixed last available fact date for a multi-day forecast",
    )
    parser.add_argument(
        "--reference-model",
        help="Model used for row-level wins/losses; defaults to weighted weekday",
    )
    parser.add_argument(
        "--sales-column", help="Observed sales column for restored-loss metrics"
    )
    parser.add_argument(
        "--models",
        default="lag7,mean7,same_weekday_mean2,weighted_weekday_formula_v1",
        help=f"Comma-separated baselines. Available: {','.join(available_baselines())}",
    )
    parser.add_argument(
        "--key-columns",
        default="bakery_id,product_id",
        help="Comma-separated entity keys in addition to date",
    )
    parser.add_argument("--min-coverage", type=float, default=1.0)
    parser.add_argument(
        "--trainable-model",
        action="append",
        default=[],
        choices=TRAINABLE_MODELS,
        help="Train a fresh causal LightGBM candidate; may be repeated",
    )
    parser.add_argument(
        "--train-end",
        help="Last training target date; defaults to date-from minus one day",
    )
    parser.add_argument(
        "--prediction-column",
        action="append",
        default=[],
        metavar="MODEL_ID=COLUMN",
        help="Attach an existing forecast column; may be repeated",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = tuple(value.strip() for value in args.models.split(",") if value.strip())
    keys = tuple(
        value.strip() for value in args.key_columns.split(",") if value.strip()
    )
    prediction_columns = []
    for value in args.prediction_column:
        if "=" not in value:
            raise ValueError(
                f"Invalid --prediction-column {value!r}; expected MODEL_ID=COLUMN"
            )
        model_id, column = value.split("=", 1)
        prediction_columns.append((model_id.strip(), column.strip()))
    config = TournamentConfig(
        input_path=args.input.resolve(),
        output_dir=args.output_dir.resolve(),
        date_from=args.date_from,
        date_to=args.date_to,
        target_col=args.target,
        target_id=args.target_id,
        scope_id=args.scope_id,
        lead_days=args.lead_days,
        history_cutoff=args.history_cutoff,
        reference_model=args.reference_model,
        sales_col=args.sales_column,
        models=models,
        prediction_columns=tuple(prediction_columns),
        trainable_models=tuple(args.trainable_model),
        train_end=args.train_end,
        key_cols=keys,
        min_coverage=args.min_coverage,
    )
    paths = run_tournament(config)
    print("Tournament complete (read-only with respect to production).")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
