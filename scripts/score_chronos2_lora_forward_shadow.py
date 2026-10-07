"""Score a completed, frozen Chronos-2 shadow against read-only sales facts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.export_clickhouse_checks import create_client  # noqa: E402
from scripts.run_chronos2_fixed_origin import _sha256, _summary  # noqa: E402


KEYS = ["date", "bakery_id", "product_id"]
CHRONOS_MODELS = {"chronos2_small_zero_shot", "chronos2_small_lora_1000"}


def align_actuals(
    predictions: pd.DataFrame, sales: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use zero only for absent SKU rows inside an otherwise covered day."""
    if predictions.duplicated(KEYS + ["model"]).any():
        raise ValueError("Frozen predictions contain duplicate SKU-day/model keys")
    if set(predictions["model"].unique()) < CHRONOS_MODELS:
        raise ValueError("Frozen predictions lack both Chronos contenders")
    if sales.duplicated(KEYS).any():
        raise ValueError("Sales facts contain duplicate SKU-day keys")
    scope = predictions.loc[
        predictions["model"] == "chronos2_small_zero_shot", KEYS
    ]
    adapted = predictions.loc[
        predictions["model"] == "chronos2_small_lora_1000", KEYS
    ]
    if len(scope) != len(adapted) or not scope.merge(
        adapted, on=KEYS, how="outer", indicator=True
    )["_merge"].eq("both").all():
        raise ValueError("Chronos contenders do not share exact SKU-day keys")
    sales = sales.rename(columns={"observed_sales_qty": "actual"})
    actual = pd.to_numeric(sales["actual"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(actual).all() or (actual < 0).any():
        raise ValueError("Sales facts contain invalid observed quantities")
    detail = predictions.merge(
        sales[KEYS + ["actual"]], on=KEYS, how="left", validate="many_to_one"
    )
    detail["actual"] = detail["actual"].fillna(0.0)
    common_keys = predictions.loc[
        predictions["model"] == "incumbent_active_at_freeze", KEYS
    ]
    common = detail.merge(common_keys, on=KEYS, how="inner", validate="many_to_one")
    return detail, common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-report", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    shadow_meta = json.loads(
        (args.shadow_report / "metadata.json").read_text(encoding="utf-8")
    )
    predictions = pd.read_parquet(args.shadow_report / "predictions.parquet")
    predictions["date"] = pd.to_datetime(predictions["date"]).dt.normalize()
    first, last = predictions["date"].min(), predictions["date"].max()
    if (
        str(first.date()) != shadow_meta["first_prospective_date"]
        or str(last.date()) != shadow_meta["last_prospective_date"]
    ):
        raise ValueError("Prediction dates differ from frozen metadata")
    now_msk = datetime.now(ZoneInfo("Europe/Moscow"))
    if now_msk.date() <= (last + pd.Timedelta(days=1)).date():
        raise ValueError("Wait at least one full day after the final target date")
    bakery_ids = tuple(sorted(predictions["bakery_id"].astype(int).unique()))
    client = create_client(args.env_file)
    extraction_started = datetime.now(ZoneInfo("UTC"))
    sales = client.query_df(
        """
        SELECT check_date AS date,
               toInt64OrZero(toString(bakery_id)) AS bakery_id,
               toInt64OrZero(toString(product_id)) AS product_id,
               sum(toFloat64(quantity)) AS observed_sales_qty
        FROM (
            SELECT DISTINCT check_datetime, check_date, bakery_id, product_id,
                            quantity, line_amount
            FROM Svezhar.fct_check_lines
            WHERE hex(cash_event_type) = 'D09FD180D0BED0B4D0B0D0B6D0B0'
              AND check_date BETWEEN toDate(%(start)s) AND toDate(%(end)s)
              AND toInt64OrZero(toString(bakery_id)) IN %(bakery_ids)s
        )
        GROUP BY date, bakery_id, product_id
        """,
        parameters={
            "start": str(first.date()),
            "end": str(last.date()),
            "bakery_ids": bakery_ids,
        },
    )
    extraction_finished = datetime.now(ZoneInfo("UTC"))
    sales["date"] = pd.to_datetime(sales["date"]).dt.normalize()
    daily_coverage = sales.groupby("date")["bakery_id"].nunique()
    expected_dates = pd.date_range(first, last, freq="D")
    if not daily_coverage.reindex(expected_dates).ge(50).all():
        raise ValueError("Future sales facts lack 50-bakery daily coverage")
    detail, common = align_actuals(predictions, sales)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sales.to_parquet(args.output_dir / "frozen_actuals.parquet", index=False)
    detail.to_parquet(args.output_dir / "detail.parquet", index=False)
    _summary(detail, ["model"]).to_csv(args.output_dir / "leaderboard.csv", index=False)
    _summary(detail, ["date", "model"]).to_csv(
        args.output_dir / "by_date.csv", index=False
    )
    _summary(common, ["model"]).to_csv(
        args.output_dir / "common_scope.csv", index=False
    )
    meta = {
        **shadow_meta,
        "score_valid": True,
        "actual_extraction_started_utc": extraction_started.isoformat(),
        "actual_extraction_finished_utc": extraction_finished.isoformat(),
        "actuals_sha256": _sha256(args.output_dir / "frozen_actuals.parquet"),
        "min_daily_bakery_coverage": int(daily_coverage.min()),
        "evaluation_target": "observed_sales_qty",
        "research_only": True,
        "economic_value_verified": False,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(_summary(detail, ["model"]).to_string(index=False))


if __name__ == "__main__":
    main()
