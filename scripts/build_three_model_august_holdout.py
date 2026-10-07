"""Rebuild three production SKU-allocation variants on one frozen August origin.

The comparison window is 2026-08-24..2026-08-31.  Each date uses the bakery
source run actually selected by the pilot report on that business date.  The
frozen Direct alpha=.25 artifact was trained through 2026-08-23, so every date
is out of sample for the model while its causal history is allowed to roll.
Nothing is written to ClickHouse.

``base_bakery_norm_recent`` is read exactly from the frozen source run.
``direct_alpha_025_v1`` is rebuilt from the frozen artifact and source run.
``base_raw_uplift`` is a causal reconstruction because its July multiplier was
pruned from ClickHouse; the preserved weekly_20260824 multiplier is used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipelines.forecast_publish.load_forecast_run import create_client  # noqa: E402
from scripts.run_current_direct_alpha_shadow import build_input  # noqa: E402
from scripts.run_direct_alpha_shadow import run_shadow  # noqa: E402
from src.experiments_v2.apply_bakery_profiles import (  # noqa: E402
    DEFAULT_BAKERY_HOUR_PROFILE_PATH,
)
from src.experiments_v2.apply_bakery_profiles_clickhouse import (  # noqa: E402
    allocate_from_clickhouse,
)


START = pd.Timestamp("2026-08-24")
END = pd.Timestamp("2026-08-31")
RAW_UPLIFT_VERSION = "weekly_20260824"
OUTPUT = ROOT / "reports/three_prod_models_august_holdout_20260911"
PILOT_DETAIL = ROOT / ".codex_tmp/pilot_model_version_eval_source/detail.csv"
PILOT_DATE = pd.Timestamp("2026-08-31")
SOURCE_RUNS = {
    pd.Timestamp("2026-08-24"): "prod_base_bakery_norm_recent_20260823_h14",
    pd.Timestamp("2026-08-25"): "prod_base_bakery_norm_recent_20260825_h14",
    pd.Timestamp("2026-08-26"): "prod_base_bakery_norm_recent_20260826_h14",
    pd.Timestamp("2026-08-27"): "prod_base_bakery_norm_recent_20260827_h14",
    pd.Timestamp("2026-08-28"): (
        "prod_assortment7d_v4_base_bakery_norm_recent_20260827_h14"
    ),
    pd.Timestamp("2026-08-29"): "prod_base_bakery_norm_recent_20260829_h14",
    pd.Timestamp("2026-08-30"): "prod_base_bakery_norm_recent_20260830_h14",
    pd.Timestamp("2026-08-31"): "prod_base_bakery_norm_recent_20260830_h14",
}


def pilot_ids() -> list[int]:
    rows = pd.read_csv(PILOT_DETAIL, usecols=["business_date", "bakery_id"])
    rows["business_date"] = pd.to_datetime(rows["business_date"]).dt.normalize()
    values = sorted(
        rows.loc[rows["business_date"].eq(PILOT_DATE), "bakery_id"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    if len(values) != 55:
        raise RuntimeError(f"Expected 55 pilot bakeries, found {len(values)}")
    return values


def export_source(client, bakeries: list[int]) -> tuple[Path, pd.DataFrame]:
    bakery_frames = []
    norm_frames = []
    for date, run_id in SOURCE_RUNS.items():
        bakery_frames.append(
            client.query_df(
                """
                select
                    forecast_date as date,
                    bakery_id,
                    any(bakery_name) as bakery_name,
                    any(city) as city,
                    any(forecast_base) as bakery_day_forecast,
                    any(forecast_final) as bakery_day_forecast_bias_adj
                from bakery_forecast_day_embedded
                where run_id = %(run_id)s
                  and forecast_date = %(date)s
                  and bakery_id in %(bakeries)s
                group by date, bakery_id
                """,
                parameters={
                    "run_id": run_id,
                    "date": date.date(),
                    "bakeries": tuple(bakeries),
                },
            )
        )
        norm_frames.append(
            client.query_df(
                """
                select
                    forecast_date as date,
                    bakery_id,
                    product_id,
                    any(product_name) as product_name,
                    any(category_name) as category_name,
                    sum(forecast_qty) as forecast_qty
                from sku_forecast_day_embedded
                where run_id = %(run_id)s
                  and forecast_date = %(date)s
                  and bakery_id in %(bakeries)s
                group by date, bakery_id, product_id
                """,
                parameters={
                    "run_id": run_id,
                    "date": date.date(),
                    "bakeries": tuple(bakeries),
                },
            )
        )
    bakery = pd.concat(bakery_frames, ignore_index=True)
    bakery["date"] = pd.to_datetime(bakery["date"]).dt.normalize()
    if bakery["bakery_id"].nunique() != 55 or bakery["date"].nunique() != 8:
        raise RuntimeError("Frozen bakery source does not cover the requested scope")
    path = OUTPUT / "bakery_source.csv"
    bakery.to_csv(path, index=False, encoding="utf-8-sig")

    norm = pd.concat(norm_frames, ignore_index=True)
    norm["date"] = pd.to_datetime(norm["date"]).dt.normalize()
    norm["variant"] = "base_bakery_norm_recent"
    norm.to_parquet(OUTPUT / "norm_plan.parquet", index=False)
    return path, norm


def build_direct(bakeries: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for date in pd.date_range(START, END):
        day_dir = OUTPUT / "direct_days" / date.strftime("%Y%m%d")
        result_path = day_dir / "shadow_rows.parquet"
        metadata_path = day_dir / "build_metadata.json"
        expected_run = SOURCE_RUNS[date]
        cached_run = None
        if metadata_path.exists():
            cached_run = json.loads(metadata_path.read_text(encoding="utf-8")).get(
                "source_run_id"
            )
        if not result_path.exists() or cached_run != expected_run:
            input_path, metadata = build_input(
                date, day_dir, source_run_id=expected_run
            )
            run_shadow(input_path, day_dir)
            (day_dir / "build_metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        frame = pd.read_parquet(
            result_path,
            columns=[
                "date",
                "bakery_id",
                "product_id",
                "product_name",
                "category",
                "selected_sku_forecast",
            ],
        ).rename(
            columns={
                "category": "category_name",
                "selected_sku_forecast": "forecast_qty",
            }
        )
        frames.append(frame[frame["bakery_id"].isin(bakeries)])
    direct = pd.concat(frames, ignore_index=True)
    direct["date"] = pd.to_datetime(direct["date"]).dt.normalize()
    direct["variant"] = "direct_alpha_025_v1"
    direct.to_parquet(OUTPUT / "direct_plan.parquet", index=False)
    return direct


def build_raw(
    bakery_path: Path, bakeries: list[int], norm: pd.DataFrame
) -> pd.DataFrame:
    raw_dir = OUTPUT / "raw_reconstruction"
    daily_path = raw_dir / "sku_day_forecast_raw_rolling_source_reconstruction.csv"
    if not daily_path.exists():
        paths = allocate_from_clickhouse(
            bakery_forecast_path=bakery_path,
            bakery_hour_profile_path=DEFAULT_BAKERY_HOUR_PROFILE_PATH,
            output_dir=raw_dir,
            profile_table="sku_hour_share_profile_smoothed_embedded",
            uplift_table="sku_hour_uplift_multiplier_embedded",
            forecast_col="bakery_day_forecast_bias_adj",
            output_suffix="raw_rolling_source_reconstruction",
            use_raw_uplift_multiplier=True,
            uplift_profile_version=RAW_UPLIFT_VERSION,
            recent_correction_mode="runner_city_prior_soft_weekpart",
            recent_correction_days=30,
            recent_sales_table="mart_sales_60d",
            assortment_table="assortment_city_products",
            max_sku_uplift_ratio=1.2,
            stockout_correction_version="stockout_20260716",
            # The raw-run snapshots required to reproduce the historical
            # hierarchical haircut were pruned together with the July uplift
            # multiplier.  Keep the reconstructable raw-uplift architecture
            # and record this limitation in build_metadata.json.
            hierarchical_haircut_target_ratio=None,
            assortment_max_age_days=-1,
            disable_assortment_coverage_guard=True,
            bakery_ids=bakeries,
        )
        daily_path = paths["sku_daily"]
    raw = pd.read_csv(daily_path, encoding="utf-8-sig")
    raw = raw.rename(columns={"sku_day_forecast": "forecast_qty"})
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    raw = raw[raw["bakery_id"].isin(bakeries)].copy()
    raw["variant"] = "base_raw_uplift_reconstructed"
    missing_bakeries = sorted(set(bakeries) - set(raw["bakery_id"].unique()))
    if missing_bakeries:
        fallback = norm[norm["bakery_id"].isin(missing_bakeries)].copy()
        fallback["variant"] = "base_raw_uplift_reconstructed"
        raw = pd.concat([raw, fallback], ignore_index=True, sort=False)
    raw.to_parquet(OUTPUT / "raw_plan.parquet", index=False)
    return raw


def validate_and_save(plans: list[pd.DataFrame], bakeries: list[int]) -> None:
    keys = ["date", "bakery_id", "product_id"]
    records = []
    for plan in plans:
        label = str(plan["variant"].iloc[0])
        duplicate_count = int(plan.duplicated(keys).sum())
        records.append(
            {
                "variant": label,
                "rows": int(len(plan)),
                "dates": int(plan["date"].nunique()),
                "bakeries": int(plan["bakery_id"].nunique()),
                "products": int(plan["product_id"].nunique()),
                "forecast_qty": float(plan["forecast_qty"].sum()),
                "duplicate_keys": duplicate_count,
                "negative_rows": int(plan["forecast_qty"].lt(0).sum()),
            }
        )
        if duplicate_count or plan["forecast_qty"].lt(0).any():
            raise RuntimeError(f"Invalid plan rows for {label}")
        if set(plan["bakery_id"].unique()) != set(bakeries):
            raise RuntimeError(f"Bakery scope mismatch for {label}")
    summary = pd.DataFrame(records)
    summary.to_csv(OUTPUT / "plan_build_summary.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "source_runs_by_date": {
            str(date.date()): run_id for date, run_id in SOURCE_RUNS.items()
        },
        "direct_model_train_through": "2026-08-23",
        "window": {"start": str(START.date()), "end": str(END.date())},
        "pilot_bakeries": bakeries,
        "raw_reconstruction": {
            "exact_historical_multiplier_available": False,
            "missing_version": "pilots_evening_20260716",
            "used_version": RAW_UPLIFT_VERSION,
            "historical_hierarchical_haircut_available": False,
            "cold_start_bakery_fallback": {
                "bakery_ids": [270],
                "strategy": "exact frozen normalized source allocation",
            },
            "note": (
                "Causal architecture reconstruction, not an exact archived output; "
                "the pruned raw-run snapshots also prevent rebuilding its haircut. "
                "Recent-SKU corrections are frozen at the first window date."
            ),
        },
        "database_write": False,
    }
    (OUTPUT / "build_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    bakeries = pilot_ids()
    client = create_client(ROOT / ".env")
    bakery_path, norm = export_source(client, bakeries)
    direct = build_direct(bakeries)
    raw = build_raw(bakery_path, bakeries, norm)
    validate_and_save([norm, direct, raw], bakeries)


if __name__ == "__main__":
    main()
