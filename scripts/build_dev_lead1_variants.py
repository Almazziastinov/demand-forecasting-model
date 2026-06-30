"""
Build three experimental lead-1 backfill variants on dev for 2026-06-22..2026-06-28.

Variants:
  prior14   -- uplifted model, recent_correction_days=14 (shorter prior window)
  bias_corr -- uplifted model, days=30, + post-allocation rolling SKU bias correction
  base_raw  -- base (norm) model, use_raw_uplift_multiplier=True

All runs are loaded to _dev tables as draft (not activated).
Run-id pattern: dev_{variant}_YYYYMMDD_h1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import clickhouse_connect  # noqa: E402
from pipelines.forecast_publish.load_forecast_run import load_forecast_run, get_clickhouse_settings  # noqa: E402
from pipelines.forecast_publish.table_names import table_name  # noqa: E402
from src.experiments_v2.apply_bakery_profiles_clickhouse import allocate_from_clickhouse  # noqa: E402
from src.experiments_v2.bakery_day_forecast import DATE_COL, run_forecast_mode  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_ENV_FILE             = ROOT / ".env"
DEFAULT_OUTPUT_DIR           = ROOT / "data" / "processed"
DEFAULT_WEATHER_PATH         = ROOT / "data" / "processed" / "bakery_weather_features.csv"
DEFAULT_BAKERY_HOUR_PROFILE  = ROOT / "data" / "processed" / "bakery_hour_profile.csv"

# uplifted variant
UPLIFTED_DATASET = ROOT / "data" / "processed" / "bakery_daily_sales_uplifted.csv"
UPLIFTED_MODEL   = ROOT / "models" / "bakery_day_model_uplifted.joblib"
UPLIFTED_META    = ROOT / "models" / "bakery_day_meta_uplifted.joblib"
UPLIFTED_BIAS    = ROOT / "models" / "bakery_day_bias_uplifted.json"

# base (norm) variant
BASE_DATASET     = ROOT / "data" / "processed" / "bakery_daily_sales.csv"
BASE_MODEL       = ROOT / "models" / "bakery_day_model.joblib"
BASE_META        = ROOT / "models" / "bakery_day_meta.joblib"
BASE_BIAS        = ROOT / "models" / "bakery_day_bias.json"
BASE_BIAS_FALLBACK = ROOT / "reports" / "bakery_day_model_bias_by_bakery.csv"

# ClickHouse tables
PROFILE_TABLE    = "sku_hour_share_profile_smoothed_embedded"
UPLIFT_TABLE     = "sku_hour_uplift_multiplier_embedded"
ASSORTMENT_TABLE = "assortment_city_products"
SALES_TABLE      = "mart_sales_60d"
TABLE_SUFFIX     = "_dev"

# Uplift profile version (dev)
UPLIFT_PROFILE_VERSION = "prod_allowlist_22_222_old_else_20260617"

# Pilot bakeries — only these are evaluated, limits ClickHouse profile streaming
PILOT_BAKERY_IDS = [20, 21, 22, 28, 80, 89, 107, 221, 222, 257]

# Top-5 ходовых SKU (for bias_corr variant)
TOP5_SKU_IDS = [1071, 10340, 205, 1076, 57]
# Lookback window for rolling bias correction (days before forecast_date)
BIAS_CORRECTION_LOOKBACK = 14

# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------
VARIANTS = {
    "prior14": {
        "description": "uplifted model, prior window 14d (vs 30d prod)",
        "dataset": UPLIFTED_DATASET,
        "model":   UPLIFTED_MODEL,
        "meta":    UPLIFTED_META,
        "bias":    UPLIFTED_BIAS,
        "use_raw_uplift": False,
        "recent_correction_days": 14,
        "apply_sku_bias_correction": False,
        "model_version": "bakery_day_lgbm_uplifted",
    },
    "bias_corr": {
        "description": "uplifted model, days=30, + rolling SKU bias correction top-5",
        "dataset": UPLIFTED_DATASET,
        "model":   UPLIFTED_MODEL,
        "meta":    UPLIFTED_META,
        "bias":    UPLIFTED_BIAS,
        "use_raw_uplift": False,
        "recent_correction_days": 30,
        "apply_sku_bias_correction": True,
        "model_version": "bakery_day_lgbm_uplifted",
    },
    "base_raw": {
        "description": "base (norm) model + raw uplift on allocation",
        "dataset": BASE_DATASET,
        "model":   BASE_MODEL,
        "meta":    BASE_META,
        "bias":    BASE_BIAS if BASE_BIAS.exists() else BASE_BIAS_FALLBACK,
        "use_raw_uplift": True,
        "recent_correction_days": 30,
        "apply_sku_bias_correction": False,
        "model_version": "bakery_day_lgbm_base",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def daterange(start: str, end: str):
    current = date.fromisoformat(start)
    stop = date.fromisoformat(end)
    while current <= stop:
        yield str(current)
        current += timedelta(days=1)


def slice_history(dataset_path: Path, forecast_date: str, output_path: Path) -> int:
    df = pd.read_csv(dataset_path, encoding="utf-8-sig")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    history = df[df[DATE_COL] < pd.Timestamp(forecast_date)].copy()
    if history.empty:
        raise ValueError(f"No history before {forecast_date} in {dataset_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(output_path, index=False, encoding="utf-8-sig")
    return len(history)


def compute_rolling_bias_correction(
    client,
    forecast_date: str,
    sku_ids: list[int],
    lookback_days: int,
) -> dict[int, float]:
    """
    Compute mean(fact/forecast) for each SKU over the [date-lookback, date-1] window.
    Returns a dict {product_id: correction_factor}.
    Correction factor < 1 means model overforecasts -> scale down.
    Skips SKUs where data is insufficient or ratio is within [0.9, 1.1].
    """
    end = (date.fromisoformat(forecast_date) - timedelta(days=1)).isoformat()
    start = (date.fromisoformat(forecast_date) - timedelta(days=lookback_days)).isoformat()

    # Load lead-1 forecasts for these SKUs in the lookback window (all bakeries)
    fc_df = client.query_df(
        """
        select
            forecast_date,
            toInt64(bakery_id)  as bakery_id,
            toInt64(product_id) as product_id,
            sum(forecast_qty)   as forecast_qty
        from sku_forecast_day_snapshots
        where lead_days = 1
          and toInt64(product_id) in %(skus)s
          and forecast_date between %(start)s and %(end)s
        group by forecast_date, bakery_id, product_id
        """,
        parameters={"skus": sku_ids, "start": start, "end": end},
    )

    # Load actuals for the same window
    act_df = client.query_df(
        """
        select
            check_date          as forecast_date,
            toInt64(bakery_id)  as bakery_id,
            toInt64(product_id) as product_id,
            sum(quantity)       as actual_qty
        from mart_sales_60d
        where toInt64(product_id) in %(skus)s
          and check_date between %(start)s and %(end)s
        group by forecast_date, bakery_id, product_id
        """,
        parameters={"skus": sku_ids, "start": start, "end": end},
    )

    if fc_df.empty or act_df.empty:
        print(f"  [bias_corr] No data for rolling correction (start={start} end={end})")
        return {}

    merged = act_df.merge(fc_df, on=["forecast_date", "bakery_id", "product_id"], how="inner")
    merged = merged[(merged["actual_qty"] > 0) & (merged["forecast_qty"] > 0)]
    if merged.empty:
        return {}

    merged["ratio"] = merged["actual_qty"] / merged["forecast_qty"]

    corrections = {}
    for sku_id in sku_ids:
        sub = merged[merged["product_id"] == sku_id]["ratio"]
        if len(sub) < 3:
            print(f"  [bias_corr] SKU {sku_id}: too few rows ({len(sub)}), skip")
            continue
        factor = float(sub.mean())
        if 0.90 <= factor <= 1.10:
            print(f"  [bias_corr] SKU {sku_id}: factor={factor:.3f} within [0.9,1.1], skip")
            continue
        # Clip to prevent extreme corrections
        factor = float(np.clip(factor, 0.5, 1.5))
        print(f"  [bias_corr] SKU {sku_id}: correction factor={factor:.3f} "
              f"(n={len(sub)}, mean ratio={sub.mean():.3f})")
        corrections[sku_id] = factor

    return corrections


def apply_sku_bias_correction(
    sku_daily_path: Path,
    sku_hourly_path: Path,
    corrections: dict[int, float],
) -> None:
    """Apply per-SKU multiplicative correction to the daily and hourly CSV files."""
    if not corrections:
        return

    # Daily
    daily = pd.read_csv(sku_daily_path, encoding="utf-8-sig")
    daily["product_id"] = pd.to_numeric(daily["product_id"], errors="coerce").astype("Int64")
    for sku_id, factor in corrections.items():
        mask = daily["product_id"] == sku_id
        daily.loc[mask, "sku_day_forecast"] = daily.loc[mask, "sku_day_forecast"] * factor
    daily.to_csv(sku_daily_path, index=False, encoding="utf-8-sig")

    # Hourly
    hourly = pd.read_csv(sku_hourly_path, encoding="utf-8-sig")
    hourly["product_id"] = pd.to_numeric(hourly["product_id"], errors="coerce").astype("Int64")
    for sku_id, factor in corrections.items():
        mask = hourly["product_id"] == sku_id
        hourly.loc[mask, "sku_hour_forecast"] = hourly.loc[mask, "sku_hour_forecast"] * factor
    hourly.to_csv(sku_hourly_path, index=False, encoding="utf-8-sig")

    total_correction = sum(corrections.values()) / len(corrections)
    print(f"  [bias_corr] Applied corrections to {len(corrections)} SKUs "
          f"(avg factor={total_correction:.3f})")


# ---------------------------------------------------------------------------
# Core: build one day, one variant
# ---------------------------------------------------------------------------

def _make_client(env_file: str):
    """Create a fresh ClickHouse client with high timeouts for large profile queries."""
    ch_settings = get_clickhouse_settings(env_file)
    return clickhouse_connect.get_client(
        host=ch_settings["host"],
        port=int(ch_settings["port"]),
        username=ch_settings["username"],
        password=ch_settings["password"],
        database=ch_settings["database"],
        secure=bool(ch_settings["secure"]),
        verify=bool(ch_settings["verify"]),
        send_receive_timeout=600,
        connect_timeout=30,
        # Tell server not to enforce execution time limit on heavy profile queries
        settings={"max_execution_time": 0},
    )


def build_day_variant(
    env_file: str,
    forecast_date: str,
    variant_name: str,
    variant: dict,
    output_dir: Path,
    weather_path: str,
    bakery_hour_profile_path: str,
) -> dict:
    # Fresh client per run — avoids stale connection state after errors
    client = _make_client(env_file)

    date_part = forecast_date.replace("-", "")
    run_id = f"dev_{variant_name}_{date_part}_h1"

    # Paths
    history_path = output_dir / f"_dev_history_{variant_name}_{date_part}.csv"
    bakery_path  = output_dir / f"bakery_dev_{variant_name}_{date_part}.csv"

    print(f"\n  Slicing history before {forecast_date}...")
    n_rows = slice_history(variant["dataset"], forecast_date, history_path)
    print(f"  History rows: {n_rows:,}")

    print(f"  Running bakery-day forecast...")
    run_forecast_mode(
        argparse.Namespace(
            dataset_path=str(history_path),
            model_path=str(variant["model"]),
            meta_path=str(variant["meta"]),
            bias_path=str(variant["bias"]),
            output_path=str(bakery_path),
            weather_path=weather_path,
            horizon_days=1,
            start_date=forecast_date,
            apply_bias_correction=True,
            bias_clip_pct=0.15,
        )
    )

    # Verify bakery forecast was created
    if not bakery_path.exists():
        raise RuntimeError(f"Bakery forecast not created: {bakery_path}")
    bak_df = pd.read_csv(bakery_path, encoding="utf-8-sig")
    # Filter to pilot bakeries only — reduces ClickHouse profile streaming from ~220 to ~10 chunks
    bak_id_col = next((c for c in bak_df.columns if "bakery_id" in c.lower()), None)
    if bak_id_col:
        before = len(bak_df)
        bak_df = bak_df[pd.to_numeric(bak_df[bak_id_col], errors="coerce").isin(PILOT_BAKERY_IDS)]
        bak_df.to_csv(bakery_path, index=False, encoding="utf-8-sig")
        print(f"  Filtered bakery forecast to pilot: {before} -> {len(bak_df)} rows")
    print(f"  Bakery forecast: {len(bak_df)} rows, "
          f"sum={bak_df.get('bakery_day_forecast_bias_adj', bak_df.iloc[:,0]).sum():.1f}")

    print(f"  Allocating to SKU/hour (use_raw_uplift={variant['use_raw_uplift']}, "
          f"recent_days={variant['recent_correction_days']})...")

    # Retry allocation up to 3 times — YC LB can close HTTP connections on heavy queries
    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            if attempt > 1:
                wait = 30 * attempt
                print(f"  Allocation attempt {attempt}/3, waiting {wait}s before retry...")
                import time as _time; _time.sleep(wait)
            allocated = allocate_from_clickhouse(
                bakery_forecast_path=bakery_path,
                bakery_hour_profile_path=bakery_hour_profile_path,
                output_dir=output_dir,
                env_file=env_file,
                # Let allocate_from_clickhouse create its own client (proved to work in first run)
                profile_table=table_name(PROFILE_TABLE, TABLE_SUFFIX),
                uplift_table=table_name(UPLIFT_TABLE, TABLE_SUFFIX),
                forecast_col="bakery_day_forecast_bias_adj",
                output_suffix=f"dev_{variant_name}_{date_part}",
                use_raw_uplift_multiplier=variant["use_raw_uplift"],
                uplift_profile_version=UPLIFT_PROFILE_VERSION,
                recent_correction_mode="runner_city_prior_soft_weekpart",
                recent_correction_days=variant["recent_correction_days"],
                recent_sales_table=SALES_TABLE,
                assortment_table=table_name(ASSORTMENT_TABLE, TABLE_SUFFIX),
            )
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            print(f"  Allocation attempt {attempt} failed: {exc}")
    if last_exc is not None:
        raise last_exc

    # Post-allocation SKU bias correction (variant B only)
    if variant["apply_sku_bias_correction"]:
        print(f"  Computing rolling bias correction (lookback={BIAS_CORRECTION_LOOKBACK}d)...")
        bias_client = _make_client(env_file)
        corrections = compute_rolling_bias_correction(
            bias_client, forecast_date, TOP5_SKU_IDS, BIAS_CORRECTION_LOOKBACK,
        )
        if corrections:
            apply_sku_bias_correction(
                allocated["sku_daily"],
                allocated["sku_hourly"],
                corrections,
            )

    print(f"  Loading to dev ClickHouse (run_id={run_id})...")
    loaded = load_forecast_run(
        env_file=env_file,
        bakery_path=bakery_path,
        sku_day_path=allocated["sku_daily"],
        sku_hour_path=allocated["sku_hourly"],
        profile_table=table_name(PROFILE_TABLE, TABLE_SUFFIX),
        lookup_source="clickhouse",
        run_id=run_id,
        model_version=variant["model_version"],
        profile_version=UPLIFT_PROFILE_VERSION,
        notes=f"Dev variant={variant_name} lead1 for {forecast_date}: {variant['description']}",
        replace_existing=True,
        weather_path=weather_path,
    )

    # Cleanup temp history file
    history_path.unlink(missing_ok=True)

    return {
        "date": forecast_date,
        "variant": variant_name,
        "run_id": run_id,
        "bakery_rows": loaded.get("bakery_rows", 0),
        "sku_day_rows": loaded.get("sku_day_rows", 0),
        "sku_hour_rows": loaded.get("sku_hour_rows", 0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build 3 experimental lead-1 variants on dev for a date range."
    )
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--date-from", default="2026-06-22")
    parser.add_argument("--date-to",   default="2026-06-28")
    parser.add_argument("--variants",  nargs="+",
                        choices=list(VARIANTS.keys()),
                        default=list(VARIANTS.keys()),
                        help="Which variants to run (default: all)")
    parser.add_argument("--weather-path",            default=str(DEFAULT_WEATHER_PATH))
    parser.add_argument("--bakery-hour-profile-path", default=str(DEFAULT_BAKERY_HOUR_PROFILE))
    parser.add_argument("--output-dir",              default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--summary-path",
                        default="reports/dev_lead1_variants_summary.json")
    args = parser.parse_args()

    dates = list(daterange(args.date_from, args.date_to))
    selected_variants = {k: VARIANTS[k] for k in args.variants}

    print("=" * 72)
    print("DEV LEAD-1 VARIANT BACKFILL")
    print("=" * 72)
    print(f"Dates:    {args.date_from} .. {args.date_to} ({len(dates)} days)")
    print(f"Variants: {list(selected_variants.keys())}")
    print(f"Total:    {len(dates) * len(selected_variants)} runs")
    print()
    for vname, vcfg in selected_variants.items():
        print(f"  [{vname}] {vcfg['description']}")
    print()

    # Use high timeout for streaming large SKU profiles from ClickHouse
    ch_settings = get_clickhouse_settings(args.env_file)
    client = clickhouse_connect.get_client(
        host=ch_settings["host"],
        port=int(ch_settings["port"]),
        username=ch_settings["username"],
        password=ch_settings["password"],
        database=ch_settings["database"],
        secure=bool(ch_settings["secure"]),
        verify=bool(ch_settings["verify"]),
        send_receive_timeout=600,
        connect_timeout=30,
    )
    output_dir = Path(args.output_dir)

    results = []
    total_started = time.monotonic()
    failed = []

    for forecast_date in dates:
        for variant_name, variant_cfg in selected_variants.items():
            run_label = f"{variant_name} / {forecast_date}"
            print(f"\n{'=' * 72}")
            print(f"Building: {run_label}")
            print(f"{'=' * 72}")
            t0 = time.monotonic()
            try:
                result = build_day_variant(
                    env_file=args.env_file,
                    forecast_date=forecast_date,
                    variant_name=variant_name,
                    variant=variant_cfg,
                    output_dir=output_dir,
                    weather_path=args.weather_path,
                    bakery_hour_profile_path=args.bakery_hour_profile_path,
                )
                elapsed = round(time.monotonic() - t0, 1)
                result["elapsed_sec"] = elapsed
                results.append(result)
                print(
                    f"DONE {run_label} in {elapsed}s | "
                    f"bakery={result['bakery_rows']} "
                    f"sku_day={result['sku_day_rows']} "
                    f"sku_hour={result['sku_hour_rows']}"
                )
            except Exception as exc:
                elapsed = round(time.monotonic() - t0, 1)
                print(f"FAILED {run_label} in {elapsed}s: {exc}")
                failed.append({"date": forecast_date, "variant": variant_name, "error": str(exc)})

    total_elapsed = round(time.monotonic() - total_started, 1)

    # Summary
    summary = {
        "date_from": args.date_from,
        "date_to": args.date_to,
        "variants": list(selected_variants.keys()),
        "total_elapsed_sec": total_elapsed,
        "success": results,
        "failed": failed,
    }
    summary_path = ROOT / args.summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print("COMPLETE")
    print("=" * 72)
    print(f"Total time: {total_elapsed:.0f}s")
    print(f"Succeeded:  {len(results)}")
    print(f"Failed:     {len(failed)}")
    if failed:
        print("\nFailed runs:")
        for f in failed:
            print(f"  {f['variant']} / {f['date']}: {f['error']}")
    print(f"\nSummary: {summary_path}")

    # Print table of results
    if results:
        print("\nLoaded runs:")
        df = pd.DataFrame(results)
        print(df[["variant","date","run_id","bakery_rows","sku_day_rows","elapsed_sec"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
