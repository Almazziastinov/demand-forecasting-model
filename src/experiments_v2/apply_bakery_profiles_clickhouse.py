"""
Apply bakery forecasts to SKU forecasts using ClickHouse-stored profiles.

This is the production-friendly allocation path: the large SKU hour profile is
streamed from ClickHouse instead of being stored as a local CSV on the app VM.
"""

from __future__ import annotations

# ruff: noqa: E402,E501

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipelines.forecast_publish.load_forecast_run import DEFAULT_ENV_PATH
from pipelines.forecast_publish.load_forecast_run import create_client
from pipelines.forecast_publish.sku_hour_profile_store import PROFILE_TABLE
from pipelines.forecast_publish.sku_hour_profile_store import UPLIFT_MULTIPLIER_TABLE
from src.experiments_v2.apply_bakery_profiles import BAKERY_HOUR_FORECAST_COL
from src.experiments_v2.apply_bakery_profiles import BAKERY_ID_COL
from src.experiments_v2.apply_bakery_profiles import BAKERY_FORECAST_COL
from src.experiments_v2.apply_bakery_profiles import DAILY_OUTPUT_NAME
from src.experiments_v2.apply_bakery_profiles import DATE_COL
from src.experiments_v2.apply_bakery_profiles import DOW_COL
from src.experiments_v2.apply_bakery_profiles import DEFAULT_BAKERY_HOUR_PROFILE_PATH
from src.experiments_v2.apply_bakery_profiles import HOUR_COL
from src.experiments_v2.apply_bakery_profiles import HOURLY_OUTPUT_NAME
from src.experiments_v2.apply_bakery_profiles import HOURLY_OUTPUT_COLS
from src.experiments_v2.apply_bakery_profiles import MIN_TIER1_N_DAYS
from src.experiments_v2.apply_bakery_profiles import PRODUCT_ID_COL
from src.experiments_v2.apply_bakery_profiles import SKU_DAY_FORECAST_COL
from src.experiments_v2.apply_bakery_profiles import SKU_HOUR_FORECAST_COL
from src.experiments_v2.apply_bakery_profiles import SKU_PROFILE_CHUNK_SIZE
from src.experiments_v2.apply_bakery_profiles import SKU_SHARE_COL
from src.experiments_v2.apply_bakery_profiles import SUMMARY_OUTPUT_NAME
from src.experiments_v2.apply_bakery_profiles import allocate_bakery_to_hour
from src.experiments_v2.apply_bakery_profiles import build_summary_from_daily
from src.experiments_v2.apply_bakery_profiles import load_bakery_day_forecast
from src.experiments_v2.apply_bakery_profiles import load_bakery_hour_profile


DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed"
SKU_UPLIFT_MULTIPLIER_COL = "sku_uplift_multiplier"


def _write_hourly_chunk(df: pd.DataFrame, path: Path, *, header: bool) -> None:
    df.to_csv(path, mode="a", index=False, encoding="utf-8-sig", header=header)


def load_profile_lookup_frames(client, *, profile_table: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tier1_sums = client.query_df(
        f"""
        select
            bakery_id,
            dow,
            hour,
            sum(mean_sku_share_in_hour_norm) as tier1_share_sum
        from {profile_table}
        where n_days >= {MIN_TIER1_N_DAYS}
        group by bakery_id, dow, hour
        """
    )
    fallback = client.query_df(
        f"""
        select
            bakery_id,
            hour,
            product_id,
            avg(mean_sku_share_in_hour_norm) as mean_sku_share_in_hour_norm
        from {profile_table}
        group by bakery_id, hour, product_id
        """
    )
    fallback_sums = (
        fallback.groupby([BAKERY_ID_COL, HOUR_COL], as_index=False)[SKU_SHARE_COL]
        .sum()
        .rename(columns={SKU_SHARE_COL: "profile_sum"})
    )
    fallback = fallback.merge(fallback_sums, on=[BAKERY_ID_COL, HOUR_COL], how="left")
    fallback[SKU_SHARE_COL] = np.where(
        fallback["profile_sum"] > 0,
        fallback[SKU_SHARE_COL] / fallback["profile_sum"],
        0.0,
    )
    fallback = fallback.drop(columns=["profile_sum"])

    thin_triples = client.query_df(
        f"""
        select distinct bakery_id, dow, hour, 1 as is_thin
        from {profile_table}
        where n_days < {MIN_TIER1_N_DAYS}
        """
    )
    return tier1_sums, fallback, thin_triples


def load_uplift_multipliers(
    client,
    *,
    uplift_table: str,
    profile_version: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    where = ""
    if profile_version:
        safe_version = profile_version.replace("'", "''")
        where = f"where profile_version = '{safe_version}'"

    exact = client.query_df(
        f"""
        select bakery_id, dow, hour, argMax(sku_uplift_multiplier, generated_at) as sku_uplift_multiplier
        from {uplift_table}
        {where}
        {"and" if where else "where"} dow >= 0
        group by bakery_id, dow, hour
        """
    )
    fallback = client.query_df(
        f"""
        select bakery_id, hour, argMax(sku_uplift_multiplier, generated_at) as sku_uplift_multiplier
        from {uplift_table}
        {where}
        {"and" if where else "where"} dow = -1
        group by bakery_id, hour
        """
    )
    return exact, fallback


def apply_multipliers(
    shares: pd.DataFrame,
    multipliers: pd.DataFrame,
    *,
    keys: list[str],
) -> pd.DataFrame:
    if shares.empty or multipliers.empty:
        return shares
    work = shares.merge(multipliers, on=keys, how="left", validate="many_to_one")
    multiplier = pd.to_numeric(
        work[SKU_UPLIFT_MULTIPLIER_COL],
        errors="coerce",
    ).fillna(1.0)
    work[SKU_SHARE_COL] = pd.to_numeric(work[SKU_SHARE_COL], errors="coerce").fillna(0.0) * multiplier
    return work.drop(columns=[SKU_UPLIFT_MULTIPLIER_COL])


def _update_source_stats(stats: dict[str, dict[str, float | int]], df: pd.DataFrame) -> None:
    if df.empty or "source" not in df.columns:
        return
    grouped = (
        df.groupby("source", as_index=False)
        .agg(
            rows=(SKU_HOUR_FORECAST_COL, "size"),
            forecast_sum=(SKU_HOUR_FORECAST_COL, "sum"),
        )
    )
    for row in grouped.to_dict("records"):
        source = str(row["source"])
        if source not in stats:
            stats[source] = {"rows": 0, "forecast_sum": 0.0}
        stats[source]["rows"] = int(stats[source]["rows"]) + int(row["rows"])
        stats[source]["forecast_sum"] = float(stats[source]["forecast_sum"]) + float(
            row["forecast_sum"]
        )


def _finalize_source_stats(stats: dict[str, dict[str, float | int]]) -> list[dict]:
    total_rows = sum(int(v["rows"]) for v in stats.values())
    total_forecast = sum(float(v["forecast_sum"]) for v in stats.values())
    result = []
    for source, values in sorted(stats.items()):
        rows = int(values["rows"])
        forecast_sum = float(values["forecast_sum"])
        result.append(
            {
                "source": source,
                "rows": rows,
                "row_share": round(rows / total_rows, 6) if total_rows else 0.0,
                "forecast_sum": round(forecast_sum, 6),
                "forecast_share": round(forecast_sum / total_forecast, 6)
                if total_forecast
                else 0.0,
            }
        )
    return result


def stream_profile_chunks(client, *, profile_table: str, chunk_size: int):
    query = f"""
        select
            bakery_id,
            dow,
            hour,
            product_id,
            n_days,
            mean_sku_share_in_hour_norm
        from {profile_table}
        where n_days >= {MIN_TIER1_N_DAYS}
        order by bakery_id, dow, hour, product_id
    """
    with client.query_df_stream(query, settings={"max_block_size": chunk_size}) as stream:
        for block in stream:
            if not block.empty:
                yield block


def allocate_from_clickhouse(
    *,
    bakery_forecast_path: str | Path,
    bakery_hour_profile_path: str | Path,
    output_dir: str | Path,
    env_file: str | Path = DEFAULT_ENV_PATH,
    profile_table: str = PROFILE_TABLE,
    uplift_table: str = UPLIFT_MULTIPLIER_TABLE,
    forecast_col: str = BAKERY_FORECAST_COL,
    output_suffix: str = "",
    use_raw_uplift_multiplier: bool = False,
    uplift_profile_version: str | None = None,
    chunk_size: int = SKU_PROFILE_CHUNK_SIZE,
) -> dict[str, Path]:
    client = create_client(env_file)
    bakery_forecast = load_bakery_day_forecast(
        bakery_forecast_path,
        forecast_col=forecast_col,
    )
    bakery_hour_profile = load_bakery_hour_profile(bakery_hour_profile_path)
    hourly_forecast = allocate_bakery_to_hour(bakery_forecast, bakery_hour_profile)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{output_suffix}" if output_suffix else ""
    hourly_path = out_dir / HOURLY_OUTPUT_NAME.replace(".csv", f"{suffix}.csv")
    daily_path = out_dir / DAILY_OUTPUT_NAME.replace(".csv", f"{suffix}.csv")
    summary_path = out_dir / SUMMARY_OUTPUT_NAME.replace(".json", f"{suffix}.json")
    if hourly_path.exists():
        hourly_path.unlink()

    hourly_cols = [DATE_COL, DOW_COL, HOUR_COL, BAKERY_ID_COL, BAKERY_HOUR_FORECAST_COL]
    hourly_lookup = hourly_forecast[hourly_cols].copy()
    hourly_lookup["_row_id"] = np.arange(len(hourly_lookup))

    tier1_sums, fallback, thin_triples = load_profile_lookup_frames(
        client,
        profile_table=profile_table,
    )
    exact_keys = tier1_sums[[BAKERY_ID_COL, DOW_COL, HOUR_COL]].drop_duplicates()
    exact_keys["has_exact"] = 1

    exact_multipliers = pd.DataFrame()
    fallback_multipliers = pd.DataFrame()
    if use_raw_uplift_multiplier:
        exact_multipliers, fallback_multipliers = load_uplift_multipliers(
            client,
            uplift_table=uplift_table,
            profile_version=uplift_profile_version,
        )

    daily_parts: list[pd.DataFrame] = []
    source_stats: dict[str, dict[str, float | int]] = {}
    sku_hour_rows = 0
    products_seen: set[str] = set()
    wrote_header = False

    for i, sku_chunk in enumerate(
        stream_profile_chunks(client, profile_table=profile_table, chunk_size=chunk_size),
        start=1,
    ):
        sku_chunk = sku_chunk.merge(
            tier1_sums,
            on=[BAKERY_ID_COL, DOW_COL, HOUR_COL],
            how="left",
            validate="many_to_one",
        )
        sku_chunk[SKU_SHARE_COL] = (
            pd.to_numeric(sku_chunk[SKU_SHARE_COL], errors="coerce").fillna(0.0)
            / pd.to_numeric(sku_chunk["tier1_share_sum"], errors="coerce").replace(0, np.nan)
        ).fillna(0.0)
        sku_chunk = sku_chunk.drop(columns=["tier1_share_sum"])
        if use_raw_uplift_multiplier:
            sku_chunk = apply_multipliers(
                sku_chunk,
                exact_multipliers,
                keys=[BAKERY_ID_COL, DOW_COL, HOUR_COL],
            )

        merged = hourly_lookup.merge(
            sku_chunk,
            on=[BAKERY_ID_COL, DOW_COL, HOUR_COL],
            how="inner",
            validate="many_to_many",
            sort=False,
        )
        merged["source"] = "exact"
        merged[SKU_HOUR_FORECAST_COL] = (
            merged[BAKERY_HOUR_FORECAST_COL] * merged[SKU_SHARE_COL]
        )
        merged = merged[[*HOURLY_OUTPUT_COLS, "_row_id", "source"]]
        _write_hourly_chunk(merged, hourly_path, header=not wrote_header)
        wrote_header = True
        sku_hour_rows += len(merged)
        _update_source_stats(source_stats, merged)
        products_seen.update(merged[PRODUCT_ID_COL].dropna().astype(str).unique().tolist())
        daily_parts.append(
            merged.groupby(
                [DATE_COL, DOW_COL, BAKERY_ID_COL, PRODUCT_ID_COL],
                as_index=False,
                sort=False,
            ).agg(sku_day_forecast=(SKU_HOUR_FORECAST_COL, "sum"))
        )
        if i % 10 == 0:
            print(f"processed clickhouse profile chunks: {i}", flush=True)

    unmatched = hourly_lookup.merge(
        exact_keys,
        on=[BAKERY_ID_COL, DOW_COL, HOUR_COL],
        how="left",
    )
    unmatched = unmatched[unmatched["has_exact"].isna()].drop(columns=["has_exact"])
    if len(unmatched):
        unmatched = unmatched.merge(
            thin_triples,
            on=[BAKERY_ID_COL, DOW_COL, HOUR_COL],
            how="left",
        )
        fallback_merged = unmatched.merge(
            fallback,
            on=[BAKERY_ID_COL, HOUR_COL],
            how="inner",
            validate="many_to_many",
            sort=False,
        )
        if use_raw_uplift_multiplier:
            fallback_merged = apply_multipliers(
                fallback_merged,
                fallback_multipliers,
                keys=[BAKERY_ID_COL, HOUR_COL],
            )
        fallback_merged[SKU_HOUR_FORECAST_COL] = (
            fallback_merged[BAKERY_HOUR_FORECAST_COL] * fallback_merged[SKU_SHARE_COL]
        )
        fallback_merged["source"] = np.where(
            fallback_merged["is_thin"].fillna(0).astype(int) == 1,
            "bakery_hour_fallback_thin",
            "bakery_hour_fallback",
        )
        fallback_merged = fallback_merged[[*HOURLY_OUTPUT_COLS, "_row_id", "source"]]
        _write_hourly_chunk(fallback_merged, hourly_path, header=not wrote_header)
        sku_hour_rows += len(fallback_merged)
        _update_source_stats(source_stats, fallback_merged)
        products_seen.update(
            fallback_merged[PRODUCT_ID_COL].dropna().astype(str).unique().tolist()
        )
        daily_parts.append(
            fallback_merged.groupby(
                [DATE_COL, DOW_COL, BAKERY_ID_COL, PRODUCT_ID_COL],
                as_index=False,
                sort=False,
            ).agg(sku_day_forecast=(SKU_HOUR_FORECAST_COL, "sum"))
        )

    sku_daily = pd.concat(daily_parts, ignore_index=True)
    sku_daily = (
        sku_daily.groupby(
            [DATE_COL, DOW_COL, BAKERY_ID_COL, PRODUCT_ID_COL],
            as_index=False,
            sort=False,
        )
        .agg(sku_day_forecast=(SKU_DAY_FORECAST_COL, "sum"))
        .sort_values([BAKERY_ID_COL, PRODUCT_ID_COL, DATE_COL])
        .reset_index(drop=True)
    )
    sku_daily.to_csv(daily_path, index=False, encoding="utf-8-sig")

    hourly_final = pd.read_csv(hourly_path, encoding="utf-8-sig")
    hourly_final.drop(columns=["_row_id"], errors="ignore").to_csv(
        hourly_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = build_summary_from_daily(
        bakery_forecast,
        hourly_forecast,
        sku_daily,
        sku_hour_rows=sku_hour_rows,
        products=len(products_seen),
        source_stats=_finalize_source_stats(source_stats),
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"sku_hourly": hourly_path, "sku_daily": daily_path, "summary": summary_path}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply bakery profiles using ClickHouse-stored SKU profiles"
    )
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH))
    parser.add_argument("--bakery-forecast-path", required=True)
    parser.add_argument(
        "--bakery-hour-profile-path",
        default=str(DEFAULT_BAKERY_HOUR_PROFILE_PATH),
    )
    parser.add_argument("--profile-table", default=PROFILE_TABLE)
    parser.add_argument("--uplift-table", default=UPLIFT_MULTIPLIER_TABLE)
    parser.add_argument("--forecast-col", default=BAKERY_FORECAST_COL)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--use-raw-uplift-multiplier", action="store_true")
    parser.add_argument("--uplift-profile-version", default=None)
    parser.add_argument("--chunk-size", type=int, default=SKU_PROFILE_CHUNK_SIZE)
    args = parser.parse_args()

    paths = allocate_from_clickhouse(
        bakery_forecast_path=args.bakery_forecast_path,
        bakery_hour_profile_path=args.bakery_hour_profile_path,
        output_dir=args.output_dir,
        env_file=args.env_file,
        profile_table=args.profile_table,
        uplift_table=args.uplift_table,
        forecast_col=args.forecast_col,
        output_suffix=args.output_suffix,
        use_raw_uplift_multiplier=args.use_raw_uplift_multiplier,
        uplift_profile_version=args.uplift_profile_version,
        chunk_size=args.chunk_size,
    )

    print("=" * 72)
    print("APPLY BAKERY PROFILES FROM CLICKHOUSE")
    print("=" * 72)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
