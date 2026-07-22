"""Compare observed-sales and demand-adjusted SKU-hour profiles offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments_v2.build_sku_hour_share_profile import (  # noqa: E402
    BAKERY_HOUR_SALES_COL,
    BAKERY_ID_COL,
    BAKERY_NAME_COL,
    CATEGORY_COL,
    CITY_COL,
    DATE_COL,
    DOW_COL,
    HOUR_COL,
    PRODUCT_ID_COL,
    PRODUCT_NAME_COL,
    SKU_HOUR_SALES_COL,
    aggregate_sku_hourly_chunk,
    build_sku_hour_share_profile,
    merge_hourly_parts,
)
from src.experiments_v2.apply_bakery_profiles import (  # noqa: E402
    MIN_TIER1_N_DAYS,
    build_sku_hour_profile_fallback,
)

DEFAULT_RAW = ROOT / "data/raw/pilot_stg_check_lines_2026-04-30_2026-07-19.csv"
DEFAULT_ADJUSTMENTS = (
    ROOT / "reports/demand_adjusted_stockout_history/hourly_adjustments.csv"
)
DEFAULT_STOCKOUTS = (
    ROOT
    / "reports/pilot_stockout_responsibility/stockout_cases_classified.csv"
)
DEFAULT_OUTPUT = ROOT / "reports/demand_adjusted_profile_experiment"
DEFAULT_TRAIN_END = pd.Timestamp("2026-07-05")
DEFAULT_HOLDOUT_START = pd.Timestamp("2026-07-06")
DEFAULT_HOLDOUT_END = pd.Timestamp("2026-07-19")
PROFILE_SHARE_COL = "mean_sku_share_in_hour_norm"
PROFILE_KEYS = [BAKERY_ID_COL, PRODUCT_ID_COL, DOW_COL, HOUR_COL]
HOUR_KEYS = [DATE_COL, BAKERY_ID_COL, PRODUCT_ID_COL, HOUR_COL]


def load_hourly_raw(
    path: Path,
    *,
    bakery_ids: set[int],
    date_from: pd.Timestamp,
    date_to: pd.Timestamp,
    chunk_size: int = 500_000,
) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        bakery = pd.to_numeric(chunk["bakery_id"], errors="coerce")
        dates = pd.to_datetime(chunk["check_date"], errors="coerce")
        chunk = chunk[
            bakery.isin(bakery_ids) & dates.between(date_from, date_to)
        ]
        if not chunk.empty:
            part = aggregate_sku_hourly_chunk(chunk)
            if not part.empty:
                parts.append(part)
    return merge_hourly_parts(parts)


def apply_hourly_adjustments(
    hourly: pd.DataFrame, adjustments: pd.DataFrame
) -> pd.DataFrame:
    """Add imputed demand, creating SKU-hour rows when observed sales were zero."""
    work = hourly.copy()
    adjustment = adjustments.copy()
    adjustment[DATE_COL] = pd.to_datetime(adjustment[DATE_COL]).dt.normalize()
    adjustment = adjustment[
        pd.to_numeric(adjustment["imputed_demand"], errors="coerce").fillna(0).gt(0)
    ]
    adjustment = adjustment.groupby(HOUR_KEYS, as_index=False)[
        "imputed_demand"
    ].sum()
    if adjustment.empty:
        return work

    metadata = (
        work.sort_values(DATE_COL)
        .groupby([BAKERY_ID_COL, PRODUCT_ID_COL], as_index=False)
        .agg(
            **{
                BAKERY_NAME_COL: (BAKERY_NAME_COL, "last"),
                CITY_COL: (CITY_COL, "last"),
                PRODUCT_NAME_COL: (PRODUCT_NAME_COL, "last"),
                CATEGORY_COL: (CATEGORY_COL, "last"),
            }
        )
    )
    merged = work.merge(adjustment, on=HOUR_KEYS, how="left")
    merged["imputed_demand"] = merged["imputed_demand"].fillna(0.0)
    merged[SKU_HOUR_SALES_COL] = (
        merged[SKU_HOUR_SALES_COL] + merged["imputed_demand"]
    )

    missing = adjustment.merge(
        work[HOUR_KEYS].drop_duplicates(),
        on=HOUR_KEYS,
        how="left",
        indicator=True,
    )
    missing = missing[missing["_merge"].eq("left_only")].drop(columns="_merge")
    if not missing.empty:
        missing = missing.merge(
            metadata,
            on=[BAKERY_ID_COL, PRODUCT_ID_COL],
            how="left",
            validate="many_to_one",
        )
        if missing[
            [BAKERY_NAME_COL, CITY_COL, PRODUCT_NAME_COL, CATEGORY_COL]
        ].isna().any(axis=None):
            raise ValueError("Missing metadata for synthetic demand-adjusted rows")
        missing[DOW_COL] = missing[DATE_COL].dt.dayofweek
        missing[SKU_HOUR_SALES_COL] = missing["imputed_demand"]
        merged = pd.concat([merged, missing[merged.columns]], ignore_index=True)
    return merged.sort_values(
        [BAKERY_ID_COL, PRODUCT_ID_COL, DATE_COL, HOUR_COL]
    ).reset_index(drop=True)


def compact_profile(
    profile: pd.DataFrame, *, min_n_days: int | None = None
) -> pd.DataFrame:
    compact = profile.groupby(PROFILE_KEYS, as_index=False).agg(
        profile_share=(PROFILE_SHARE_COL, "sum"),
        profile_n_days=("n_days", "max"),
    )
    if min_n_days is not None:
        compact = compact[compact["profile_n_days"].ge(min_n_days)].copy()
    totals = compact.groupby([BAKERY_ID_COL, DOW_COL, HOUR_COL])[
        "profile_share"
    ].transform("sum")
    compact["profile_share"] = np.where(
        totals > 0, compact["profile_share"] / totals, 0.0
    )
    return compact


def blend_profiles(
    baseline: pd.DataFrame,
    adjusted: pd.DataFrame,
    *,
    adjusted_weight: float,
) -> pd.DataFrame:
    """Blend serving shares while retaining adjusted membership evidence."""
    if not 0.0 <= adjusted_weight <= 1.0:
        raise ValueError("adjusted_weight must be between 0 and 1")
    left = compact_profile(baseline).rename(
        columns={
            "profile_share": "baseline_share",
            "profile_n_days": "baseline_n_days",
        }
    )
    right = compact_profile(adjusted).rename(
        columns={
            "profile_share": "adjusted_share",
            "profile_n_days": "adjusted_n_days",
        }
    )
    blended = left.merge(right, on=PROFILE_KEYS, how="outer")
    blended[["baseline_share", "adjusted_share"]] = blended[
        ["baseline_share", "adjusted_share"]
    ].fillna(0.0)
    blended[PROFILE_SHARE_COL] = (
        (1.0 - adjusted_weight) * blended["baseline_share"]
        + adjusted_weight * blended["adjusted_share"]
    )
    blended["n_days"] = blended["adjusted_n_days"].fillna(
        blended["baseline_n_days"]
    )
    totals = blended.groupby([BAKERY_ID_COL, DOW_COL, HOUR_COL])[
        PROFILE_SHARE_COL
    ].transform("sum")
    blended[PROFILE_SHARE_COL] = np.where(
        totals > 0,
        blended[PROFILE_SHARE_COL] / totals,
        0.0,
    )
    return blended[[*PROFILE_KEYS, PROFILE_SHARE_COL, "n_days"]]


def select_adjusted_contexts(
    baseline: pd.DataFrame,
    adjusted: pd.DataFrame,
    adjusted_triples: set[tuple[int, int, int]],
) -> pd.DataFrame:
    """Use adjusted history only for selected bakery/dow/hour contexts."""
    triple_cols = [BAKERY_ID_COL, DOW_COL, HOUR_COL]
    baseline_mask = [
        tuple(map(int, values)) not in adjusted_triples
        for values in baseline[triple_cols].to_numpy()
    ]
    adjusted_mask = [
        tuple(map(int, values)) in adjusted_triples
        for values in adjusted[triple_cols].to_numpy()
    ]
    return pd.concat(
        [baseline.loc[baseline_mask], adjusted.loc[adjusted_mask]],
        ignore_index=True,
    )


def build_membership_seed_profile(
    baseline: pd.DataFrame,
    adjusted: pd.DataFrame,
    new_exact_rows: set[tuple[int, int, int, int]],
    *,
    seed_weight: float,
) -> pd.DataFrame:
    """Promote reconstructed tier-1 rows with a controlled initial share."""
    if not 0.0 <= seed_weight <= 1.0:
        raise ValueError("seed_weight must be between 0 and 1")
    baseline_mask = [
        tuple(map(int, values)) not in new_exact_rows
        for values in baseline[PROFILE_KEYS].to_numpy()
    ]
    promoted_mask = [
        tuple(map(int, values)) in new_exact_rows
        for values in adjusted[PROFILE_KEYS].to_numpy()
    ]
    promoted = adjusted.loc[promoted_mask].copy()
    promoted[PROFILE_SHARE_COL] = promoted[PROFILE_SHARE_COL] * seed_weight
    return pd.concat([baseline.loc[baseline_mask], promoted], ignore_index=True)


def build_serving_profiles(
    profile: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exact = compact_profile(profile, min_n_days=MIN_TIER1_N_DAYS)
    if exact.empty:
        exact = pd.DataFrame(columns=[*PROFILE_KEYS, "profile_share", "profile_n_days"])
    fallback = build_sku_hour_profile_fallback(
        profile,
        normalize_sku_shares=True,
    ).rename(columns={PROFILE_SHARE_COL: "profile_share"})
    fallback = fallback.groupby(
        [BAKERY_ID_COL, HOUR_COL, PRODUCT_ID_COL], as_index=False
    )["profile_share"].sum()
    totals = fallback.groupby([BAKERY_ID_COL, HOUR_COL])[
        "profile_share"
    ].transform("sum")
    fallback["profile_share"] = np.where(
        totals > 0, fallback["profile_share"] / totals, 0.0
    )
    return exact, fallback


def build_scored_rows(
    profile: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    allowed_exact_triples: set[tuple[int, int, int]] | None = None,
    fallback_source_profile: pd.DataFrame | None = None,
) -> pd.DataFrame:
    actual = holdout.groupby(HOUR_KEYS + [DOW_COL], as_index=False).agg(
        actual_qty=(SKU_HOUR_SALES_COL, "sum")
    )
    bakery_hour = actual.groupby(
        [DATE_COL, BAKERY_ID_COL, DOW_COL, HOUR_COL], as_index=False
    )["actual_qty"].sum().rename(columns={"actual_qty": BAKERY_HOUR_SALES_COL})
    contexts = bakery_hour[
        [DATE_COL, BAKERY_ID_COL, DOW_COL, HOUR_COL, BAKERY_HOUR_SALES_COL]
    ]
    exact, fallback = build_serving_profiles(profile)
    if fallback_source_profile is not None:
        _, fallback = build_serving_profiles(fallback_source_profile)
    if allowed_exact_triples is not None:
        allowed_mask = [
                (int(row.bakery_id), int(row.dow), int(row.hour))
                in allowed_exact_triples
                for row in exact.itertuples()
        ]
        exact = exact.loc[allowed_mask].copy()
    exact_triples = exact[
        [BAKERY_ID_COL, DOW_COL, HOUR_COL]
    ].drop_duplicates().assign(has_exact=True)
    context_routing = contexts.merge(
        exact_triples,
        on=[BAKERY_ID_COL, DOW_COL, HOUR_COL],
        how="left",
    )
    exact_contexts = context_routing[context_routing["has_exact"].eq(True)].drop(
        columns="has_exact"
    )
    fallback_contexts = context_routing[
        context_routing["has_exact"].isna()
    ].drop(columns="has_exact")
    predicted = pd.concat(
        [
            exact_contexts.merge(
                exact,
                on=[BAKERY_ID_COL, DOW_COL, HOUR_COL],
                how="inner",
            ),
            fallback_contexts.merge(
                fallback,
                on=[BAKERY_ID_COL, HOUR_COL],
                how="inner",
            ),
        ],
        ignore_index=True,
    )
    scored = predicted.merge(
        actual,
        on=HOUR_KEYS + [DOW_COL],
        how="outer",
    )
    # A context without an exact profile produces one left-merge placeholder
    # with no product. Actual SKU rows are retained by the outer merge below,
    # so the placeholder must not become an evaluation row.
    scored = scored[scored[PRODUCT_ID_COL].notna()].copy()
    scored[PRODUCT_ID_COL] = scored[PRODUCT_ID_COL].astype(int)
    scored = scored.merge(
        bakery_hour,
        on=[DATE_COL, BAKERY_ID_COL, DOW_COL, HOUR_COL],
        how="left",
        suffixes=("", "_actual"),
    )
    if f"{BAKERY_HOUR_SALES_COL}_actual" in scored:
        scored[BAKERY_HOUR_SALES_COL] = scored[BAKERY_HOUR_SALES_COL].fillna(
            scored[f"{BAKERY_HOUR_SALES_COL}_actual"]
        )
        scored = scored.drop(columns=f"{BAKERY_HOUR_SALES_COL}_actual")
    scored[["profile_share", "actual_qty"]] = scored[
        ["profile_share", "actual_qty"]
    ].fillna(0.0)
    scored["predicted_qty"] = (
        scored["profile_share"] * scored[BAKERY_HOUR_SALES_COL]
    )
    scored["error"] = scored["predicted_qty"] - scored["actual_qty"]
    scored["abs_error"] = scored["error"].abs()
    return scored


def attach_evaluation_scopes(
    scored: pd.DataFrame,
    stockouts: pd.DataFrame,
    adjusted_pairs: set[tuple[int, int]],
    new_exact_triples: set[tuple[int, int, int]],
    new_tier1_member_triples: set[tuple[int, int, int]],
) -> pd.DataFrame:
    result = scored.copy()
    stockout_sku_keys = set(
        map(
            tuple,
            stockouts[[DATE_COL, BAKERY_ID_COL, PRODUCT_ID_COL]].to_numpy(),
        )
    )
    stockout_bakery_keys = set(
        map(tuple, stockouts[[DATE_COL, BAKERY_ID_COL]].drop_duplicates().to_numpy())
    )
    result["is_stockout_sku_day"] = [
        (row.date, int(row.bakery_id), int(row.product_id)) in stockout_sku_keys
        for row in result.itertuples()
    ]
    result["is_stockout_bakery_day"] = [
        (row.date, int(row.bakery_id)) in stockout_bakery_keys
        for row in result.itertuples()
    ]
    result["is_adjusted_pair"] = [
        (int(row.bakery_id), int(row.product_id)) in adjusted_pairs
        for row in result.itertuples()
    ]
    result["is_new_exact_routing"] = [
        (int(row.bakery_id), int(row.dow), int(row.hour)) in new_exact_triples
        for row in result.itertuples()
    ]
    result["has_new_tier1_member"] = [
        (int(row.bakery_id), int(row.dow), int(row.hour))
        in new_tier1_member_triples
        for row in result.itertuples()
    ]
    return result


def summarize_scores(scored: pd.DataFrame, *, variant: str) -> pd.DataFrame:
    scopes = {
        "all_holdout": pd.Series(True, index=scored.index),
        "clean_bakery_days": ~scored["is_stockout_bakery_day"],
        "clean_sku_days": ~scored["is_stockout_sku_day"],
        "adjusted_pairs_clean_sku_days": (
            scored["is_adjusted_pair"] & ~scored["is_stockout_sku_day"]
        ),
        "stable_routing_clean_sku_days": (
            ~scored["is_new_exact_routing"] & ~scored["is_stockout_sku_day"]
        ),
        "new_exact_routing_clean_sku_days": (
            scored["is_new_exact_routing"] & ~scored["is_stockout_sku_day"]
        ),
        "stable_tier1_membership_clean_sku_days": (
            ~scored["has_new_tier1_member"] & ~scored["is_stockout_sku_day"]
        ),
        "new_tier1_member_clean_sku_days": (
            scored["has_new_tier1_member"] & ~scored["is_stockout_sku_day"]
        ),
    }
    rows = []
    for scope, mask in scopes.items():
        group = scored[mask]
        actual = float(group["actual_qty"].sum())
        predicted = float(group["predicted_qty"].sum())
        sku_day = group.groupby(
            [DATE_COL, BAKERY_ID_COL, PRODUCT_ID_COL], as_index=False
        ).agg(
            actual_qty=("actual_qty", "sum"),
            predicted_qty=("predicted_qty", "sum"),
        )
        sku_day["error"] = sku_day["predicted_qty"] - sku_day["actual_qty"]
        sku_day["abs_error"] = sku_day["error"].abs()
        rows.append(
            {
                "variant": variant,
                "scope": scope,
                "rows": int(len(group)),
                "actual_qty": actual,
                "predicted_qty": predicted,
                "bias_qty": predicted - actual,
                "mae": float(group["abs_error"].mean()) if len(group) else None,
                "wape": (
                    float(group["abs_error"].sum() / actual)
                    if actual > 0
                    else None
                ),
                "underforecast_qty": float((-group["error"]).clip(lower=0).sum()),
                "overforecast_qty": float(group["error"].clip(lower=0).sum()),
                "sku_days": int(len(sku_day)),
                "sku_day_mae": (
                    float(sku_day["abs_error"].mean()) if len(sku_day) else None
                ),
                "sku_day_wape": (
                    float(sku_day["abs_error"].sum() / actual)
                    if actual > 0
                    else None
                ),
                "sku_day_underforecast_qty": float(
                    (-sku_day["error"]).clip(lower=0).sum()
                ),
                "sku_day_overforecast_qty": float(
                    sku_day["error"].clip(lower=0).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_profile_changes(
    baseline: pd.DataFrame, adjusted: pd.DataFrame
) -> pd.DataFrame:
    left = compact_profile(
        baseline, min_n_days=MIN_TIER1_N_DAYS
    ).rename(
        columns={"profile_share": "baseline_share"}
    )
    right = compact_profile(
        adjusted, min_n_days=MIN_TIER1_N_DAYS
    ).rename(
        columns={"profile_share": "adjusted_share"}
    )
    changes = left.merge(
        right,
        on=PROFILE_KEYS,
        how="outer",
        suffixes=("_base", "_adjusted"),
    )
    changes[["baseline_share", "adjusted_share"]] = changes[
        ["baseline_share", "adjusted_share"]
    ].fillna(0.0)
    changes["share_delta"] = changes["adjusted_share"] - changes["baseline_share"]
    return changes.sort_values(
        "share_delta", key=lambda values: values.abs(), ascending=False
    )


def build_metric_deltas(
    metrics: pd.DataFrame,
    *,
    variant: str,
) -> dict[str, dict[str, float]]:
    metric_lookup = metrics.set_index(["variant", "scope"])
    result = {}
    delta_columns = [
        "mae",
        "wape",
        "underforecast_qty",
        "overforecast_qty",
        "sku_day_mae",
        "sku_day_wape",
        "sku_day_underforecast_qty",
        "sku_day_overforecast_qty",
    ]
    for scope in metrics["scope"].unique():
        base = metric_lookup.loc[("observed_sales_profile", scope)]
        adjusted = metric_lookup.loc[(variant, scope)]
        result[scope] = {
            f"{column}_delta": float(adjusted[column] - base[column])
            for column in delta_columns
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=str(DEFAULT_RAW))
    parser.add_argument("--adjustments", default=str(DEFAULT_ADJUSTMENTS))
    parser.add_argument("--stockouts", default=str(DEFAULT_STOCKOUTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--train-end", default=str(DEFAULT_TRAIN_END.date()))
    parser.add_argument("--holdout-start", default=str(DEFAULT_HOLDOUT_START.date()))
    parser.add_argument("--holdout-end", default=str(DEFAULT_HOLDOUT_END.date()))
    parser.add_argument(
        "--blend-weights",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75],
    )
    parser.add_argument(
        "--membership-seed-weights",
        nargs="+",
        type=float,
        default=[0.05, 0.1, 0.25, 0.5, 1.0],
    )
    args = parser.parse_args()

    train_end = pd.Timestamp(args.train_end)
    holdout_start = pd.Timestamp(args.holdout_start)
    holdout_end = pd.Timestamp(args.holdout_end)
    adjustments = pd.read_csv(args.adjustments)
    adjustments[DATE_COL] = pd.to_datetime(adjustments[DATE_COL]).dt.normalize()
    adjustments = adjustments[adjustments[DATE_COL].le(train_end)].copy()
    bakery_ids = set(pd.to_numeric(adjustments[BAKERY_ID_COL]).astype(int))
    stockouts = pd.read_csv(args.stockouts)
    stockouts[DATE_COL] = pd.to_datetime(stockouts[DATE_COL]).dt.normalize()
    stockouts = stockouts[stockouts[DATE_COL].between(holdout_start, holdout_end)]
    stockouts[[BAKERY_ID_COL, PRODUCT_ID_COL]] = stockouts[
        [BAKERY_ID_COL, PRODUCT_ID_COL]
    ].astype(int)

    hourly = load_hourly_raw(
        Path(args.raw),
        bakery_ids=bakery_ids,
        date_from=pd.Timestamp("2026-05-01"),
        date_to=holdout_end,
    )
    train = hourly[hourly[DATE_COL].le(train_end)].copy()
    holdout = hourly[hourly[DATE_COL].between(holdout_start, holdout_end)].copy()
    adjusted_train = apply_hourly_adjustments(train, adjustments)
    baseline_profile, _ = build_sku_hour_share_profile(train)
    adjusted_profile, _ = build_sku_hour_share_profile(adjusted_train)
    baseline_exact, _ = build_serving_profiles(baseline_profile)
    adjusted_exact, _ = build_serving_profiles(adjusted_profile)
    triple_cols = [BAKERY_ID_COL, DOW_COL, HOUR_COL]
    baseline_exact_triples = set(
        map(tuple, baseline_exact[triple_cols].drop_duplicates().to_numpy())
    )
    adjusted_exact_triples = set(
        map(tuple, adjusted_exact[triple_cols].drop_duplicates().to_numpy())
    )
    new_exact_triples = adjusted_exact_triples - baseline_exact_triples
    removed_exact_triples = baseline_exact_triples - adjusted_exact_triples
    baseline_exact_rows = set(map(tuple, baseline_exact[PROFILE_KEYS].to_numpy()))
    adjusted_exact_rows = set(map(tuple, adjusted_exact[PROFILE_KEYS].to_numpy()))
    new_tier1_rows = adjusted_exact_rows - baseline_exact_rows
    removed_tier1_rows = baseline_exact_rows - adjusted_exact_rows
    new_tier1_member_triples = {
        (int(bakery_id), int(dow), int(hour))
        for bakery_id, _product_id, dow, hour in new_tier1_rows
    }

    adjusted_pairs = set(
        map(
            tuple,
            adjustments.loc[
                adjustments["imputed_demand"].gt(0),
                [BAKERY_ID_COL, PRODUCT_ID_COL],
            ]
            .drop_duplicates()
            .astype(int)
            .to_numpy(),
        )
    )
    metric_parts = []
    scored_outputs = []
    variants = [
        ("observed_sales_profile", baseline_profile, None, None),
        ("demand_adjusted_profile", adjusted_profile, None, None),
        (
            "demand_adjusted_guarded_routing",
            adjusted_profile,
            baseline_exact_triples,
            None,
        ),
        (
            "demand_adjusted_new_membership_only",
            select_adjusted_contexts(
                baseline_profile,
                adjusted_profile,
                new_tier1_member_triples,
            ),
            baseline_exact_triples,
            baseline_profile,
        ),
    ]
    variants.extend(
        (
            f"demand_adjusted_blend_{weight:g}_guarded_routing",
            blend_profiles(
                baseline_profile,
                adjusted_profile,
                adjusted_weight=weight,
            ),
            baseline_exact_triples,
            None,
        )
        for weight in args.blend_weights
    )
    variants.extend(
        (
            f"demand_adjusted_membership_seed_{weight:g}",
            build_membership_seed_profile(
                baseline_profile,
                adjusted_profile,
                new_tier1_rows,
                seed_weight=weight,
            ),
            baseline_exact_triples,
            baseline_profile,
        )
        for weight in args.membership_seed_weights
    )
    variants.extend(
        (
            f"demand_adjusted_new_membership_blend_{weight:g}",
            select_adjusted_contexts(
                baseline_profile,
                blend_profiles(
                    baseline_profile,
                    adjusted_profile,
                    adjusted_weight=weight,
                ),
                new_tier1_member_triples,
            ),
            baseline_exact_triples,
            baseline_profile,
        )
        for weight in args.blend_weights
    )
    for variant, profile, allowed_exact, fallback_source in variants:
        scored = attach_evaluation_scopes(
            build_scored_rows(
                profile,
                holdout,
                allowed_exact_triples=allowed_exact,
                fallback_source_profile=fallback_source,
            ),
            stockouts,
            adjusted_pairs,
            new_exact_triples,
            new_tier1_member_triples,
        )
        scored["variant"] = variant
        scored_outputs.append(scored)
        metric_parts.append(summarize_scores(scored, variant=variant))
    metrics = pd.concat(metric_parts, ignore_index=True)
    changes = build_profile_changes(baseline_profile, adjusted_profile)
    stable_routing_mask = [
            (int(row.bakery_id), int(row.dow), int(row.hour))
            not in new_exact_triples
            for row in changes.itertuples()
    ]
    stable_changes = changes.loc[stable_routing_mask]
    stable_membership_mask = [
            (int(row.bakery_id), int(row.dow), int(row.hour))
            not in new_tier1_member_triples
            for row in changes.itertuples()
    ]
    stable_membership_changes = changes.loc[stable_membership_mask]

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    changes.to_csv(output / "profile_changes.csv", index=False)
    pd.concat(scored_outputs, ignore_index=True).to_csv(
        output / "scored_rows.csv", index=False
    )
    baseline_profile.to_csv(output / "baseline_profile.csv", index=False)
    adjusted_profile.to_csv(output / "adjusted_profile.csv", index=False)

    deltas = build_metric_deltas(
        metrics,
        variant="demand_adjusted_profile",
    )
    guarded_deltas = build_metric_deltas(
        metrics,
        variant="demand_adjusted_guarded_routing",
    )
    summary = {
        "train_end": str(train_end.date()),
        "holdout_start": str(holdout_start.date()),
        "holdout_end": str(holdout_end.date()),
        "bakeries": int(len(bakery_ids)),
        "train_imputed_units": float(adjustments["imputed_demand"].sum()),
        "train_adjusted_cases": int(
            adjustments[[DATE_COL, BAKERY_ID_COL, PRODUCT_ID_COL]]
            .drop_duplicates()
            .shape[0]
        ),
        "adjusted_pairs": int(len(adjusted_pairs)),
        "profile_rows": int(len(changes)),
        "profile_rows_changed": int(changes["share_delta"].abs().gt(1e-12).sum()),
        "max_abs_share_delta_pp": float(changes["share_delta"].abs().max() * 100),
        "max_abs_share_delta_stable_routing_pp": float(
            stable_changes["share_delta"].abs().max() * 100
        ),
        "max_abs_share_delta_stable_tier1_membership_pp": float(
            stable_membership_changes["share_delta"].abs().max() * 100
        ),
        "baseline_exact_triples": int(len(baseline_exact_triples)),
        "adjusted_exact_triples": int(len(adjusted_exact_triples)),
        "new_exact_triples": int(len(new_exact_triples)),
        "removed_exact_triples": int(len(removed_exact_triples)),
        "new_tier1_rows": int(len(new_tier1_rows)),
        "removed_tier1_rows": int(len(removed_tier1_rows)),
        "triples_with_new_tier1_member": int(len(new_tier1_member_triples)),
        "metric_deltas_adjusted_minus_baseline": deltas,
        "metric_deltas_guarded_routing_minus_baseline": guarded_deltas,
        "production_write": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
