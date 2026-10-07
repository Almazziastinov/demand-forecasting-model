"""Unified causal architecture tournament on the official 55-bakery scope.

Every forecast variant is evaluated on the same SKU-day universe, reconstructed
demand, stock flows, two-day FIFO ledger, prices, and costs.  The script is
research-only and never writes to ClickHouse or production services.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from simulate_two_day_economics import simulate_group


ROOT = Path(__file__).resolve().parents[1]
SCOPE = (
    ROOT / "reports/comparable_produced_scope_p50_loss075_20260911/predictions.parquet"
)
POISSON = ROOT / "reports/direct_poisson_allocation_full_20260910/predictions.parquet"
SOURCE = (
    ROOT
    / "reports/historical_daily_retraining_gate_loss075_full_20260910"
    / "predictions.parquet"
)
PANEL = ROOT / ".codex_tmp/historical_clickhouse_panel/causal_panel.parquet"
DEMAND = (
    ROOT
    / "reports/network_causal_expanded_demand_20260911/network_daily_demand.parquet"
)
MAPPING = ROOT / "reports/markup_price_mapping_20260826/mapped_products.csv"
SKU_UPLIFT = (
    ROOT / "reports/network_sku_targeted_uplift_20260911/causal_uplift_features.parquet"
)
SKU_GATE = (
    ROOT / "reports/network_sku_targeted_uplift_20260911/rolling_gate_decisions.csv"
)
OUTPUT = ROOT / "reports/unified_architecture_tournament_20260911"

KEYS = ["date", "bakery_id", "product_id"]
DAY = ["date", "bakery_id"]
DISCOUNT = 0.30


def normalize_plan(rows: pd.DataFrame, raw: pd.Series, total: pd.Series) -> pd.Series:
    """Normalize a non-negative SKU signal inside each bakery-day."""
    safe = pd.to_numeric(raw, errors="coerce").fillna(0.0).clip(lower=0.0)
    denominator = safe.groupby([rows["date"], rows["bakery_id"]]).transform("sum")
    fallback = rows["scope_local_share"]
    share = (safe / denominator.replace(0.0, np.nan)).fillna(fallback)
    share_sum = share.groupby([rows["date"], rows["bakery_id"]]).transform("sum")
    return total * share / share_sum.replace(0.0, np.nan)


def complete_rolling_ratio(
    daily: pd.DataFrame,
    group_columns: list[str],
    *,
    prior: float,
    window: int = 14,
) -> pd.DataFrame:
    """Return a D-1 rolling actual/predicted ratio on a complete daily grid."""
    date_min = daily["date"].min()
    date_max = daily["date"].max()
    dates = pd.DataFrame({"date": pd.date_range(date_min, date_max)})
    if group_columns:
        groups = daily[group_columns].drop_duplicates().copy()
        groups["_join"] = 1
        dates["_join"] = 1
        grid = groups.merge(dates, on="_join", how="inner").drop(columns="_join")
    else:
        grid = dates
    work = grid.merge(daily, on=[*group_columns, "date"], how="left")
    work[["actual", "predicted"]] = work[["actual", "predicted"]].fillna(0.0)
    work = work.sort_values([*group_columns, "date"])
    if group_columns:
        grouped = work.groupby(group_columns, sort=False)
        work["actual_prior"] = grouped["actual"].transform(
            lambda values: values.shift(1).rolling(window, min_periods=3).sum()
        )
        work["predicted_prior"] = grouped["predicted"].transform(
            lambda values: values.shift(1).rolling(window, min_periods=3).sum()
        )
    else:
        work["actual_prior"] = (
            work["actual"].shift(1).rolling(window, min_periods=3).sum()
        )
        work["predicted_prior"] = (
            work["predicted"].shift(1).rolling(window, min_periods=3).sum()
        )
    work["ratio"] = (
        (work["actual_prior"] + prior) / (work["predicted_prior"] + prior)
    ).clip(0.70, 1.30)
    work["ratio"] = work["ratio"].fillna(1.0)
    return work[[*group_columns, "date", "actual_prior", "predicted_prior", "ratio"]]


def add_broader_sku_signals(rows: pd.DataFrame, poisson: pd.DataFrame) -> pd.DataFrame:
    """Attach strictly causal city/network SKU mix levels and residual regimes."""
    full = poisson.merge(
        rows[["bakery_id", "city"]].drop_duplicates("bakery_id"),
        on="bakery_id",
        how="left",
        validate="many_to_one",
    )
    city_daily = full.groupby(["date", "city", "product_id"], as_index=False).agg(
        actual=("sold", "sum"), predicted=("direct_plan_poisson", "sum")
    )
    network_daily = full.groupby(["date", "product_id"], as_index=False).agg(
        actual=("sold", "sum"), predicted=("direct_plan_poisson", "sum")
    )
    city_ratio = complete_rolling_ratio(city_daily, ["city", "product_id"], prior=20.0)
    network_ratio = complete_rolling_ratio(network_daily, ["product_id"], prior=50.0)

    city_ratio["city_prior_raw"] = city_ratio["actual_prior"] + 1.0
    network_ratio["network_prior_raw"] = network_ratio["actual_prior"] + 1.0
    result = rows.merge(
        city_ratio[["date", "city", "product_id", "ratio", "city_prior_raw"]].rename(
            columns={"ratio": "city_sku_regime_ratio"}
        ),
        on=["date", "city", "product_id"],
        how="left",
        validate="many_to_one",
    )
    result = result.merge(
        network_ratio[["date", "product_id", "ratio", "network_prior_raw"]].rename(
            columns={"ratio": "network_sku_regime_ratio"}
        ),
        on=["date", "product_id"],
        how="left",
        validate="many_to_one",
    )
    for column in ["city_sku_regime_ratio", "network_sku_regime_ratio"]:
        result[column] = result[column].fillna(1.0)
    for column in ["city_prior_raw", "network_prior_raw"]:
        result[column] = result[column].fillna(1.0)
    return result


def add_volume_regimes(rows: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """Attach D-1 forecast-error regimes at bakery, city, and network levels."""
    bakery_city = rows[["bakery_id", "city"]].drop_duplicates("bakery_id")
    daily = source.groupby(DAY, as_index=False).agg(
        actual=("observed_sales_qty", "sum"), predicted=("direct_total", "first")
    )
    daily = daily.merge(bakery_city, on="bakery_id", how="left", validate="many_to_one")
    bakery_ratio = complete_rolling_ratio(daily, ["bakery_id"], prior=1_000.0)
    city_daily = daily.groupby(["date", "city"], as_index=False)[
        ["actual", "predicted"]
    ].sum()
    city_ratio = complete_rolling_ratio(city_daily, ["city"], prior=5_000.0)
    network_daily = daily.groupby("date", as_index=False)[["actual", "predicted"]].sum()
    network_ratio = complete_rolling_ratio(network_daily, [], prior=10_000.0)

    result = rows.merge(
        bakery_ratio[["date", "bakery_id", "ratio"]].rename(
            columns={"ratio": "bakery_volume_ratio"}
        ),
        on=DAY,
        how="left",
        validate="many_to_one",
    )
    result = result.merge(
        city_ratio[["date", "city", "ratio"]].rename(
            columns={"ratio": "city_volume_ratio"}
        ),
        on=["date", "city"],
        how="left",
        validate="many_to_one",
    )
    result = result.merge(
        network_ratio[["date", "ratio"]].rename(
            columns={"ratio": "network_volume_ratio"}
        ),
        on="date",
        how="left",
        validate="many_to_one",
    )
    for column in ["bakery_volume_ratio", "city_volume_ratio", "network_volume_ratio"]:
        result[column] = result[column].fillna(1.0)
    result["hierarchical_volume_ratio"] = (
        0.50 * result["bakery_volume_ratio"]
        + 0.30 * result["city_volume_ratio"]
        + 0.20 * result["network_volume_ratio"]
    )
    return result


def load_rows() -> tuple[pd.DataFrame, list[str]]:
    rows = pd.read_parquet(SCOPE)
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    corrected_demand = pd.read_parquet(DEMAND)
    corrected_demand["date"] = pd.to_datetime(corrected_demand["date"]).dt.normalize()
    rows = rows.drop(columns=["demand"], errors="ignore").merge(
        corrected_demand[KEYS + ["observed_sales_qty", "lost_expanded"]].rename(
            columns={"observed_sales_qty": "observed_sales_qty_corrected"}
        ),
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    rows["observed_sales_qty"] = rows["observed_sales_qty_corrected"].fillna(
        rows["observed_sales_qty"]
    )
    rows = rows.drop(columns="observed_sales_qty_corrected")
    rows["lost_expanded"] = rows["lost_expanded"].fillna(0.0)
    rows["demand"] = rows["observed_sales_qty"] + rows["lost_expanded"]
    panel = pd.read_parquet(
        PANEL,
        columns=KEYS
        + [
            "city",
            "release_qty",
            "incoming_move_qty",
            "outgoing_move_qty",
            "written_off_qty",
        ],
    )
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    rows = rows.merge(panel, on=KEYS, how="left", validate="one_to_one")
    rows["city"] = rows["city"].fillna("unknown")
    for column in [
        "release_qty",
        "incoming_move_qty",
        "outgoing_move_qty",
        "written_off_qty",
    ]:
        rows[column] = (
            pd.to_numeric(rows[column], errors="coerce").fillna(0.0).clip(lower=0.0)
        )
    rows["opening_stock"] = 0.0
    rows["received"] = rows["incoming_move_qty"]
    rows["sent"] = rows["outgoing_move_qty"]
    rows["produced"] = rows["release_qty"]

    local_sum = rows.groupby(DAY)["share"].transform("sum")
    rows["scope_local_share"] = rows["share"] / local_sum.replace(0.0, np.nan)
    rows["plan_local_blend"] = rows["direct_total"] * rows["scope_local_share"]
    rows["plan_recent_7"] = normalize_plan(
        rows, rows["s7"] / 7.0 + 1e-6, rows["direct_total"]
    )
    rows["plan_same_weekday"] = normalize_plan(
        rows, rows["sw"] / 4.0 + 1e-6, rows["direct_total"]
    )
    rows["plan_broad_56"] = normalize_plan(
        rows, rows["s56"] / 56.0 + 1e-6, rows["direct_total"]
    )

    poisson = pd.read_parquet(
        POISSON,
        columns=KEYS + ["sold", "direct_raw_demand", "direct_plan_poisson"],
    )
    poisson["date"] = pd.to_datetime(poisson["date"]).dt.normalize()
    rows = rows.merge(
        poisson[KEYS + ["direct_raw_demand"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    rows["plan_direct_poisson"] = normalize_plan(
        rows, rows["direct_raw_demand"], rows["direct_total"]
    )
    rows = add_broader_sku_signals(rows, poisson)

    city_prior_plan = normalize_plan(rows, rows["city_prior_raw"], rows["direct_total"])
    network_prior_plan = normalize_plan(
        rows, rows["network_prior_raw"], rows["direct_total"]
    )
    rows["plan_direct_city_prior20"] = (
        0.80 * rows["plan_direct_poisson"] + 0.20 * city_prior_plan
    )
    rows["plan_direct_network_prior20"] = (
        0.80 * rows["plan_direct_poisson"] + 0.20 * network_prior_plan
    )
    rows["plan_direct_city_regime"] = normalize_plan(
        rows,
        rows["direct_raw_demand"] * np.sqrt(rows["city_sku_regime_ratio"]),
        rows["direct_total"],
    )
    rows["plan_direct_network_regime"] = normalize_plan(
        rows,
        rows["direct_raw_demand"] * np.sqrt(rows["network_sku_regime_ratio"]),
        rows["direct_total"],
    )
    rows["plan_direct_city_network_regime"] = normalize_plan(
        rows,
        rows["direct_raw_demand"]
        * rows["city_sku_regime_ratio"].pow(0.35)
        * rows["network_sku_regime_ratio"].pow(0.15),
        rows["direct_total"],
    )

    source = pd.read_parquet(
        SOURCE,
        columns=KEYS + ["observed_sales_qty", "direct_total"],
    )
    source["date"] = pd.to_datetime(source["date"]).dt.normalize()
    rows = add_volume_regimes(rows, source)
    for label, ratio in [
        ("bakery", "bakery_volume_ratio"),
        ("city", "city_volume_ratio"),
        ("network", "network_volume_ratio"),
        ("hierarchical", "hierarchical_volume_ratio"),
    ]:
        correction = (1.0 + 0.25 * (rows[ratio] - 1.0)).clip(0.95, 1.05)
        rows[f"plan_direct_{label}_volume_regime"] = (
            rows["plan_direct_poisson"] * correction
        )
    full_correction = (1.0 + 0.25 * (rows["hierarchical_volume_ratio"] - 1.0)).clip(
        0.95, 1.05
    )
    rows["plan_direct_full_hierarchy"] = (
        rows["plan_direct_city_network_regime"] * full_correction
    )

    uplift = pd.read_parquet(SKU_UPLIFT)
    uplift["date"] = pd.to_datetime(uplift["date"]).dt.normalize()
    rows = rows.merge(
        uplift[KEYS + ["expected_lost"]], on=KEYS, how="left", validate="one_to_one"
    )
    rows["expected_lost"] = rows["expected_lost"].fillna(0.0)
    rows["plan_local_sku_uplift075"] = (
        rows["plan_local_blend"] + 0.75 * rows["expected_lost"]
    )
    gate = pd.read_csv(SKU_GATE, usecols=["date", "bakery_id", "use_candidate"])
    gate["date"] = pd.to_datetime(gate["date"]).dt.normalize()
    rows = rows.merge(gate, on=DAY, how="left", validate="many_to_one")
    rows["plan_local_sku_uplift075_gate"] = np.where(
        rows["use_candidate"].fillna(False),
        rows["plan_local_sku_uplift075"],
        rows["plan_local_blend"],
    )

    # Plans must be formed on the complete common assortment.  Economics are
    # filtered only afterwards, identically for every variant, so unavailable
    # cost mappings cannot silently redistribute forecast volume.
    mapping = pd.read_csv(MAPPING, encoding="utf-8-sig")
    mapping = (
        mapping[mapping["valid_economics"].astype(bool)]
        .sort_values("unit_price")
        .drop_duplicates("product_id", keep="last")
    )
    rows = rows.merge(
        mapping[["product_id", "unit_cost", "unit_price"]],
        on="product_id",
        how="inner",
        validate="many_to_one",
    )
    rows["sale_price"] = rows["avg_sales_price"].where(
        rows["avg_sales_price"].gt(0), rows["unit_price"]
    )

    plan_columns = [column for column in rows.columns if column.startswith("plan_")]
    return rows, plan_columns


def summarize(simulation: pd.DataFrame, period_columns: list[str]) -> pd.DataFrame:
    result = simulation.copy()
    result["imbalance"] = result["under"] + result["expired"]
    grouped = result.groupby([*period_columns, "variant"], as_index=False).agg(
        demand=("demand", "sum"),
        production=("production", "sum"),
        served=("served", "sum"),
        under=("under", "sum"),
        expired=("expired", "sum"),
        imbalance=("imbalance", "sum"),
        gross_profit=("gross_profit", "sum"),
    )
    grouped["service_pct"] = 100.0 * grouped["served"] / grouped["demand"]
    fact = grouped[grouped["variant"].eq("actual_supply")][
        [*period_columns, "gross_profit"]
    ].rename(columns={"gross_profit": "actual_gross_profit"})
    grouped = grouped.merge(fact, on=period_columns, how="left", validate="many_to_one")
    grouped["gp_delta_vs_actual"] = (
        grouped["gross_profit"] - grouped["actual_gross_profit"]
    )
    grouped["gp_delta_vs_actual_pct"] = (
        100.0 * grouped["gp_delta_vs_actual"] / grouped["actual_gross_profit"]
    )
    return grouped


def main() -> None:
    rows, plan_columns = load_rows()
    variants: dict[str, str | None] = {"actual_supply": None}
    variants.update({column.removeprefix("plan_"): column for column in plan_columns})
    print(
        f"rows={len(rows):,} dates={rows.date.nunique()} "
        f"bakeries={rows.bakery_id.nunique()} "
        f"products={rows.product_id.nunique()} variants={len(variants)}",
        flush=True,
    )
    groups = rows.groupby(["bakery_id", "product_id"], sort=False)
    daily_parts = []
    for variant_number, (variant, plan_column) in enumerate(variants.items(), start=1):
        variant_rows = pd.concat(
            [simulate_group(group, plan_column) for _, group in groups],
            ignore_index=True,
        )
        variant_rows = variant_rows.merge(
            rows[KEYS + ["sale_price", "unit_cost"]],
            on=KEYS,
            how="left",
            validate="one_to_one",
        )
        variant_rows["variant"] = variant
        variant_rows["under"] = variant_rows["lost"]
        variant_rows["expired"] = variant_rows["expired_strategy_stock"]
        revenue = variant_rows["sold_fresh"] * variant_rows[
            "sale_price"
        ] + variant_rows["sold_yesterday"] * variant_rows["sale_price"] * (
            1.0 - DISCOUNT
        )
        variant_rows["gross_profit"] = revenue - (
            variant_rows["production"] * variant_rows["unit_cost"]
        )
        daily_parts.append(
            variant_rows.groupby(["date", "variant"], as_index=False).agg(
                demand=("demand", "sum"),
                production=("production", "sum"),
                served=("served", "sum"),
                under=("under", "sum"),
                expired=("expired", "sum"),
                gross_profit=("gross_profit", "sum"),
            )
        )
        print(
            f"simulated variant={variant_number}/{len(variants)} {variant}",
            flush=True,
        )
    simulation = pd.concat(daily_parts, ignore_index=True)
    simulation["month"] = simulation["date"].dt.to_period("M").astype(str)
    total = summarize(simulation.assign(scope="full"), ["scope"])
    monthly = summarize(simulation, ["month"])
    stability = (
        monthly[~monthly["variant"].eq("actual_supply")]
        .groupby("variant", as_index=False)
        .agg(
            positive_months=(
                "gp_delta_vs_actual",
                lambda values: int(values.gt(0).sum()),
            ),
            worst_month_delta=("gp_delta_vs_actual", "min"),
            best_month_delta=("gp_delta_vs_actual", "max"),
        )
    )
    leaderboard = total.merge(stability, on="variant", how="left")
    leaderboard = leaderboard.sort_values("gross_profit", ascending=False)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(OUTPUT / "leaderboard.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(OUTPUT / "monthly.csv", index=False, encoding="utf-8-sig")
    simulation.to_parquet(OUTPUT / "daily.parquet", index=False)
    metadata = {
        "production_write": False,
        "dates": [str(rows["date"].min().date()), str(rows["date"].max().date())],
        "bakeries": int(rows["bakery_id"].nunique()),
        "rows": int(len(rows)),
        "primary_metric": "gross_profit and gp_delta_vs_actual",
        "demand": "corrected causal L1+L2 reconstructed demand",
        "inventory": "common two-day FIFO with common reconciliation inflow",
        "variants": variants,
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        leaderboard[
            [
                "variant",
                "gross_profit",
                "gp_delta_vs_actual",
                "gp_delta_vs_actual_pct",
                "positive_months",
                "under",
                "expired",
                "service_pct",
            ]
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
