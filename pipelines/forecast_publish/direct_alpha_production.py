"""Production Direct alpha=.25 post-processing for a bakery-level source run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pipelines.forecast_publish.direct_daily_to_hour import (
    expand_direct_sku_day_to_hour,
)
from pipelines.forecast_publish.load_forecast_run import load_forecast_run
from pipelines.forecast_publish.load_forecast_run import (
    DEFAULT_ENV_PATH,
    DEFAULT_SCHEMA_PATH,
    create_client,
)
from src.experiments_v2.direct_alpha_allocation import (
    DAY_KEYS,
    DirectAlphaAllocationConfig,
    build_selected_direct_plan,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = ROOT / "models" / "direct_alpha_025_v1"
DEFAULT_HOUR_PROFILE = ROOT / "data" / "processed" / "bakery_hour_profile.csv"
SALE_HEX = "D09FD180D0BED0B4D0B0D0B6D0B0"


def _reindex_sum(
    history: pd.DataFrame,
    day: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    values = (
        history[history["date"].between(start, end)]
        .groupby(["bakery_id", "product_id"])["sold"]
        .sum()
    )
    index = pd.MultiIndex.from_frame(day[["bakery_id", "product_id"]])
    return pd.Series(
        values.reindex(index, fill_value=0.0).to_numpy(),
        index=day.index,
        dtype="float64",
    )


def _normalized(values: pd.Series, groups: list[pd.Series]) -> pd.Series:
    total = values.groupby(groups).transform("sum")
    return (values / total.replace(0.0, np.nan)).fillna(0.0)


def build_day_features(day: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    result = day.copy().reset_index(drop=True)
    date = pd.Timestamp(result["date"].iloc[0])
    recent = _reindex_sum(
        history, result, date - pd.Timedelta(7, "D"), date - pd.Timedelta(1, "D")
    )
    prior = _reindex_sum(
        history, result, date - pd.Timedelta(14, "D"), date - pd.Timedelta(8, "D")
    )
    broad = _reindex_sum(
        history, result, date - pd.Timedelta(56, "D"), date - pd.Timedelta(1, "D")
    )
    weekday_dates = [date - pd.Timedelta(7 * step, "D") for step in range(1, 5)]
    weekday_values = (
        history[history["date"].isin(weekday_dates)]
        .groupby(["bakery_id", "product_id"])["sold"]
        .sum()
    )
    index = pd.MultiIndex.from_frame(result[["bakery_id", "product_id"]])
    weekday = pd.Series(
        weekday_values.reindex(index, fill_value=0.0).to_numpy(), index=result.index
    )
    presence_values = (
        history[
            history["date"].between(
                date - pd.Timedelta(28, "D"), date - pd.Timedelta(1, "D")
            )
        ]
        .loc[lambda frame: frame["sold"].gt(0.0)]
        .groupby(["bakery_id", "product_id"])["date"]
        .nunique()
    )
    presence = pd.Series(
        presence_values.reindex(index, fill_value=0).to_numpy(), index=result.index
    )
    bakery_groups = [result["bakery_id"]]
    result["recent_7_mean"] = recent / 7.0
    result["prior_7_mean"] = prior / 7.0
    result["broad_56_mean"] = broad / 56.0
    result["same_weekday_4_mean"] = weekday / 4.0
    result["presence_28"] = presence / 28.0
    result["recent_7_share"] = _normalized(recent, bakery_groups)
    result["broad_56_share"] = _normalized(broad, bakery_groups)
    result["same_weekday_4_share"] = _normalized(weekday, bakery_groups)
    category_broad = broad.groupby([result["bakery_id"], result["category"]]).transform(
        "sum"
    )
    bakery_broad = broad.groupby(result["bakery_id"]).transform("sum")
    result["historical_category_share"] = (
        category_broad / bakery_broad.replace(0.0, np.nan)
    ).fillna(0.0)
    result["recent_trend"] = ((recent + 1.0) / (prior + 1.0)).clip(0.25, 4.0)
    bakery_total = result.groupby(DAY_KEYS)["incumbent_sku_forecast"].transform("sum")
    result["log_bakery_total"] = np.log1p(bakery_total)
    result["dow"] = date.dayofweek
    return result


def _add_floor_reference(rows: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    history = labels.copy()
    history["dow"] = history["date"].dt.dayofweek
    outputs: list[pd.DataFrame] = []
    for date, day in rows.groupby("date", sort=True):
        date = pd.Timestamp(date)
        sample = history[
            history["date"].between(
                date - pd.Timedelta(56, "D"), date - pd.Timedelta(1, "D")
            )
            & history["dow"].eq(date.dayofweek)
            & history["demand_point_estimate"].gt(0.0)
        ]
        reference = sample.groupby(["bakery_id", "product_id"], as_index=False).agg(
            floor_history_n=("demand_point_estimate", "size"),
            floor_demand_p67=(
                "demand_point_estimate",
                lambda values: values.quantile(0.67),
            ),
            historical_stockout_rate=("is_clear_stockout", "mean"),
            historical_lost_mean=("imputed_demand", "mean"),
        )
        outputs.append(
            day.merge(
                reference,
                on=["bakery_id", "product_id"],
                how="left",
                validate="many_to_one",
            )
        )
    result = pd.concat(outputs, ignore_index=True)
    result["floor_history_n"] = result["floor_history_n"].fillna(0).astype(int)
    for column in [
        "floor_demand_p67",
        "historical_stockout_rate",
        "historical_lost_mean",
    ]:
        result[column] = result[column].fillna(0.0)
    return result


def _load_source(
    client, source_run_id: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    run = client.query_df(
        """
        select max(generated_at) generated_at
        from forecast_runs_embedded
        where run_id=%(run_id)s
        """,
        parameters={"run_id": source_run_id},
    )
    if run.empty or pd.isna(run.iloc[0]["generated_at"]):
        raise RuntimeError(f"Source run not found: {source_run_id}")
    history_through = pd.Timestamp(
        run.iloc[0]["generated_at"]
    ).normalize() - pd.Timedelta(1, "D")
    universe = client.query_df(
        """
        select forecast_date date, bakery_id, product_id,
               any(product_name) product_name, any(category_name) category,
               sum(forecast_qty) incumbent_sku_forecast
        from sku_forecast_day_embedded
        where run_id=%(run_id)s
        group by date,bakery_id,product_id
        """,
        parameters={"run_id": source_run_id},
    )
    bakery = client.query_df(
        """
        select forecast_date date,bakery_id,any(bakery_name) bakery_name,any(city) city
        from bakery_forecast_day_embedded where run_id=%(run_id)s
        group by date,bakery_id
        """,
        parameters={"run_id": source_run_id},
    )
    if universe.empty or bakery.empty:
        raise RuntimeError(f"Source run is empty: {source_run_id}")
    for frame in (universe, bakery):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    universe["product_id"] = universe["product_id"].astype("int64")
    return universe, bakery, history_through


def _load_sales(
    client, universe: pd.DataFrame, history_through: pd.Timestamp
) -> pd.DataFrame:
    bakery_ids = tuple(sorted(universe["bakery_id"].astype(int).unique()))
    history = client.query_df(
        f"""
        select check_date date,toInt64OrZero(toString(bakery_id)) bakery_id,
               toInt64OrZero(toString(product_id)) product_id,
               sum(toFloat64(quantity)) sold
        from (select distinct check_datetime,check_date,bakery_id,product_id,quantity,
              price,line_amount,cash_event_type from Svezhar.fct_check_lines
              where hex(cash_event_type)='{SALE_HEX}'
                and check_date between toDate(%(date_from)s) and toDate(%(date_to)s)
                and toInt64OrZero(toString(bakery_id)) in %(bakery_ids)s and quantity>0)
        group by date,bakery_id,product_id
        """,
        parameters={
            "date_from": str((history_through - pd.Timedelta(59, "D")).date()),
            "date_to": str(history_through.date()),
            "bakery_ids": bakery_ids,
        },
    )
    history["date"] = pd.to_datetime(history["date"]).dt.normalize()
    history["product_id"] = history["product_id"].astype("int64")
    return history


def _load_cold_start_registry(
    client,
    history: pd.DataFrame,
    universe: pd.DataFrame,
    as_of_date: pd.Timestamp,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    product_ids = tuple(sorted(universe["product_id"].astype(int).unique()))
    first_sales = client.query_df(
        f"""
        select toInt64OrZero(toString(product_id)) product_id,
               min(check_date) first_sale
        from (select distinct check_datetime,check_date,bakery_id,product_id,
              quantity,price,line_amount,cash_event_type
              from Svezhar.fct_check_lines
              where hex(cash_event_type)='{SALE_HEX}' and quantity>0
                and toInt64OrZero(toString(product_id)) in %(product_ids)s)
        group by product_id
        """,
        parameters={"product_ids": product_ids},
    )
    first_sales["first_sale"] = pd.to_datetime(first_sales["first_sale"])
    age = as_of_date - first_sales["first_sale"]
    cold_ids = set(first_sales.loc[age.dt.days.le(14), "product_id"].astype(int))
    sold = history[history["product_id"].isin(cold_ids)].copy()
    if sold.empty:
        return pd.DataFrame(columns=["bakery_id", "product_id", "cold_start_floor"])
    sold = sold.sort_values(["bakery_id", "product_id", "date"])
    sold["cold_start_floor"] = sold.groupby(["bakery_id", "product_id"])[
        "sold"
    ].transform(
        lambda values: values.ewm(
            alpha=0.90, adjust=False, min_periods=3
        ).mean()
    )
    registry = sold.groupby(["bakery_id", "product_id"], as_index=False).agg(
        sales_days=("date", "nunique"),
        cold_start_floor=("cold_start_floor", "last"),
    )
    return registry[
        registry["sales_days"].ge(3) & registry["cold_start_floor"].notna()
    ].reset_index(drop=True)


def _load_cold_bakery_ids(
    universe: pd.DataFrame, as_of_date: pd.Timestamp
) -> set[int]:
    daily = pd.read_csv(ROOT / "data" / "processed" / "bakery_daily_sales.csv")
    daily["date"] = pd.to_datetime(daily["date"])
    positive = daily[pd.to_numeric(daily["bakery_sales"]).gt(0.0)]
    first_sales = positive.groupby("bakery_id", as_index=False).agg(
        first_sale=("date", "min")
    )
    first_sales = first_sales[
        first_sales["bakery_id"].isin(universe["bakery_id"].unique())
    ]
    age = as_of_date - first_sales["first_sale"]
    return set(first_sales.loc[age.dt.days.le(28), "bakery_id"].astype(int))


def _hour_profile_with_fallback(sku_day: pd.DataFrame, path: Path) -> pd.DataFrame:
    profile = pd.read_csv(path, encoding="utf-8-sig")
    needed = sku_day[["date", "bakery_id"]].drop_duplicates().copy()
    needed["dow"] = pd.to_datetime(needed["date"]).dt.dayofweek
    available = set(
        map(tuple, profile[["bakery_id", "dow"]].drop_duplicates().to_numpy())
    )
    missing = [
        tuple(values)
        for values in needed[["bakery_id", "dow"]].to_numpy()
        if tuple(values) not in available
    ]
    if missing:
        network = profile.groupby(["dow", "hour"], as_index=False)[
            "mean_hour_share_norm"
        ].mean()
        fallback = pd.concat(
            [
                network[network["dow"].eq(dow)].assign(bakery_id=bakery_id)
                for bakery_id, dow in missing
            ],
            ignore_index=True,
        )
        profile = pd.concat([profile, fallback], ignore_index=True, sort=False)
    return profile


def run_direct_alpha_production(
    *,
    client,
    source_run_id: str,
    env_file: str | Path,
    schema_path: str | Path,
    output_dir: str | Path,
    run_id: str,
    activate: bool,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    hour_profile_path: str | Path = DEFAULT_HOUR_PROFILE,
) -> dict:
    artifacts = Path(artifact_dir)
    metadata = json.loads((artifacts / "metadata.json").read_text(encoding="utf-8"))
    direct = joblib.load(artifacts / "direct_model.joblib")
    classifier = joblib.load(artifacts / "stockout_classifier.joblib")
    severity = joblib.load(artifacts / "lost_severity_model.joblib")
    universe, bakery, history_through = _load_source(client, source_run_id)
    history = _load_sales(client, universe, history_through)
    first_forecast_date = pd.Timestamp(universe["date"].min()).normalize()
    floor_csv = artifacts / "floor_history.csv.gz"
    if floor_csv.exists():
        labels = pd.read_csv(floor_csv, compression="gzip")
    else:
        labels = pd.read_parquet(artifacts / "floor_history.parquet")
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    labels["product_id"] = labels["product_id"].astype("int64")
    cold_registry = _load_cold_start_registry(
        client,
        history,
        universe,
        first_forecast_date,
        labels,
    )
    cold_bakery_ids = _load_cold_bakery_ids(universe, first_forecast_date)
    cold_registry = cold_registry[
        ~cold_registry["bakery_id"].isin(cold_bakery_ids)
    ].copy()
    if cold_registry.empty:
        cold_registry = pd.DataFrame(
            columns=["bakery_id", "product_id", "cold_start_floor"]
        )
    features = pd.concat(
        [
            build_day_features(day, history)
            for _, day in universe.groupby("date", sort=True)
        ],
        ignore_index=True,
    )
    features = features.merge(
        cold_registry[["bakery_id", "product_id", "cold_start_floor"]],
        on=["bakery_id", "product_id"],
        how="left",
        validate="many_to_one",
    )
    features["is_cold_start"] = features["cold_start_floor"].notna()
    has_mature = (~features["is_cold_start"]).groupby(
        [features[key] for key in DAY_KEYS]
    ).transform("any")
    features.loc[~has_mature, "is_cold_start"] = False
    cold_features = features[features["is_cold_start"]].copy()
    features = features[~features["is_cold_start"]].copy()
    if features.empty:
        raise RuntimeError("Direct mature SKU allocation pool is empty")
    for source, target in [
        ("bakery_id", "bakery_code"),
        ("product_id", "product_code"),
        ("category", "category_code"),
    ]:
        mapping = metadata["mappings"][source]
        features[target] = (
            features[source].astype(str).map(mapping).fillna(-1).astype(int)
        )
    columns = metadata["features"]
    features["direct_raw_demand"] = np.maximum(direct.predict(features[columns]), 1e-9)
    raw_total = features.groupby(DAY_KEYS)["direct_raw_demand"].transform("sum")
    source_totals = universe.groupby(DAY_KEYS)["incumbent_sku_forecast"].sum()
    mature_index = pd.MultiIndex.from_frame(features[DAY_KEYS])
    bakery_total = pd.Series(
        source_totals.reindex(mature_index).to_numpy(),
        index=features.index,
    )
    features["direct_forecast"] = (
        features["direct_raw_demand"] / raw_total * bakery_total
    )
    factors = {int(key): value for key, value in metadata["p50_factors"].items()}
    features["p50_factor"] = (
        features["bakery_id"].map(factors).fillna(metadata["p50_fallback"])
    )
    features["direct_p50"] = features["direct_forecast"] * features["p50_factor"]
    probability = classifier.predict_proba(features[columns])[:, 1]
    conditional = np.expm1(severity.predict(features[columns])).clip(min=0.0)
    features["predictive_uplift"] = probability * conditional
    features["loss_scale"] = 1.0
    labels["demand_point_estimate"] = (
        labels["demand_lower_bound"] + labels["imputed_demand"]
    )
    features = _add_floor_reference(features, labels)
    selected = build_selected_direct_plan(features, DirectAlphaAllocationConfig())
    history_mass = selected.groupby(DAY_KEYS)["broad_56_mean"].transform("sum")
    cold = history_mass.le(0.0)
    selected.loc[cold, "selected_sku_forecast"] = selected.loc[
        cold, "incumbent_sku_forecast"
    ].clip(lower=0.0)
    cold_bakery_mask = selected["bakery_id"].isin(cold_bakery_ids)
    selected.loc[cold_bakery_mask, "selected_sku_forecast"] = selected.loc[
        cold_bakery_mask, "incumbent_sku_forecast"
    ].clip(lower=0.0)
    if not cold_features.empty:
        cold_output = cold_features[
            ["date", "bakery_id", "product_id", "cold_start_floor"]
        ].rename(columns={"cold_start_floor": "selected_sku_forecast"})
        selected = pd.concat([selected, cold_output], ignore_index=True, sort=False)
    sku_day = selected[
        ["date", "bakery_id", "product_id", "selected_sku_forecast"]
    ].rename(columns={"selected_sku_forecast": "sku_day_forecast"})
    totals = (
        sku_day.groupby(DAY_KEYS, as_index=False)["sku_day_forecast"]
        .sum()
        .rename(columns={"sku_day_forecast": "bakery_day_forecast"})
    )
    bakery = bakery.merge(totals, on=DAY_KEYS, how="inner", validate="one_to_one")
    bakery["bakery_day_forecast_bias_adj"] = bakery["bakery_day_forecast"]
    profile = _hour_profile_with_fallback(sku_day, Path(hour_profile_path))
    sku_hour = expand_direct_sku_day_to_hour(sku_day, profile)
    output = Path(output_dir) / "direct_alpha_025"
    output.mkdir(parents=True, exist_ok=True)
    bakery_path, day_path, hour_path = (
        output / "bakery_day.csv",
        output / "sku_day.csv",
        output / "sku_hour.csv",
    )
    bakery.to_csv(bakery_path, index=False, encoding="utf-8-sig")
    sku_day.to_csv(day_path, index=False, encoding="utf-8-sig")
    sku_hour.to_csv(hour_path, index=False, encoding="utf-8-sig")
    loaded = load_forecast_run(
        env_file=env_file,
        schema_path=schema_path,
        bakery_path=bakery_path,
        sku_day_path=day_path,
        sku_hour_path=hour_path,
        lookup_source="clickhouse",
        run_id=run_id,
        model_version="direct_alpha_025_v1",
        profile_version="bakery_dow_timing_v1",
        notes=(
            f"Direct alpha=.25 from {source_run_id}; "
            f"history through {history_through.date()}"
        ),
        replace_existing=True,
    )
    if activate:
        from pipelines.forecast_publish.activate_run import activate_run
        from pipelines.forecast_publish.table_names import (
            get_table_suffix_from_env_file,
        )

        activate_run(
            client, run_id, table_suffix=get_table_suffix_from_env_file(env_file)
        )
    return {
        "run_id": run_id,
        "source_run_id": source_run_id,
        "history_through": str(history_through.date()),
        "loaded_rows": loaded,
        "activated": activate,
        "cold_start_pairs": int(len(cold_registry)),
        "cold_bakery_ids": sorted(cold_bakery_ids),
    }


def latest_base_norm_recent_run(client) -> str:
    rows = client.query_df(
        """
        select run_id
        from forecast_runs_embedded
        where model_version = 'bakery_day_lgbm_base'
          and profile_version = 'clickhouse_norm_recent'
        order by generated_at desc
        limit 1
        """
    )
    if rows.empty:
        raise RuntimeError("No base_norm_recent source run found")
    return str(rows.iloc[0]["run_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH))
    parser.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()
    client = create_client(args.env_file)
    source_run_id = args.source_run_id or latest_base_norm_recent_run(client)
    run_id = args.run_id or source_run_id.replace(
        "base_bakery_norm_recent", "direct_alpha_025"
    )
    result = run_direct_alpha_production(
        client=client,
        source_run_id=source_run_id,
        env_file=args.env_file,
        schema_path=args.schema_path,
        output_dir=args.output_dir,
        run_id=run_id,
        activate=args.activate,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
