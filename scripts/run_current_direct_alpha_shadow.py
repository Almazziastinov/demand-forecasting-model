"""Run one current-horizon Direct alpha=.25 shadow without database writes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "apps/forecast_embedded"))

from app.db import get_client  # noqa: E402
from scripts.backtest_direct_bakery_sku_allocation import (  # noqa: E402
    FEATURES,
    build_day_features,
)
from scripts.build_direct_uplift_floor_candidates import add_floor_reference  # noqa: E402
from scripts.build_relaxed_stockout_demand import SALE_HEX  # noqa: E402
from scripts.run_direct_alpha_shadow import run_shadow  # noqa: E402
from src.experiments_v2.direct_alpha_allocation import DAY_KEYS  # noqa: E402


FEATURE_CACHE = ROOT / ".codex_tmp/direct_bakery_sku_features_20260827.parquet"
ARTIFACT_DIR = ROOT / "models/direct_alpha_025_v1"
LABELS = ARTIFACT_DIR / "floor_history.parquet"
HOUR_PROFILE = ROOT / "data/processed/bakery_hour_profile.csv"
DEFAULT_OUTPUT = ROOT / "reports/direct_alpha_current_shadow_20260831"


def active_run(client) -> tuple[str, pd.Timestamp]:
    rows = client.query_df(
        """
        select run_id, generated_at
        from forecast_runs_embedded
        where status = 'active'
        order by generated_at desc
        limit 1
        """
    )
    if rows.empty:
        raise RuntimeError("No active forecast run")
    return str(rows.iloc[0]["run_id"]), pd.Timestamp(rows.iloc[0]["generated_at"])


def load_universe(client, run_id: str, forecast_date: pd.Timestamp) -> pd.DataFrame:
    return client.query_df(
        """
        select forecast_date date, bakery_id, product_id,
               any(product_name) product_name,
               any(category_name) category,
               sum(forecast_qty) incumbent_sku_forecast
        from sku_forecast_day_embedded
        where run_id = %(run_id)s and forecast_date = toDate(%(forecast_date)s)
        group by date, bakery_id, product_id
        """,
        parameters={"run_id": run_id, "forecast_date": str(forecast_date.date())},
    )


def load_sales(
    client, bakery_ids: tuple[int, ...], start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    return client.query_df(
        f"""
        select check_date date, toInt64OrZero(toString(bakery_id)) bakery_id,
               toInt64OrZero(toString(product_id)) product_id,
               sum(toFloat64(quantity)) sold
        from (
            select distinct check_datetime, check_date, bakery_id, product_id,
                   quantity, price, line_amount, cash_event_type
            from Svezhar.fct_check_lines
            where hex(cash_event_type) = '{SALE_HEX}'
              and check_date between toDate(%(date_from)s) and toDate(%(date_to)s)
              and toInt64OrZero(toString(bakery_id)) in %(bakery_ids)s
              and quantity > 0
        )
        group by date, bakery_id, product_id
        """,
        parameters={
            "date_from": str(start.date()),
            "date_to": str(end.date()),
            "bakery_ids": bakery_ids,
        },
    )


def encoded_features(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    result = test.copy()
    for source, target in [
        ("bakery_id", "bakery_code"),
        ("product_id", "product_code"),
        ("category", "category_code"),
    ]:
        mapping = (
            train[[source, target]].drop_duplicates(source).set_index(source)[target]
        )
        result[target] = result[source].map(mapping).fillna(-1).astype(int)
    return result


def load_metadata() -> dict:
    return json.loads((ARTIFACT_DIR / "metadata.json").read_text(encoding="utf-8"))


def predict_artifacts(test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    direct = joblib.load(ARTIFACT_DIR / "direct_model.joblib")
    classifier = joblib.load(ARTIFACT_DIR / "stockout_classifier.joblib")
    severity = joblib.load(ARTIFACT_DIR / "lost_severity_model.joblib")
    direct_raw = np.maximum(direct.predict(test[FEATURES]), 1e-9)
    probability = classifier.predict_proba(test[FEATURES])[:, 1]
    conditional = np.expm1(severity.predict(test[FEATURES])).clip(min=0.0)
    return direct_raw, probability, conditional


def build_input(forecast_date: pd.Timestamp, output: Path) -> tuple[Path, dict]:
    client = get_client()
    run_id, generated_at = active_run(client)
    history_through = generated_at.normalize() - pd.Timedelta(days=1)
    universe = load_universe(client, run_id, forecast_date)
    if universe.empty:
        raise RuntimeError(
            f"Active run {run_id} has no rows for {forecast_date.date()}"
        )
    universe["date"] = pd.to_datetime(universe["date"]).dt.normalize()
    universe["product_id"] = universe["product_id"].astype("int64")
    bakery_ids = tuple(sorted(universe["bakery_id"].astype(int).unique()))
    history = load_sales(
        client,
        bakery_ids,
        history_through - pd.Timedelta(days=55),
        history_through,
    )
    history["date"] = pd.to_datetime(history["date"]).dt.normalize()
    history["product_id"] = history["product_id"].astype("int64")

    train = pd.read_parquet(FEATURE_CACHE)
    train["date"] = pd.to_datetime(train["date"]).dt.normalize()
    train = train[train["date"].lt(forecast_date)].copy()
    test = build_day_features(universe, history)
    test = encoded_features(train, test)
    direct_raw, probability, conditional = predict_artifacts(test)
    test["direct_raw_demand"] = direct_raw
    raw_total = test.groupby(DAY_KEYS)["direct_raw_demand"].transform("sum")
    bakery_total = test.groupby(DAY_KEYS)["incumbent_sku_forecast"].transform("sum")
    test["direct_forecast"] = test["direct_raw_demand"] / raw_total * bakery_total

    artifact_meta = load_metadata()
    factors = {int(key): value for key, value in artifact_meta["p50_factors"].items()}
    test["p50_factor"] = (
        test["bakery_id"].map(factors).fillna(artifact_meta["p50_fallback"])
    )
    test["direct_p50"] = test["direct_forecast"] * test["p50_factor"]

    labels = pd.read_parquet(LABELS)
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    labels["product_id"] = labels["product_id"].astype("int64")
    test["predicted_stockout_probability"] = probability
    test["predicted_lost_if_stockout"] = conditional
    test["predictive_uplift"] = probability * conditional
    test["loss_scale"] = 1.0

    floor_labels = labels.copy()
    floor_labels["demand_point_estimate"] = (
        floor_labels["demand_lower_bound"] + floor_labels["imputed_demand"]
    )
    test = add_floor_reference(test, floor_labels)
    output.mkdir(parents=True, exist_ok=True)
    input_path = output / "shadow_input.parquet"
    test.to_parquet(input_path, index=False)
    metadata = {
        "source_run_id": run_id,
        "forecast_date": str(forecast_date.date()),
        "history_through": str(history_through.date()),
        "rows": int(len(test)),
        "bakeries": int(test["bakery_id"].nunique()),
        "database_write": False,
    }
    return input_path, metadata


def export_publish_files(output: Path, source_run_id: str) -> dict[str, object]:
    from pipelines.forecast_publish.direct_daily_to_hour import (
        expand_direct_sku_day_to_hour,
    )

    rows = pd.read_parquet(output / "shadow_rows.parquet")
    sku_day = rows.rename(columns={"selected_sku_forecast": "sku_day_forecast"})
    sku_day_path = output / "sku_day_forecast.csv"
    sku_day[["date", "bakery_id", "product_id", "sku_day_forecast"]].to_csv(
        sku_day_path, index=False, encoding="utf-8-sig"
    )
    profile = pd.read_csv(HOUR_PROFILE, encoding="utf-8-sig")
    needed = sku_day[["date", "bakery_id"]].drop_duplicates().copy()
    needed["dow"] = pd.to_datetime(needed["date"]).dt.dayofweek
    available = set(
        map(tuple, profile[["bakery_id", "dow"]].drop_duplicates().to_numpy())
    )
    missing_pairs = [
        tuple(values)
        for values in needed[["bakery_id", "dow"]].to_numpy()
        if tuple(values) not in available
    ]
    if missing_pairs:
        network = profile.groupby(["dow", "hour"], as_index=False)[
            "mean_hour_share_norm"
        ].mean()
        fallback = pd.concat(
            [
                network[network["dow"].eq(dow)].assign(bakery_id=bakery_id)
                for bakery_id, dow in missing_pairs
            ],
            ignore_index=True,
        )
        profile = pd.concat([profile, fallback], ignore_index=True, sort=False)
    sku_hour = expand_direct_sku_day_to_hour(sku_day, profile)
    sku_hour_path = output / "sku_hour_forecast.csv"
    sku_hour[["date", "bakery_id", "product_id", "hour", "sku_hour_forecast"]].to_csv(
        sku_hour_path, index=False, encoding="utf-8-sig"
    )
    bakery = get_client().query_df(
        """
        select forecast_date date, bakery_id, any(bakery_name) bakery_name,
               any(city) city, sum(forecast_final) bakery_day_forecast
        from bakery_forecast_day_embedded
        where run_id = %(run_id)s
          and forecast_date in %(forecast_dates)s
        group by date, bakery_id
        """,
        parameters={
            "run_id": source_run_id,
            "forecast_dates": tuple(
                str(pd.Timestamp(value).date())
                for value in pd.to_datetime(sku_day["date"]).unique()
            ),
        },
    )
    selected_bakery = (
        sku_day.groupby(["date", "bakery_id"], as_index=False)["sku_day_forecast"]
        .sum()
        .rename(columns={"sku_day_forecast": "selected_bakery_forecast"})
    )
    bakery = bakery.merge(
        selected_bakery,
        on=["date", "bakery_id"],
        how="inner",
        validate="one_to_one",
    )
    bakery["bakery_day_forecast"] = bakery["selected_bakery_forecast"]
    bakery["bakery_day_forecast_bias_adj"] = bakery["selected_bakery_forecast"]
    bakery = bakery.drop(columns="selected_bakery_forecast")
    bakery_path = output / "bakery_day_forecast.csv"
    bakery.to_csv(bakery_path, index=False, encoding="utf-8-sig")
    return {
        "bakery": str(bakery_path),
        "sku_day": str(sku_day_path),
        "sku_hour": str(sku_hour_path),
        "network_hour_profile_fallback_pairs": [
            {"bakery_id": int(bakery_id), "dow": int(dow)}
            for bakery_id, dow in missing_pairs
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-date", type=pd.Timestamp, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    input_path, metadata = build_input(args.forecast_date.normalize(), args.output_dir)
    summary = run_shadow(input_path, args.output_dir)
    summary["source"] = metadata
    summary["publish_files"] = export_publish_files(
        args.output_dir, metadata["source_run_id"]
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
