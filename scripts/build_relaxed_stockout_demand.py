"""Build relaxed same-day-rate stockout labels for the pilot bakeries."""

from __future__ import annotations

import argparse
from functools import reduce
from pathlib import Path

import clickhouse_connect
import pandas as pd
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "reports/stockout_adjusted_demand_dataset/sku_day_demand.csv"
DEFAULT_OUTPUT = ROOT / "reports/relaxed_stockout_demand_20260826/sku_day_demand.csv"
DEFAULT_BAKERY_DATASET = ROOT / "data/processed/stg_daily_v1/bakery_daily_sales.csv"
KEYS = ["date", "bakery_id", "product_id"]
SALE_HEX = "D09FD180D0BED0B4D0B0D0B6D0B0"


def client_from_env(path: Path):
    values = dotenv_values(path)
    prefix = "CLICKHOUSE_" if "CLICKHOUSE_HOST" in values else ""
    return clickhouse_connect.get_client(
        host=values[f"{prefix}HOST"],
        port=int(values[f"{prefix}PORT"]),
        username=values[f"{prefix}USER"],
        password=values[f"{prefix}PASSWORD"],
        database=values[f"{prefix}DATABASE"],
        secure=True,
        verify=False,
        connect_timeout=30,
        send_receive_timeout=180,
    )


def query_components(client, date_from: str, date_to: str, bakery_ids: tuple[int, ...]):
    parameters = {"date_from": date_from, "date_to": date_to, "bakery_ids": bakery_ids}
    queries = {
        "produced": """
            select date, toInt64OrZero(toString(bid)) bakery_id,
                   toInt64OrZero(toString(pid)) product_id, sum(qty) produced
            from (
                select toDate(argMax(release_date, _updated_at)) date,
                       argMax(bakery_id, _updated_at) bid,
                       argMax(product_id, _updated_at) pid,
                       toFloat64(argMax(quantity, _updated_at)) qty
                from Svezhar.fct_production_release
                where release_date between toDate(%(date_from)s) and toDate(%(date_to)s)
                  and toInt64OrZero(toString(bakery_id)) in %(bakery_ids)s
                group by release_id, line_id
                having argMax(is_deleted, _updated_at) not in ('1', 'true', 'Да')
            ) group by date, bakery_id, product_id
        """,
        "sales": f"""
            select check_date date, toInt64OrZero(toString(bakery_id)) bakery_id,
                   toInt64OrZero(toString(product_id)) product_id,
                   sum(toFloat64(quantity)) sold,
                   max(toUnixTimestamp(check_datetime)) last_sale_ts
            from (
                select distinct check_datetime, check_date, bakery_id, product_id,
                       quantity, price, line_amount, cash_event_type
                from Svezhar.fct_check_lines
                where hex(cash_event_type) = '{SALE_HEX}'
                  and check_date between toDate(%(date_from)s) and toDate(%(date_to)s)
                  and toInt64OrZero(toString(bakery_id)) in %(bakery_ids)s
                  and quantity > 0
            ) group by date, bakery_id, product_id
        """,
        "received": """
            select date, toInt64OrZero(toString(receiver)) bakery_id,
                   toInt64OrZero(toString(pid)) product_id, sum(qty) received
            from (
                select toDate(argMax(move_date, _updated_at)) date,
                       argMax(receiver_id, _updated_at) receiver,
                       argMax(product_id, _updated_at) pid,
                       toFloat64(argMax(quantity, _updated_at)) qty
                from Svezhar.fct_moves
                where move_date between toDate(%(date_from)s) and toDate(%(date_to)s)
                  and toInt64OrZero(toString(receiver_id)) in %(bakery_ids)s
                group by move_id, line_id
                having argMax(is_deleted, _updated_at) not in ('1', 'true', 'Да')
            ) group by date, bakery_id, product_id
        """,
        "sent": """
            select date, toInt64OrZero(toString(sender)) bakery_id,
                   toInt64OrZero(toString(pid)) product_id, sum(qty) sent
            from (
                select toDate(argMax(move_date, _updated_at)) date,
                       argMax(sender_id, _updated_at) sender,
                       argMax(product_id, _updated_at) pid,
                       toFloat64(argMax(quantity, _updated_at)) qty
                from Svezhar.fct_moves
                where move_date between toDate(%(date_from)s) and toDate(%(date_to)s)
                  and toInt64OrZero(toString(sender_id)) in %(bakery_ids)s
                group by move_id, line_id
                having argMax(is_deleted, _updated_at) not in ('1', 'true', 'Да')
            ) group by date, bakery_id, product_id
        """,
    }
    frames = [client.query_df(sql, parameters=parameters) for sql in queries.values()]
    return reduce(lambda left, right: left.merge(right, on=KEYS, how="outer"), frames)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bakery-dataset", type=Path, default=DEFAULT_BAKERY_DATASET)
    parser.add_argument("--universe-from-components", action="store_true")
    parser.add_argument("--open-hour", type=float, default=7.0)
    parser.add_argument("--stockout-before-hour", type=float, default=19.0)
    parser.add_argument("--close-hour", type=float, default=23.0)
    args = parser.parse_args()

    demand = pd.read_csv(args.source, encoding="utf-8-sig", low_memory=False)
    demand["date"] = pd.to_datetime(demand["date"]).dt.normalize()
    if args.universe_from_components:
        bakery_days = pd.read_csv(
            args.bakery_dataset,
            usecols=["date", "bakery_id"],
            encoding="utf-8-sig",
            low_memory=False,
        )
        bakery_days["date"] = pd.to_datetime(bakery_days["date"]).dt.normalize()
        bakery_ids = tuple(sorted(bakery_days["bakery_id"].astype(int).unique()))
        start = pd.Timestamp("2026-05-01")
        end = bakery_days["date"].max()
    else:
        bakery_ids = tuple(sorted(demand["bakery_id"].astype(int).unique()))
        start = demand["date"].min() - pd.Timedelta(days=1)
        end = demand["date"].max()
    components = query_components(
        client_from_env(args.env_file), str(start.date()), str(end.date()), bakery_ids
    )
    components["date"] = pd.to_datetime(components["date"]).dt.normalize()
    for column in ["produced", "sold", "received", "sent"]:
        components[column] = pd.to_numeric(
            components[column], errors="coerce"
        ).fillna(0.0)
    components["closing"] = (
        components["produced"] + components["received"]
        - components["sent"] - components["sold"]
    ).clip(lower=0.0)
    opening = components[KEYS + ["closing"]].copy()
    opening["date"] += pd.Timedelta(days=1)
    opening = opening.rename(columns={"closing": "opening_stock"})
    components = components.merge(opening, on=KEYS, how="left")
    components["opening_stock"] = components["opening_stock"].fillna(0.0)
    components["available"] = (
        components["produced"] + components["opening_stock"]
        + components["received"] - components["sent"]
    )
    last_sale = pd.to_datetime(components["last_sale_ts"], unit="s", errors="coerce")
    components["last_sale_hour_exact"] = (
        last_sale.dt.hour + last_sale.dt.minute / 60 + last_sale.dt.second / 3600
    )
    controlled_pairs = set(
        components.loc[
            components[["produced", "received", "sent"]].sum(axis=1).gt(0),
            ["bakery_id", "product_id"],
        ].itertuples(index=False, name=None)
    )
    is_controlled = pd.Series(
        list(zip(components["bakery_id"], components["product_id"], strict=True)),
        index=components.index,
    ).isin(controlled_pairs)
    components["relaxed_stockout"] = (
        is_controlled
        & components["available"].gt(0)
        & components["sold"].gt(0)
        & components["available"].le(components["sold"] + 1e-9)
        & components["last_sale_hour_exact"].lt(args.stockout_before_hour)
        & components["last_sale_hour_exact"].gt(args.open_hour)
    )
    elapsed = (components["last_sale_hour_exact"] - args.open_hour).clip(lower=0.25)
    remaining = (args.close_hour - components["last_sale_hour_exact"]).clip(lower=0.0)
    components["raw_rate_lost"] = components["sold"] / elapsed * remaining

    component_labels = components[
        KEYS
        + [
            "sold",
            "relaxed_stockout",
            "last_sale_hour_exact",
            "raw_rate_lost",
        ]
    ].copy()
    if args.universe_from_components:
        result = component_labels.rename(columns={"sold": "demand_lower_bound"})
        result["suggested_training_weight"] = 1.0
    else:
        result = demand.drop(
            columns=[
                "is_clear_stockout", "last_sale_hour", "reference_days",
                "raw_imputed_demand", "imputed_demand", "case_cap",
            ],
            errors="ignore",
        ).merge(
            component_labels.drop(columns="sold"),
            on=KEYS,
            how="left",
            validate="one_to_one",
        )
    result["is_clear_stockout"] = result["relaxed_stockout"].fillna(False)
    result["last_sale_hour"] = result["last_sale_hour_exact"].where(
        result["is_clear_stockout"]
    )
    result["reference_days"] = result["is_clear_stockout"].astype(int) * 3
    result["raw_imputed_demand"] = result["raw_rate_lost"].where(
        result["is_clear_stockout"], 0.0
    ).fillna(0.0)
    observed = pd.to_numeric(result["demand_lower_bound"], errors="coerce").fillna(0.0)
    result["case_cap"] = pd.concat(
        [pd.Series(10.0, index=result.index), observed.clip(lower=4.0) * 0.5], axis=1
    ).min(axis=1)
    result["imputed_demand"] = result[["raw_imputed_demand", "case_cap"]].min(axis=1)
    result["demand_point_estimate"] = observed + result["imputed_demand"]
    result = result.drop(
        columns=["relaxed_stockout", "last_sale_hour_exact", "raw_rate_lost"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    selected = result[result["is_clear_stockout"]]
    print(
        f"rows={len(result)} stockouts={len(selected)} "
        f"raw_lost={selected['raw_imputed_demand'].sum():.3f} "
        f"capped_lost={selected['imputed_demand'].sum():.3f}"
    )


if __name__ == "__main__":
    main()
