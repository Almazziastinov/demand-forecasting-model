"""Build a local inventory-aware stockout audit for ten pilot bakeries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments_v2.stockout_demand_preprocessing import (  # noqa: E402
    build_inventory_balance,
)

PILOT_BAKERY_IDS = {20, 21, 22, 28, 80, 89, 107, 221, 222, 257}


def _numeric_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype("int64")


def _name_key(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.casefold()


def load_name_maps(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    bakery_parts = []
    product_parts = []
    usecols = ["bakery_id", "bakery_name", "product_id", "product_name"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=750_000):
        chunk["bakery_id"] = _numeric_id(chunk["bakery_id"])
        chunk = chunk[chunk["bakery_id"].isin(PILOT_BAKERY_IDS)].copy()
        if chunk.empty:
            continue
        chunk["product_id"] = _numeric_id(chunk["product_id"])
        bakery_parts.append(chunk[["bakery_id", "bakery_name"]].drop_duplicates())
        product_parts.append(chunk[["product_id", "product_name"]].drop_duplicates())
    bakeries = pd.concat(bakery_parts, ignore_index=True).drop_duplicates()
    products = pd.concat(product_parts, ignore_index=True).drop_duplicates()
    bakeries["bakery_name_key"] = _name_key(bakeries["bakery_name"])
    products["product_name_key"] = _name_key(products["product_name"])
    bakeries = bakeries.drop_duplicates("bakery_name_key")
    products = products.drop_duplicates("product_name_key")
    return bakeries, products


def load_daily(
    path: Path, bakeries: pd.DataFrame, products: pd.DataFrame
) -> pd.DataFrame:
    source = pd.read_csv(
        path,
        usecols=[
            "Дата",
            "Пекарня",
            "Номенклатура",
            "Продано",
            "Выпуск",
            "Остаток",
            "stock_lag1",
        ],
    )
    source["date"] = pd.to_datetime(source["Дата"], errors="coerce")
    source["bakery_name_key"] = _name_key(source["Пекарня"])
    source["product_name_key"] = _name_key(source["Номенклатура"])
    source = source.merge(
        bakeries[["bakery_id", "bakery_name", "bakery_name_key"]],
        on="bakery_name_key",
        how="inner",
    ).merge(
        products[["product_id", "product_name", "product_name_key"]],
        on="product_name_key",
        how="inner",
    )
    source = source[source["date"] >= pd.Timestamp("2026-03-01")].copy()
    return source.rename(
        columns={
            "Продано": "sold",
            "Выпуск": "produced",
            "Остаток": "closing_stock",
            "stock_lag1": "opening_stock",
        }
    )[
        [
            "date",
            "bakery_id",
            "bakery_name",
            "product_id",
            "product_name",
            "sold",
            "produced",
            "opening_stock",
            "closing_stock",
        ]
    ]


def load_moves(path: Path) -> pd.DataFrame:
    moves = pd.read_csv(path)
    moves["date"] = pd.to_datetime(moves["move_date"], errors="coerce")
    moves["product_id"] = _numeric_id(moves["product_id"])
    moves["sender_id"] = _numeric_id(moves["sender_id"])
    moves["receiver_id"] = _numeric_id(moves["receiver_id"])
    moves["quantity"] = pd.to_numeric(moves["quantity"], errors="coerce").fillna(0.0)
    moves = moves.drop_duplicates(
        ["move_id", "date", "product_id", "sender_id", "receiver_id", "quantity"]
    )
    incoming = (
        moves[moves["receiver_id"].isin(PILOT_BAKERY_IDS)]
        .groupby(["date", "receiver_id", "product_id"], as_index=False)["quantity"]
        .sum()
        .rename(columns={"receiver_id": "bakery_id", "quantity": "incoming_move_qty"})
    )
    outgoing = (
        moves[moves["sender_id"].isin(PILOT_BAKERY_IDS)]
        .groupby(["date", "sender_id", "product_id"], as_index=False)["quantity"]
        .sum()
        .rename(columns={"sender_id": "bakery_id", "quantity": "outgoing_move_qty"})
    )
    return incoming.merge(outgoing, on=["date", "bakery_id", "product_id"], how="outer")


def main() -> None:
    bakeries, products = load_name_maps(ROOT / "data/raw/sales_hrs_stg_2026.csv")
    daily = load_daily(
        ROOT / "data/processed/preprocessed_data_merged.csv", bakeries, products
    )
    moves = load_moves(ROOT / "data/raw/moves_clickhouse_2025-01-15_2026-05-12.csv")
    # In this legacy prepared dataset `Выпуск` empirically already includes
    # opening stock: Выпуск + net moves - Продано reproduces the reported
    # closing stock within one unit for ~97% of rows.
    balance = build_inventory_balance(
        daily,
        moves,
        produced_includes_opening_stock=True,
    )
    balance["old_release_stockout"] = (
        balance["sold"] / balance["produced"].replace(0.0, float("nan")) >= 0.90
    )

    output_dir = ROOT / "reports/stockout_inventory_balance_10"
    output_dir.mkdir(parents=True, exist_ok=True)
    balance.to_csv(output_dir / "inventory_balance.csv", index=False)
    balance[balance["is_inventory_stockout"]].to_csv(
        output_dir / "inventory_stockout_cases.csv", index=False
    )
    inconsistent = balance[~balance["balance_is_consistent"]].copy()
    inconsistent.reindex(
        inconsistent["balance_error"].abs().sort_values(ascending=False).index
    ).head(200).to_csv(output_dir / "largest_balance_errors.csv", index=False)

    summary = {
        "rows": int(len(balance)),
        "date_min": str(balance["date"].min().date()) if len(balance) else None,
        "date_max": str(balance["date"].max().date()) if len(balance) else None,
        "bakeries": sorted(balance["bakery_id"].unique().tolist()),
        "moves_rows_matched": int(
            (
                (balance["incoming_move_qty"] > 0) | (balance["outgoing_move_qty"] > 0)
            ).sum()
        ),
        "balance_consistent_rows": int(balance["balance_is_consistent"].sum()),
        "balance_consistent_share": float(balance["balance_is_consistent"].mean()),
        "inventory_stockouts": int(balance["is_inventory_stockout"].sum()),
        "old_release_stockouts": int(balance["old_release_stockout"].sum()),
        "both_stockout_flags": int(
            (balance["is_inventory_stockout"] & balance["old_release_stockout"]).sum()
        ),
        "median_abs_balance_error": float(balance["balance_error"].abs().median()),
        "p90_abs_balance_error": float(balance["balance_error"].abs().quantile(0.90)),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
