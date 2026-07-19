"""Audit local inputs needed for stockout-demand research.

This script is intentionally local-only. It does not connect to ClickHouse and
does not mutate production state. Its job is to answer whether the workstation
already has enough raw files to build the inventory-aware stockout dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "stockout_research_inputs"

PILOT_BAKERY_IDS = {16, 20, 21, 22, 28, 80, 89, 107, 221, 222, 257}

EXPECTED_INPUTS = {
    "pilot_daily_mart_zero": ROOT / "data" / "raw" / "pilot_mart_zero_sales_2026-04-30_2026-07-19.csv",
    "pilot_hourly_sales": ROOT / "data" / "raw" / "pilot_stg_check_lines_2026-04-30_2026-07-19.csv",
    "legacy_hourly_sales": ROOT / "data" / "raw" / "sales_stg_2025_2026.csv",
    "legacy_moves": ROOT / "data" / "raw" / "moves_clickhouse_2025-01-15_2026-05-12.csv",
    "legacy_daily_preprocessed": ROOT / "data" / "processed" / "preprocessed_data_merged.csv",
}

EXPECTED_COLUMNS = {
    "pilot_daily_mart_zero": {
        "date",
        "bakery_id",
        "bakery_name",
        "product_id",
        "product_name",
        "category_name",
        "qty_sold",
        "qty_produced",
        "qty_received",
        "qty_sent",
        "stock_balance",
    },
    "pilot_hourly_sales": {
        "check_datetime",
        "check_date",
        "quantity",
        "bakery_id",
        "bakery_name",
        "product_id",
        "product_name",
        "category_name",
    },
    "legacy_hourly_sales": {
        "check_datetime",
        "check_date",
        "cash_event_type",
        "quantity",
        "bakery_id",
        "bakery_name",
        "product_id",
        "product_name",
        "category_name",
    },
    "legacy_moves": {
        "move_id",
        "move_date",
        "product_id",
        "sender_id",
        "receiver_id",
        "quantity",
    },
    "legacy_daily_preprocessed": {
        "Дата",
        "Пекарня",
        "Номенклатура",
        "Продано",
        "Выпуск",
        "Остаток",
        "stock_lag1",
    },
}


def sniff_csv(path: Path) -> tuple[str, str]:
    first_line_bytes = path.read_bytes().splitlines()[0] if path.stat().st_size else b""
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        text = first_line_bytes.decode(encoding, errors="replace")
        first_line = text.strip()
        delimiter_counts = {delimiter: first_line.count(delimiter) for delimiter in [",", ";", "\t"]}
        best_delimiter, best_count = max(delimiter_counts.items(), key=lambda item: item[1])
        if best_count > 0:
            return encoding, best_delimiter
        try:
            dialect = csv.Sniffer().sniff(first_line, delimiters=[",", ";", "\t"])
            return encoding, dialect.delimiter
        except csv.Error:
            continue
    return "utf-8-sig", ","


def read_columns(path: Path) -> tuple[list[str], str, str]:
    detected_encoding, delimiter = sniff_csv(path)
    encodings = [detected_encoding, "utf-8-sig", "utf-8", "cp1251"]
    last_error: Exception | None = None
    for encoding in dict.fromkeys(encodings):
        try:
            frame = pd.read_csv(path, nrows=0, encoding=encoding, sep=delimiter)
            return list(frame.columns), encoding, delimiter
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise ValueError(f"Could not read CSV header: {path}")


def summarize_csv_dates(
    path: Path,
    *,
    date_col: str,
    bakery_col: str | None = None,
    chunk_size: int = 500_000,
) -> dict[str, Any]:
    columns, encoding, delimiter = read_columns(path)
    usecols = [date_col]
    if bakery_col and bakery_col in columns:
        usecols.append(bakery_col)

    rows = 0
    date_min: pd.Timestamp | None = None
    date_max: pd.Timestamp | None = None
    pilot_rows = 0
    pilot_bakeries: set[int] = set()

    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        encoding=encoding,
        sep=delimiter,
        chunksize=chunk_size,
    ):
        rows += len(chunk)
        dates = pd.to_datetime(chunk[date_col], errors="coerce")
        valid_dates = dates.dropna()
        if not valid_dates.empty:
            current_min = valid_dates.min()
            current_max = valid_dates.max()
            date_min = current_min if date_min is None else min(date_min, current_min)
            date_max = current_max if date_max is None else max(date_max, current_max)

        if bakery_col and bakery_col in chunk.columns:
            bakery_ids = pd.to_numeric(chunk[bakery_col], errors="coerce")
            pilot_mask = bakery_ids.isin(PILOT_BAKERY_IDS)
            pilot_rows += int(pilot_mask.sum())
            pilot_bakeries.update(
                int(value) for value in bakery_ids[pilot_mask].dropna().unique()
            )

    return {
        "rows": rows,
        "date_min": None if date_min is None else str(date_min.date()),
        "date_max": None if date_max is None else str(date_max.date()),
        "pilot_rows": pilot_rows if bakery_col else None,
        "pilot_bakeries": sorted(pilot_bakeries) if bakery_col else None,
    }


def audit_expected_input(name: str, path: Path, *, full_scan: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "required_columns": sorted(EXPECTED_COLUMNS[name]),
    }
    if not path.exists():
        result["status"] = "missing"
        return result

    result["size_bytes"] = path.stat().st_size
    columns, encoding, delimiter = read_columns(path)
    missing = sorted(EXPECTED_COLUMNS[name] - set(columns))
    result.update(
        {
            "status": "ok" if not missing else "bad_schema",
            "encoding": encoding,
            "delimiter": delimiter,
            "columns": columns,
            "missing_columns": missing,
        }
    )

    if full_scan and not missing:
        if name in {"pilot_hourly_sales", "legacy_hourly_sales"}:
            result["scan"] = summarize_csv_dates(
                path, date_col="check_date", bakery_col="bakery_id"
            )
        elif name == "legacy_moves":
            result["scan"] = summarize_csv_dates(path, date_col="move_date")
        elif name == "legacy_daily_preprocessed":
            result["scan"] = summarize_csv_dates(path, date_col="Дата")
        elif name == "pilot_daily_mart_zero":
            result["scan"] = summarize_csv_dates(path, date_col="date", bakery_col="bakery_id")
    return result


def audit_local_candidates() -> list[dict[str, Any]]:
    candidates = []
    for path in sorted((ROOT / "data" / "raw").glob("*")):
        item: dict[str, Any] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "suffix": path.suffix.lower(),
        }
        if path.suffix.lower() == ".csv":
            try:
                columns, encoding, delimiter = read_columns(path)
                item.update(
                    {
                        "encoding": encoding,
                        "delimiter": delimiter,
                        "columns": columns,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - diagnostic script
                item["error"] = f"{type(exc).__name__}: {exc}"
        candidates.append(item)
    return candidates


def build_recommendations(expected: dict[str, dict[str, Any]]) -> list[str]:
    recommendations = []
    pilot_ready = all(
        expected[name]["status"] == "ok"
        for name in ["pilot_daily_mart_zero", "pilot_hourly_sales"]
    )
    legacy_ready = all(
        expected[name]["status"] == "ok"
        for name in ["legacy_hourly_sales", "legacy_moves", "legacy_daily_preprocessed"]
    )

    if pilot_ready:
        recommendations.append(
            "Pilot mart_zero inputs are ready. Run analyze_pilot_mart_zero_stockout_balance.py next."
        )
    if legacy_ready:
        recommendations.append(
            "Legacy long-history inputs are ready. Run the older stockout-demand scripts if you need the March/long-profile workflow."
        )
    if not pilot_ready:
        for name in ["pilot_daily_mart_zero", "pilot_hourly_sales"]:
            result = expected[name]
            if result["status"] == "missing":
                recommendations.append(f"Create pilot input {name}: {result['path']}")
            elif result["status"] == "bad_schema":
                missing = ", ".join(result["missing_columns"])
                recommendations.append(f"Fix schema for pilot input {name}; missing columns: {missing}")
    if not legacy_ready:
        recommendations.append(
            "Legacy long-history inputs are incomplete; restore them from the previous workstation or export them separately."
        )
    return recommendations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit local stockout-demand research input files."
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Scan whole CSVs for row counts, date ranges, and pilot bakery coverage.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for JSON audit output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected = {
        name: audit_expected_input(name, path, full_scan=args.full_scan)
        for name, path in EXPECTED_INPUTS.items()
    }
    report = {
        "full_scan": args.full_scan,
        "pilot_bakery_ids": sorted(PILOT_BAKERY_IDS),
        "expected_inputs": expected,
        "local_raw_candidates": audit_local_candidates(),
        "recommendations": build_recommendations(expected),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "input_audit.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote: {output_path}")


if __name__ == "__main__":
    main()
