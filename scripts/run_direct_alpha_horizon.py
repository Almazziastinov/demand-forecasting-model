"""Build a multi-day Direct alpha=.25 draft package without database writes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_current_direct_alpha_shadow import (  # noqa: E402
    build_input,
    export_publish_files,
)
from scripts.run_direct_alpha_shadow import concentration, run_shadow  # noqa: E402


def build_horizon(date_from: pd.Timestamp, date_to: pd.Timestamp, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    day_dirs: list[Path] = []
    sources: list[dict] = []
    for forecast_date in pd.date_range(date_from, date_to, freq="D"):
        day_dir = output / "days" / str(forecast_date.date())
        source_path = day_dir / "source.json"
        if (day_dir / "shadow_rows.parquet").exists() and source_path.exists():
            source = json.loads(source_path.read_text(encoding="utf-8"))
        else:
            for attempt in range(5):
                try:
                    input_path, source = build_input(forecast_date, day_dir)
                    run_shadow(input_path, day_dir)
                    source_path.write_text(
                        json.dumps(source, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(3 * (attempt + 1))
        day_dirs.append(day_dir)
        sources.append(source)

    source_run_ids = {item["source_run_id"] for item in sources}
    history_dates = {item["history_through"] for item in sources}
    if len(source_run_ids) != 1 or len(history_dates) != 1:
        raise RuntimeError(
            f"Horizon source changed during build: runs={source_run_ids}, "
            f"history={history_dates}"
        )

    rows = pd.concat(
        [pd.read_parquet(path / "shadow_rows.parquet") for path in day_dirs],
        ignore_index=True,
    )
    rows.to_parquet(output / "shadow_rows.parquet", index=False)
    publish_files = export_publish_files(output, next(iter(source_run_ids)))
    summary = {
        "contract": "direct_alpha_025_floor_tail_v1",
        "forecast_from": str(date_from.date()),
        "forecast_to": str(date_to.date()),
        "source_run_id": next(iter(source_run_ids)),
        "history_through": next(iter(history_dates)),
        "dates": int(rows["date"].nunique()),
        "bakeries": int(rows["bakery_id"].nunique()),
        "sku_day_rows": int(len(rows)),
        "selected_total": float(rows["selected_sku_forecast"].sum()),
        "tail_cap_rows": int(rows["tail_cap_applied"].sum()),
        "cold_start_fallback_bakery_days": int(
            rows.loc[rows["cold_start_fallback"], ["date", "bakery_id"]]
            .drop_duplicates()
            .shape[0]
        ),
        "concentration": concentration(rows, "selected_sku_forecast"),
        "publish_files": publish_files,
        "database_write": False,
        "activation": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date-from", type=pd.Timestamp, required=True)
    parser.add_argument("--date-to", type=pd.Timestamp, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.date_to < args.date_from:
        parser.error("--date-to must not be earlier than --date-from")
    summary = build_horizon(
        args.date_from.normalize(), args.date_to.normalize(), args.output_dir
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
