"""Walk-forward backtest for cold-start and mature-SKU corrections together."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.backtest_sku_systematic_correction import (  # noqa: E402
    DEFAULT_OUTPUT,
    build_summary,
    create_client,
    forecast_metrics,
    load_frame,
)
from src.experiments_v2.sku_cold_start import (  # noqa: E402
    ColdStartConfig,
    apply_category_neutral_cold_start,
    build_cold_start_registry,
)
from src.experiments_v2.sku_systematic_correction import (  # noqa: E402
    CorrectionConfig,
    apply_category_neutral_corrections,
    build_correction_registry,
)


OUTPUT = DEFAULT_OUTPUT.parent / "combined_sku_correction_backtest"


def main() -> None:
    date_to = pd.Timestamp("2026-07-28")
    test_days = 28
    mature_config = CorrectionConfig()
    cold_config = ColdStartConfig()
    date_from = date_to - pd.Timedelta(
        days=mature_config.history_days + test_days
    )
    frame = load_frame(
        create_client(".env"),
        date_from=date_from,
        date_to=date_to,
        excluded_product_ids=None,
    )
    mature_history = frame.copy()
    outputs: list[pd.DataFrame] = []
    mature_registries: list[pd.DataFrame] = []
    cold_registries: list[pd.DataFrame] = []
    test_start = date_to - pd.Timedelta(days=test_days - 1)

    for forecast_date in pd.date_range(test_start, date_to, freq="D"):
        day = frame[frame["date"].eq(forecast_date)].copy()
        if day.empty:
            continue
        cold_registry = build_cold_start_registry(
            frame,
            as_of_date=forecast_date,
            config=cold_config,
        )
        if not cold_registry.empty:
            cold_registry["registry_as_of"] = forecast_date
            cold_registries.append(cold_registry)
        cold_day = apply_category_neutral_cold_start(day, cold_registry)
        cold_day["forecast_qty"] = cold_day["cold_start_forecast_qty"]

        mature_registry = build_correction_registry(
            mature_history,
            as_of_date=forecast_date,
            config=mature_config,
        )
        if not mature_registry.empty:
            mature_registry["registry_as_of"] = forecast_date
            mature_registries.append(mature_registry)
        corrected = apply_category_neutral_corrections(
            cold_day,
            mature_registry,
        )
        corrected["base_forecast_qty"] = day["forecast_qty"].to_numpy()
        outputs.append(corrected)

    result = pd.concat(outputs, ignore_index=True)
    result["forecast_qty"] = result["base_forecast_qty"]
    mature_registry_history = pd.concat(
        mature_registries,
        ignore_index=True,
    )
    cold_registry_history = pd.concat(cold_registries, ignore_index=True)
    summary = build_summary(result, mature_registry_history)
    new_sku = result[result["product_id"].isin(cold_config.product_ids)]
    summary["new_sku"] = {
        "baseline": forecast_metrics(new_sku, forecast_col="forecast_qty"),
        "corrected": forecast_metrics(
            new_sku,
            forecast_col="corrected_forecast_qty",
        ),
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT / "backtest_rows.csv", index=False, encoding="utf-8-sig")
    mature_registry_history.to_csv(
        OUTPUT / "mature_registry_history.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cold_registry_history.to_csv(
        OUTPUT / "cold_registry_history.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
