import pandas as pd

from pipelines.forecast_publish.nightly_refresh import build_run_id
from pipelines.forecast_publish.nightly_refresh import resolve_default_dates


def test_resolve_default_dates_uses_previous_day_for_history_end() -> None:
    dates = resolve_default_dates(pd.Timestamp("2026-05-19 00:00:00"))
    assert dates.history_end_date == "2026-05-18"
    assert dates.forecast_start_date == "2026-05-19"


def test_build_run_id_contains_forecast_start_and_horizon() -> None:
    run_id = build_run_id(forecast_start_date="2026-05-19", horizon_days=7, prefix="nightly")
    assert run_id == "nightly_20260519_h07"
