from __future__ import annotations

from pathlib import Path

import pytest

from pipelines.forecast_publish import production_dataset_refresh
from pipelines.forecast_publish import run_production_inference as inference
from pipelines.forecast_publish.run_production_inference import (
    assert_nonprod_tables,
    build_parser,
    validate_profile_refresh_summary,
)


def test_dataset_refresh_defaults_use_bakery_daily_aggregate_template():
    parser = build_parser()
    args = parser.parse_args([])

    assert args.sql_template == str(production_dataset_refresh.DEFAULT_SQL_TEMPLATE)
    assert "clickhouse_bakery_daily_template.sql" in args.sql_template
    assert args.raw_output == str(production_dataset_refresh.DEFAULT_RAW_OUTPUT)
    assert "bakery_daily_sales_clickhouse.csv" in args.raw_output


def _write_guard_env(
    name: str,
    *,
    app_env: str = "dev",
    table_suffix: str = "_dev",
) -> Path:
    work_dir = Path("tests") / "_tmp_forecast_publish"
    work_dir.mkdir(parents=True, exist_ok=True)
    env_file = work_dir / name
    env_file.write_text(
        "\n".join(
            [
                f"APP_ENV={app_env}",
                "CLICKHOUSE_HOST=localhost",
                "CLICKHOUSE_PORT=8443",
                "CLICKHOUSE_USER=user",
                "CLICKHOUSE_PASSWORD=password",
                "CLICKHOUSE_DATABASE=demand_forecast",
                f"FORECAST_TABLE_SUFFIX={table_suffix}",
            ]
        ),
        encoding="utf-8",
    )
    return env_file


def test_nonprod_database_guard_accepts_dev_database():
    env_file = _write_guard_env(".env.guard.accept")

    assert_nonprod_tables(env_file)


def test_nonprod_database_guard_rejects_prod_app_env():
    env_file = _write_guard_env(".env.guard.prod", app_env="prod")

    with pytest.raises(RuntimeError, match="APP_ENV=prod"):
        assert_nonprod_tables(env_file)


def test_nonprod_table_guard_rejects_missing_suffix():
    env_file = _write_guard_env(".env.guard.suffix", table_suffix="")

    with pytest.raises(RuntimeError, match="FORECAST_TABLE_SUFFIX"):
        assert_nonprod_tables(env_file)


def test_profile_refresh_freshness_rejects_stale_summary(tmp_path: Path) -> None:
    summary = tmp_path / "profile.json"
    summary.write_text('{"date_to": "2026-07-10"}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="SKU profile is stale"):
        validate_profile_refresh_summary(
            summary,
            forecast_start="2026-07-20",
            max_age_days=8,
        )


def test_guard_failure_stops_before_run_load(monkeypatch, tmp_path: Path) -> None:
    args = build_parser().parse_args([])
    args.output_dir = str(tmp_path)
    args.activate_run = "none"
    bakery_path = tmp_path / "bakery.csv"
    bakery_path.write_text("date\n2026-07-22\n", encoding="utf-8")
    loaded = []

    monkeypatch.setattr(inference, "run_bakery_forecast", lambda *_: bakery_path)
    monkeypatch.setattr(
        inference,
        "allocate_from_clickhouse",
        lambda **_: (_ for _ in ()).throw(
            RuntimeError("Assortment coverage guard found established SKU")
        ),
    )
    monkeypatch.setattr(
        inference, "load_forecast_run", lambda **kwargs: loaded.append(kwargs)
    )

    with pytest.raises(RuntimeError, match="Assortment coverage guard"):
        inference.run_scenario(args, "base_raw_uplift")

    assert loaded == []


def test_guard_pass_allows_nonactivated_run_load(monkeypatch, tmp_path: Path) -> None:
    args = build_parser().parse_args([])
    args.output_dir = str(tmp_path)
    args.activate_run = "none"
    bakery_path = tmp_path / "bakery.csv"
    sku_day_path = tmp_path / "sku_day.csv"
    sku_hour_path = tmp_path / "sku_hour.csv"
    allocation_summary_path = tmp_path / "allocation.json"
    bakery_path.write_text("date\n2026-07-22\n", encoding="utf-8")
    sku_day_path.write_text("date\n2026-07-22\n", encoding="utf-8")
    sku_hour_path.write_text("date\n2026-07-22\n", encoding="utf-8")
    allocation_summary_path.write_text("{}", encoding="utf-8")
    loaded = []

    monkeypatch.setattr(inference, "run_bakery_forecast", lambda *_: bakery_path)
    monkeypatch.setattr(
        inference,
        "allocate_from_clickhouse",
        lambda **_: {
            "sku_daily": sku_day_path,
            "sku_hourly": sku_hour_path,
            "summary": allocation_summary_path,
        },
    )
    monkeypatch.setattr(
        inference,
        "load_forecast_run",
        lambda **kwargs: loaded.append(kwargs) or {"rows": 1},
    )

    result = inference.run_scenario(args, "base_raw_uplift")

    assert len(loaded) == 1
    assert loaded[0]["run_id"].startswith("prod_base_bakery_raw_uplift_sku_20260722")
    assert not result["activated"]
