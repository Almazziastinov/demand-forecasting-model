from __future__ import annotations

from pipelines.forecast_publish import production_dataset_refresh
from pipelines.forecast_publish.run_production_inference import build_parser


def test_dataset_refresh_defaults_use_bakery_daily_aggregate_template():
    parser = build_parser()
    args = parser.parse_args([])

    assert args.sql_template == str(production_dataset_refresh.DEFAULT_SQL_TEMPLATE)
    assert "clickhouse_bakery_daily_template.sql" in args.sql_template
    assert args.raw_output == str(production_dataset_refresh.DEFAULT_RAW_OUTPUT)
    assert "bakery_daily_sales_clickhouse.csv" in args.raw_output
