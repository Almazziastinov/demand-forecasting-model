import pandas as pd

from src.partner_forecast_economics_service import PartnerForecastEconomicsService


def test_partner_summary_ranks_skus_and_keeps_week_history(tmp_path) -> None:
    pd.DataFrame(
        [
            dict(
                bakery_id=1, fold="w1", date_min="2026-01-01",
                date_max="2026-01-07", observed_days=7, actual_profit=100,
                model_profit=115, profit_delta=15, actual_underbake=10,
                model_underbake=8,
            ),
            dict(
                bakery_id=1, fold="w2", date_min="2026-01-08",
                date_max="2026-01-14", observed_days=7, actual_profit=120,
                model_profit=132, profit_delta=12, actual_underbake=9,
                model_underbake=7,
            ),
        ]
    ).to_csv(tmp_path / "bakery_week.csv", index=False)
    pd.DataFrame(
        [
            dict(
                bakery_id=1, fold="w1", product_id=1,
                product_name="A", category_group="выпечка", actual_volume=20,
                actual_profit=40, model_profit=50, profit_delta=10,
                actual_underbake=4, model_underbake=3,
            ),
            dict(
                bakery_id=1, fold="w2", product_id=1,
                product_name="A", category_group="выпечка", actual_volume=25,
                actual_profit=45, model_profit=53, profit_delta=8,
                actual_underbake=4, model_underbake=3,
            ),
            dict(
                bakery_id=1, fold="w1", product_id=2,
                product_name="B", category_group="фастфуд", actual_volume=100,
                actual_profit=30, model_profit=35, profit_delta=5,
                actual_underbake=2, model_underbake=1,
            ),
            dict(
                bakery_id=1, fold="w2", product_id=2,
                product_name="B", category_group="фастфуд", actual_volume=90,
                actual_profit=35, model_profit=39, profit_delta=4,
                actual_underbake=2, model_underbake=1,
            ),
        ]
    ).to_csv(tmp_path / "bakery_sku_week.csv", index=False)

    result = PartnerForecastEconomicsService(tmp_path).get_scope_summary({1})

    assert result is not None
    assert result["bakery_count"] == 1
    assert result["profit_delta"] == 27
    assert [row["product_name"] for row in result["top_sku"]] == ["B", "A"]
    assert result["top_sku"][1]["weekly_delta"] == [10.0, 8.0]

    bakery_group = PartnerForecastEconomicsService(tmp_path).get_scope_summary(
        {1}, "выпечка"
    )
    assert bakery_group is not None
    assert bakery_group["actual_profit"] == 85
    assert bakery_group["model_profit"] == 103
    assert [row["product_name"] for row in bakery_group["top_sku"]] == ["A"]
