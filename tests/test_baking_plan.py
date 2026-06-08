from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = ROOT / "apps" / "forecast_embedded"
sys.path.insert(0, str(APP_ROOT))

from app.services.baking_plan import BakingWindow  # noqa: E402
from app.services.baking_plan import allocate_template_row  # noqa: E402
from app.services.baking_plan import build_product_hour_lookup  # noqa: E402
from app.services.baking_plan import coverage_hours  # noqa: E402
from app.services.baking_plan import normalize_sku_name  # noqa: E402
from app.services.baking_plan import revenue_bucket  # noqa: E402
from app.services.baking_plan import sku_match_keys  # noqa: E402


def test_coverage_hours_assigns_sales_until_next_bake_window() -> None:
    windows = [
        BakingWindow(column=3, label="4:00-7:00", start_hour=4, end_hour=7),
        BakingWindow(column=8, label="11:00-12:00", start_hour=11, end_hour=12),
    ]

    assert coverage_hours(windows)[3] == [6, 7, 8, 9, 10, 11]
    assert coverage_hours(windows)[8][0] == 12


def test_allocate_template_row_sums_hourly_forecast_by_coverage_window() -> None:
    rows = [
        {"product_name": "Треугольник курица", "hour": 6, "forecast_qty": 3},
        {"product_name": "Треугольник курица", "hour": 7, "forecast_qty": 12},
        {"product_name": "Треугольник курица", "hour": 8, "forecast_qty": 15},
        {"product_name": "Треугольник курица", "hour": 9, "forecast_qty": 15},
        {"product_name": "Треугольник курица", "hour": 10, "forecast_qty": 20},
        {"product_name": "Треугольник курица", "hour": 11, "forecast_qty": 30},
        {"product_name": "Треугольник курица", "hour": 12, "forecast_qty": 27},
    ]
    lookup = build_product_hour_lookup(rows)
    windows = [
        BakingWindow(column=3, label="4:00-7:00", start_hour=4, end_hour=7),
        BakingWindow(column=8, label="11:00-12:00", start_hour=11, end_hour=12),
    ]

    result = allocate_template_row(
        template_sku_name="Треугольник курица (тесто ночного брожжения)",
        row_windows=windows,
        product_hour_lookup=lookup,
    )

    assert result[3] == 95
    assert result[8] == 27


def test_normalize_sku_name_removes_template_notes() -> None:
    assert (
        normalize_sku_name("Сосиска в тесте (ночная дефростация)")
        == "сосиска в тесте"
    )


def test_revenue_bucket_matches_template_thresholds() -> None:
    assert revenue_bucket(1_499_999) == "до 1,5 млн"
    assert revenue_bucket(1_500_000) == "до 2,5 млн"
    assert revenue_bucket(2_500_000) == "от 2,5 млн"
    assert revenue_bucket(3_000_000) == "от 3млн"


def test_sku_match_keys_include_known_forecast_aliases() -> None:
    assert "треугольник курица" in sku_match_keys("Треугольник курица безд")
    assert "жар пицца оригинальная" in sku_match_keys("ЖарПицца Оригинальная")
    assert "пирожок булочка с яблоками" in sku_match_keys("Пирожок яблоко")


def test_allocate_template_row_uses_alias_lookup() -> None:
    rows = [
        {"product_name": "Треугольник курица безд", "hour": 6, "forecast_qty": 10},
        {"product_name": "Треугольник курица безд", "hour": 12, "forecast_qty": 5},
    ]
    lookup = build_product_hour_lookup(rows)
    windows = [
        BakingWindow(column=3, label="4:00-7:00", start_hour=4, end_hour=7),
        BakingWindow(column=8, label="11:00-12:00", start_hour=11, end_hour=12),
    ]

    result = allocate_template_row(
        template_sku_name="Треугольник курица  (тесто ночного брожжения)",
        row_windows=windows,
        product_hour_lookup=lookup,
    )

    assert result == {3: 10, 8: 5}
