"""Build final API-detail analytics workbook excluding coffee shop rows."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_city_detail_analytics as base


API_DETAIL_FILE = Path(
    r"C:\Users\dns\Desktop\Projects\demand-forecasting-model\outputs\detail_categories_api_checkpoint.json"
)
OUTPUT_FILE = Path(
    r"C:\Users\dns\Documents\Codex\2026-06-30\new-chat\outputs\Отзывы покупателей единый реестр аналитика API категории без кофеен.xlsx"
)


def normalize_api_detail(detail: str, normalized_category: str) -> str:
    if detail != "Другое":
        return detail
    if normalized_category == "Сервис":
        return "Сервис"
    if normalized_category == "Качество продукции":
        return "Низкое качество еды"
    return detail


def is_coffee_shop(row: dict) -> bool:
    return str(row.get("формат_точки") or "").strip().lower() == "кофейня"


def main() -> None:
    if "Сервис" not in base.DETAIL_CATEGORIES:
        insert_at = base.DETAIL_CATEGORIES.index("Другое")
        base.DETAIL_CATEGORIES.insert(insert_at, "Сервис")

    wb = base.openpyxl.load_workbook(base.INPUT_FILE)
    rows = base.read_rows(wb)
    address_mapping = base.load_address_mapping(wb)
    base.enrich_rows(rows, address_mapping)
    base.fill_remaining_cities_with_api(rows)

    api_details = json.loads(API_DETAIL_FILE.read_text(encoding="utf-8"))
    for row in rows:
        if row.get("тип_отзыва") != "Жалоба":
            row["детальная_категория"] = ""
            continue
        key = f"{row.get('исходный_лист')}|{row.get('исходная_строка')}"
        row["детальная_категория"] = normalize_api_detail(
            api_details.get(key, "Другое"),
            row.get("категория_нормализованная", ""),
        )

    before = len(rows)
    removed_rows = [row for row in rows if is_coffee_shop(row)]
    rows = [row for row in rows if not is_coffee_shop(row)]
    removed_by_type = Counter(row.get("тип_отзыва") for row in removed_rows)

    base.clear_sheets(wb)
    base.rewrite_registry(wb, rows)
    base.write_summary(wb, rows)
    cats = base.write_negative_categories(wb, rows)
    base.write_pivot(wb, rows, "адрес", "По пекарням", cats)
    base.write_pivot(wb, rows, "город", "По городам", cats)
    base.write_all_reviews(wb, rows)
    base.write_city_fill(wb, rows)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUTPUT_FILE)
    print(f"input_rows={before}")
    print(f"removed_coffee_shop_rows={len(removed_rows)}")
    print(f"removed_by_type={dict(removed_by_type)}")
    print(f"kept_rows={len(rows)}")
    print(f"output={OUTPUT_FILE}")


if __name__ == "__main__":
    main()
