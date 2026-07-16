"""Combine the legacy 2026 review report with the current no-coffee report."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_city_detail_analytics as base


OLD_FILE = Path(r"C:\Users\dns\Desktop\reviews_analysis_2026.xlsx")
NEW_FILE = Path(
    r"C:\Users\dns\Documents\Codex\2026-06-30\new-chat\outputs\Отзывы покупателей единый реестр аналитика API категории без кофеен v2.xlsx"
)
OUTPUT_FILE = Path(
    r"C:\Users\dns\Documents\Codex\2026-06-30\new-chat\outputs\Отзывы покупателей объединенный отчет.xlsx"
)

DETAIL_CATEGORIES = [
    "Низкое качество еды",
    "Невежливое обслуживание",
    "Несвежие продукты",
    "Проблемы с чистотой",
    "Сервис",
    "Недостаток ассортимента",
    "Ошибки при расчетах",
    "Наличие вредителей",
    "Плохая упаковка товаров",
    "Другое",
]


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def text_key(value: Any) -> str:
    return " ".join(clean(value).lower().split())


def parse_date(value: Any) -> Any:
    if isinstance(value, datetime):
        return value
    value = clean(value)
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d", "%d.%m.%Y", "%d.%m.%y"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return value or None


def normalize_detail(value: Any) -> str:
    value = clean(value)
    return value if value in DETAIL_CATEGORIES else ("Другое" if value else "")


def sentiment_to_type(sentiment: str) -> str:
    sentiment = clean(sentiment).lower()
    if sentiment == "позитивный":
        return "Благодарность"
    if sentiment == "негативный":
        return "Жалоба"
    if sentiment == "нейтральный":
        return "Нейтральный"
    return sentiment.capitalize() if sentiment else "Неизвестно"


def type_to_sentiment(review_type: str) -> str:
    review_type = clean(review_type)
    if review_type == "Благодарность":
        return "позитивный"
    if review_type == "Жалоба":
        return "негативный"
    if review_type == "Нейтральный":
        return "нейтральный"
    return ""


def read_new_rows() -> list[dict[str, Any]]:
    wb = base.openpyxl.load_workbook(NEW_FILE, data_only=True)
    ws = wb["Единый реестр отзывов"]
    headers = [clean(cell.value) for cell in ws[1]]
    rows: list[dict[str, Any]] = []
    for values in ws.iter_rows(min_row=2, values_only=True):
        row = dict(zip(headers, values))
        review_type = clean(row.get("тип_отзыва"))
        rows.append(
            {
                "источник_отчета": "текущий без кофеен",
                "дата": row.get("дата"),
                "год": row.get("год"),
                "месяц": row.get("месяц"),
                "источник": row.get("источник"),
                "адрес": row.get("адрес") or "Неизвестно",
                "город": row.get("город") or "Неизвестно",
                "тип_отзыва": review_type,
                "тональность": type_to_sentiment(review_type),
                "категория": row.get("категория_нормализованная") or row.get("категория_исходная"),
                "детальная_категория": normalize_detail(row.get("детальная_категория")),
                "текст_отзыва": row.get("текст_отзыва"),
                "формат_точки": row.get("формат_точки"),
                "исходный_id": "",
                "исходный_лист": row.get("исходный_лист"),
                "исходная_строка": row.get("исходная_строка"),
                "дубликат_по_тексту": "нет",
            }
        )
    return rows


def read_old_rows(existing_texts: set[str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    wb = base.openpyxl.load_workbook(OLD_FILE, data_only=True)
    ws = wb["Все отзывы"]
    headers = [clean(cell.value) for cell in ws[1]]
    rows: list[dict[str, Any]] = []
    seen_old: set[str] = set()
    stats = {"old_rows": 0, "skipped_overlap_with_new": 0, "skipped_old_duplicate": 0}

    for values in ws.iter_rows(min_row=2, values_only=True):
        stats["old_rows"] += 1
        item = dict(zip(headers, values))
        key = text_key(item.get("Текст отзыва"))
        if key and key in existing_texts:
            stats["skipped_overlap_with_new"] += 1
            continue
        if key and key in seen_old:
            stats["skipped_old_duplicate"] += 1
            continue
        if key:
            seen_old.add(key)

        date = parse_date(item.get("Дата отзыва") or item.get("Дата сообщения"))
        review_type = sentiment_to_type(clean(item.get("Тональность")))
        rows.append(
            {
                "источник_отчета": "старый reviews_analysis_2026",
                "дата": date,
                "год": date.year if isinstance(date, datetime) else "",
                "месяц": date.month if isinstance(date, datetime) else "",
                "источник": item.get("Источник"),
                "адрес": item.get("Адрес") or "Неизвестно",
                "город": item.get("Город") or "Неизвестно",
                "тип_отзыва": review_type,
                "тональность": clean(item.get("Тональность")),
                "категория": "",
                "детальная_категория": normalize_detail(item.get("Категория проблемы")),
                "текст_отзыва": item.get("Текст отзыва"),
                "формат_точки": "",
                "исходный_id": item.get("ID"),
                "исходный_лист": "reviews_analysis_2026/Все отзывы",
                "исходная_строка": stats["old_rows"] + 1,
                "дубликат_по_тексту": "нет",
            }
        )
    return rows, stats


def enrich_old_rows(rows: list[dict[str, Any]]) -> None:
    old_rows = [row for row in rows if row.get("источник_отчета") == "старый reviews_analysis_2026"]
    if not old_rows:
        return
    base.enrich_rows(old_rows, {})
    for row in old_rows:
        row["детальная_категория"] = normalize_detail(row.get("детальная_категория"))


def date_sort_value(row: dict[str, Any]) -> tuple[int, Any]:
    value = row.get("дата")
    return (0, value) if isinstance(value, datetime) else (1, clean(value))


def style_sheet(sheet) -> None:
    base.style_header(sheet)
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions


def write_registry(wb, rows: list[dict[str, Any]]) -> None:
    ws = wb.create_sheet("Единый реестр")
    headers = [
        "источник_отчета",
        "дата",
        "год",
        "месяц",
        "источник",
        "адрес",
        "город",
        "тип_отзыва",
        "тональность",
        "категория",
        "детальная_категория",
        "текст_отзыва",
        "формат_точки",
        "исходный_id",
        "исходный_лист",
        "исходная_строка",
        "дубликат_по_тексту",
    ]
    ws.append(headers)
    for row in rows:
        ws.append([row.get(header) for header in headers])
    style_sheet(ws)
    widths = [26, 16, 10, 10, 22, 36, 22, 18, 16, 28, 28, 90, 16, 14, 28, 14, 18]
    for index, width in enumerate(widths, 1):
        ws.column_dimensions[base.get_column_letter(index)].width = width
    for cell in ws["B"]:
        if isinstance(cell.value, datetime):
            cell.number_format = "yyyy-mm-dd"


def write_summary(wb, rows: list[dict[str, Any]], stats: dict[str, int]) -> None:
    ws = wb.create_sheet("Общая сводка")
    ws["A1"] = "Объединенный отчет отзывов"
    ws["A1"].font = base.Font(bold=True, size=16)
    source_counts = Counter(row.get("источник_отчета") for row in rows)
    type_counts = Counter(row.get("тип_отзыва") for row in rows)
    sentiment_counts = Counter(row.get("тональность") for row in rows)
    dates = [row.get("дата") for row in rows if isinstance(row.get("дата"), datetime)]
    data = [
        ("Всего отзывов в объединенном отчете", len(rows)),
        ("Из текущего отчета без кофеен", source_counts["текущий без кофеен"]),
        ("Добавлено из старого отчета", source_counts["старый reviews_analysis_2026"]),
        ("Старых строк пропущено как дубли нового отчета", stats["skipped_overlap_with_new"]),
        ("Старых строк пропущено как внутренние дубли", stats["skipped_old_duplicate"]),
        ("Благодарностей", type_counts["Благодарность"]),
        ("Жалоб", type_counts["Жалоба"]),
        ("Нейтральных", type_counts["Нейтральный"]),
        ("Позитивная тональность", sentiment_counts["позитивный"]),
        ("Негативная тональность", sentiment_counts["негативный"]),
        ("Нейтральная тональность", sentiment_counts["нейтральный"]),
        ("Минимальная дата", min(dates) if dates else ""),
        ("Максимальная дата", max(dates) if dates else ""),
    ]
    ws.append([])
    for label, value in data:
        ws.append([label, value])
    ws.column_dimensions["A"].width = 48
    ws.column_dimensions["B"].width = 18
    for cell in ws["B"]:
        if isinstance(cell.value, datetime):
            cell.number_format = "yyyy-mm-dd"


def negative_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("тип_отзыва") == "Жалоба" or clean(row.get("тональность")).lower() == "негативный"
    ]


def write_negative_categories(wb, rows: list[dict[str, Any]]) -> list[str]:
    complaints = negative_rows(rows)
    counts = Counter(row.get("детальная_категория") or "Другое" for row in complaints)
    total = len(complaints)
    ws = wb.create_sheet("Категории негатива")
    ws.append([f"Категории проблем в негативных отзывах / жалобах (всего: {total})", "", ""])
    ws.append(["Категория", "Количество", "Доля"])
    ordered = [cat for cat in DETAIL_CATEGORIES if counts[cat]]
    for cat in ordered:
        ws.append([cat, counts[cat], counts[cat] / total if total else 0])
    base.style_header(ws)
    ws.column_dimensions["A"].width = 34
    ws.column_dimensions["B"].width = 14
    ws.column_dimensions["C"].width = 12
    for cell in ws["C"]:
        cell.number_format = "0.0%"
    return ordered


def write_pivot(wb, rows: list[dict[str, Any]], key: str, sheet_name: str, categories: list[str]) -> None:
    ws = wb.create_sheet(sheet_name)
    headers = [
        key,
        "Всего",
        "Благодарностей",
        "Жалоб",
        "Нейтральных",
        "% благодарностей",
        "% жалоб",
        "% нейтральных",
        *categories,
    ]
    ws.append(headers)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get(key) or "Неизвестно"].append(row)
    for group_key, items in sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True):
        count = len(items)
        type_counts = Counter(item.get("тип_отзыва") for item in items)
        negative_counts = Counter(
            item.get("детальная_категория") or "Другое" for item in negative_rows(items)
        )
        ws.append(
            [
                group_key,
                count,
                type_counts["Благодарность"],
                type_counts["Жалоба"],
                type_counts["Нейтральный"],
                type_counts["Благодарность"] / count if count else 0,
                type_counts["Жалоба"] / count if count else 0,
                type_counts["Нейтральный"] / count if count else 0,
                *[negative_counts[cat] for cat in categories],
            ]
        )
    style_sheet(ws)
    ws.column_dimensions["A"].width = 34
    for col in range(2, ws.max_column + 1):
        ws.column_dimensions[base.get_column_letter(col)].width = 15
    for row in ws.iter_rows(min_row=2, min_col=6, max_col=8):
        for cell in row:
            cell.number_format = "0.0%"


def write_all_reviews(wb, rows: list[dict[str, Any]]) -> None:
    ws = wb.create_sheet("Все отзывы")
    headers = [
        "Дата",
        "Год",
        "Источник",
        "Адрес",
        "Город",
        "Тип отзыва",
        "Тональность",
        "Категория",
        "Детальная категория",
        "Текст отзыва",
        "Источник отчета",
    ]
    ws.append(headers)
    for row in rows:
        ws.append(
            [
                row.get("дата"),
                row.get("год"),
                row.get("источник"),
                row.get("адрес"),
                row.get("город"),
                row.get("тип_отзыва"),
                row.get("тональность"),
                row.get("категория"),
                row.get("детальная_категория"),
                row.get("текст_отзыва"),
                row.get("источник_отчета"),
            ]
        )
    style_sheet(ws)
    widths = [16, 10, 22, 36, 22, 16, 16, 28, 28, 90, 28]
    for index, width in enumerate(widths, 1):
        ws.column_dimensions[base.get_column_letter(index)].width = width
    for cell in ws["A"]:
        if isinstance(cell.value, datetime):
            cell.number_format = "yyyy-mm-dd"


def main() -> None:
    new_rows = read_new_rows()
    new_texts = {text_key(row.get("текст_отзыва")) for row in new_rows if text_key(row.get("текст_отзыва"))}
    old_rows, stats = read_old_rows(new_texts)
    rows = new_rows + old_rows
    enrich_old_rows(rows)
    rows.sort(key=date_sort_value)

    wb = base.openpyxl.Workbook()
    del wb[wb.sheetnames[0]]
    write_registry(wb, rows)
    write_summary(wb, rows, stats)
    categories = write_negative_categories(wb, rows)
    write_pivot(wb, rows, "адрес", "По пекарням", categories)
    write_pivot(wb, rows, "город", "По городам", categories)
    write_all_reviews(wb, rows)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUTPUT_FILE)
    print(f"new_rows={len(new_rows)}")
    print(f"old_added_rows={len(old_rows)}")
    print(f"stats={stats}")
    print(f"combined_rows={len(rows)}")
    print(f"output={OUTPUT_FILE}")


if __name__ == "__main__":
    main()
