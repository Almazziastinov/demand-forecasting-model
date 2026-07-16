"""Normalize bakery addresses in the current no-coffee review analytics workbook."""

from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_city_detail_analytics as base


INPUT_FILE = Path(
    r"C:\Users\dns\Documents\Codex\2026-06-30\new-chat\outputs\Отзывы покупателей единый реестр аналитика API категории без кофеен v2.xlsx"
)
OUTPUT_FILE = Path(
    r"C:\Users\dns\Documents\Codex\2026-06-30\new-chat\outputs\Отзывы покупателей единый реестр аналитика API категории без кофеен адреса нормализованы.xlsx"
)

CITY_ALIASES = {
    "наб челны": "Набережные Челны",
    "наб. челны": "Набережные Челны",
    "наб.челны": "Набережные Челны",
    "заниск": "Заинск",
    ". чебоксары": "Чебоксары",
    "казаеь": "Казань",
    "ул. сибирский тракт, д. 25": "Казань",
    "новокунецк": "Новокузнецк",
    "новокузнцк": "Новокузнецк",
    "новокузнец": "Новокузнецк",
    "зеленодольчк": "Зеленодольск",
    "зеленодольс": "Зеленодольск",
}

STREET_PREFIX_RE = re.compile(
    r"\b(ул|улица|проспект|пр-т|пр|д|дом)\.?\b",
    flags=re.IGNORECASE,
)
HOUSE_RE = re.compile(
    r"\b\d+(?:[а-вгдеёжзиклмнопрстуфхцчшщэюa-z]|[кст]\d+)?(?:/\d+(?:[а-вгдеёжзиклмнопрстуфхцчшщэюa-z]|[кст]\d+)?)?\b",
    flags=re.IGNORECASE,
)
NOISE_WORD_RE = re.compile(
    r"\b(кофейня|отзыв|жалоба|горячая|линия|2гис|2\s+гис|гис|яндекс|справочник|вконтакте|вк)\b",
    flags=re.IGNORECASE,
)


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def norm_text(value: Any) -> str:
    value = clean(value).lower().replace("ё", "е")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_city(value: Any) -> str:
    city = base.normalize_city(value)
    return CITY_ALIASES.get(norm_text(city), city)


def house_index(parts: list[str]) -> int | None:
    matches = [index for index, part in enumerate(parts) if HOUSE_RE.fullmatch(part)]
    if not matches:
        return None
    for index in matches:
        if index > 0:
            return index
    return matches[0]


def remove_city_words(text: str, city: str, city_patterns: list[str]) -> str:
    result = text
    city_values = set(city_patterns)
    if city and city != "Неизвестно":
        city_values.add(city)
    for alias, canonical in CITY_ALIASES.items():
        city_values.add(alias)
        city_values.add(canonical)
    for value in sorted(city_values, key=lambda item: len(str(item)), reverse=True):
        value = clean(value)
        if not value:
            continue
        result = re.sub(
            r"\b" + re.escape(value).replace(r"\ ", r"\s+") + r"\b",
            " ",
            result,
            flags=re.IGNORECASE,
        )
        if len(value) > 3:
            result = re.sub(re.escape(value) + r"\b", " ", result, flags=re.IGNORECASE)
    return result


def address_key(address: str, city: str, city_patterns: list[str]) -> tuple[str, str, str]:
    city = normalize_city(city)
    text = norm_text(address)
    text = re.sub(r"\+?\d[\d\s()+-]{8,}", " ", text)
    text = re.sub(r"\b20[2-3]\d\b", " ", text)
    text = re.sub(r"\b(\d+)\s*[- ]\s*(я|й|ая|ый)\b", r"\1\2", text)
    text = re.sub(r"\b(?:г\.|г\s+|город\s+)", " ", text)
    text = re.sub(r"\b2\s*г\s*и\s*с", " ", text, flags=re.IGNORECASE)
    text = NOISE_WORD_RE.sub(" ", text)
    text = remove_city_words(text, city, city_patterns)
    text = STREET_PREFIX_RE.sub(" ", text)
    text = re.sub(r"[^а-яa-z0-9/]+", " ", text)
    parts = text.split()

    index = house_index(parts)
    if index is None:
        return city, " ".join(parts), ""

    house = parts[index].upper()
    street = " ".join(parts[:index])
    return city, street, house


def valid_key(key: tuple[str, str, str]) -> bool:
    city, street, house = key
    return bool(city and city != "Неизвестно" and street and house)


def display_from_address(address: str, city: str, city_patterns: list[str]) -> str:
    city = normalize_city(city)
    text = clean(address)
    text = re.sub(r"\+?\d[\d\s()+-]{8,}", " ", text)
    text = re.sub(r"\b20[2-3]\d\b", " ", text)
    text = re.sub(r"\b(\d+)\s*[- ]\s*(я|й|ая|ый)\b", r"\1\2", text)
    text = re.sub(r"\b(?:г\.|г\s+|город\s+)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b2\s*г\s*и\s*с", " ", text, flags=re.IGNORECASE)
    text = NOISE_WORD_RE.sub(" ", text)
    text = remove_city_words(text, city, city_patterns)
    text = re.sub(r"\s+", " ", text).strip(" .,")
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\bулица\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+улица\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" .,")
    parts = text.split()
    index = house_index(parts)
    if index is not None:
        text = " ".join(parts[: index + 1]).strip(" .,")
    if not text:
        return "Неизвестно"
    if city and city != "Неизвестно":
        return f"{city}, {text}"
    return text


def extract_address_candidate(text: str) -> str:
    text = clean(text).replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    patterns = [
        r"(?:Отзыв\s+)?(?:2ГИС|Яндекс Справочник|ВКонтакте|Вконтакте)[.,]\s*(?P<addr>[^.]{3,90}?)(?=\.?\s+\d{1,2}[./]\d{1,2}[./]\d{2,4}|[.])",
        r"(?:Жалоба|Отзыв)\s+горячая\s+линия\.\s*(?:\+?\d[\d\s()+-]{7,}\.?\s*)?(?P<addr>[^.]{3,90}?)(?=\.)",
        r"\b(?:по адресу|адресу)\s+(?P<addr>[^.]{3,90}?)(?=\.)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = base.normalize_extracted_address(match.group("addr"))
        candidate = re.sub(r"^\+?\d[\d\s()+-]{7,}\.?\s*", "", candidate).strip(" .,")
        if candidate:
            return candidate
    return ""


def build_canonical_map(rows: list[dict[str, Any]], city_patterns: list[str]) -> dict[tuple[str, str, str], str]:
    variants: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        address = clean(row.get("адрес"))
        city = normalize_city(row.get("город"))
        key = address_key(address, city, city_patterns)
        if not valid_key(key):
            continue
        variants[key][display_from_address(address, city, city_patterns)] += 1

    canonical: dict[tuple[str, str, str], str] = {}
    for key, counts in variants.items():
        canonical[key] = sorted(counts.items(), key=lambda item: (-item[1], len(item[0]), item[0]))[0][0]
    return canonical


def build_key_counts(rows: list[dict[str, Any]], city_patterns: list[str]) -> Counter[tuple[str, str, str]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        key = address_key(clean(row.get("адрес")), normalize_city(row.get("город")), city_patterns)
        if valid_key(key):
            counts[key] += 1
    return counts


def fuzzy_redirects(
    canonical: dict[tuple[str, str, str], str],
    key_counts: Counter[tuple[str, str, str]],
    threshold: float = 0.875,
) -> dict[tuple[str, str, str], tuple[str, str, str]]:
    keys = list(canonical)
    parent = {key: key for key in keys}

    def find(key: tuple[str, str, str]) -> tuple[str, str, str]:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: tuple[str, str, str], right: tuple[str, str, str]) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        winner = sorted(
            [root_left, root_right],
            key=lambda key: (-key_counts[key], len(canonical[key]), canonical[key]),
        )[0]
        loser = root_right if winner == root_left else root_left
        parent[loser] = winner

    by_city_house: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for key in keys:
        city, street, house = key
        if city and city != "Неизвестно" and street and house:
            by_city_house[(city, house)].append(key)

    for candidates in by_city_house.values():
        for index, left in enumerate(candidates):
            left_street = left[1]
            for right in candidates[index + 1 :]:
                right_street = right[1]
                ratio = SequenceMatcher(None, left_street, right_street).ratio()
                if ratio >= threshold:
                    union(left, right)

    return {key: find(key) for key in keys if find(key) != key}


def search_known_address_in_text(
    text: str,
    city: str,
    canonical: dict[tuple[str, str, str], str],
) -> str:
    city = normalize_city(city)
    normalized = norm_text(text)
    normalized = STREET_PREFIX_RE.sub(" ", normalized)
    normalized = re.sub(r"[^а-яa-z0-9/]+", " ", normalized)
    normalized = " ".join(normalized.split())

    candidates = [
        (key, display)
        for key, display in canonical.items()
        if key[0] == city and key[1] and key[2]
    ]
    for key, display in sorted(candidates, key=lambda item: len(item[0][1]), reverse=True):
        _, street, house = key
        street_pattern = re.escape(street).replace(r"\ ", r"\s+")
        house_pattern = re.escape(house.lower())
        if re.search(rf"\b{street_pattern}\s+{house_pattern}\b", normalized):
            return display
        if re.search(rf"\b{house_pattern}\s+{street_pattern}\b", normalized):
            return display
    return ""


def normalize_row_address(
    row: dict[str, Any],
    canonical: dict[tuple[str, str, str], str],
    redirects: dict[tuple[str, str, str], tuple[str, str, str]],
    city_patterns: list[str],
) -> tuple[str, str, str]:
    original = clean(row.get("адрес"))
    city = normalize_city(row.get("город"))
    key = address_key(original, city, city_patterns)

    if valid_key(key) and key in redirects:
        target = redirects[key]
        return canonical[target], target[0], "fuzzy"

    if valid_key(key) and key in canonical:
        normalized = canonical[key]
        method = "канонический ключ" if normalized != original else "без изменений"
        return normalized, city, method

    candidate = extract_address_candidate(clean(row.get("текст_отзыва")))
    candidate_key = address_key(candidate, city, city_patterns)
    if valid_key(candidate_key) and candidate_key in redirects:
        target = redirects[candidate_key]
        return canonical[target], target[0], "fuzzy из текста"
    if valid_key(candidate_key):
        normalized = canonical.get(candidate_key) or display_from_address(candidate, city, city_patterns)
        return normalized, candidate_key[0], "из текста"

    known_from_text = search_known_address_in_text(clean(row.get("текст_отзыва")), city, canonical)
    if known_from_text:
        return known_from_text, city, "из текста: известная точка"

    if original and original != "Неизвестно" and valid_key(key):
        return display_from_address(original, city, city_patterns), city, "форматирование"

    return "Неизвестно", city, "не найден"


def add_address_audit_sheet(workbook, audit_rows: list[dict[str, Any]]) -> None:
    if "Нормализация адресов" in workbook.sheetnames:
        del workbook["Нормализация адресов"]
    sheet = workbook.create_sheet("Нормализация адресов")
    headers = [
        "адрес_исходный",
        "адрес_нормализованный",
        "город_исходный",
        "город_нормализованный",
        "метод",
        "количество_отзывов",
    ]
    sheet.append(headers)
    grouped = Counter(
        (
            row["адрес_исходный"],
            row["адрес_нормализованный"],
            row["город_исходный"],
            row["город_нормализованный"],
            row["метод"],
        )
        for row in audit_rows
    )
    for values, count in sorted(grouped.items(), key=lambda item: (-item[1], item[0])):
        sheet.append([*values, count])
    base.style_header(sheet)
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    widths = [36, 36, 20, 20, 20, 18]
    for index, width in enumerate(widths, 1):
        sheet.column_dimensions[base.get_column_letter(index)].width = width


def main() -> None:
    if "Сервис" not in base.DETAIL_CATEGORIES:
        insert_at = base.DETAIL_CATEGORIES.index("Другое")
        base.DETAIL_CATEGORIES.insert(insert_at, "Сервис")

    workbook = base.openpyxl.load_workbook(INPUT_FILE)
    rows = base.read_rows(workbook)
    city_patterns = base.known_city_patterns(rows)
    canonical = build_canonical_map(rows, city_patterns)
    key_counts = build_key_counts(rows, city_patterns)
    redirects = fuzzy_redirects(canonical, key_counts)

    audit_rows = []
    for row in rows:
        original_address = clean(row.get("адрес"))
        original_city = clean(row.get("город"))
        normalized_address, normalized_city, method = normalize_row_address(row, canonical, redirects, city_patterns)
        row["адрес"] = normalized_address
        row["город"] = normalized_city
        if method in {"из текста", "fuzzy из текста"}:
            row["адрес_источник"] = "по тексту"
        elif method in {"канонический ключ", "форматирование", "fuzzy"}:
            row["адрес_источник"] = "нормализация"
        elif method == "не найден":
            row["адрес_источник"] = "не найден"
        audit_rows.append(
            {
                "адрес_исходный": original_address,
                "адрес_нормализованный": normalized_address,
                "город_исходный": original_city,
                "город_нормализованный": normalized_city,
                "метод": method,
            }
        )

    base.clear_sheets(workbook)
    base.rewrite_registry(workbook, rows)
    base.write_summary(workbook, rows)
    categories = base.write_negative_categories(workbook, rows)
    base.write_pivot(workbook, rows, "адрес", "По пекарням", categories)
    base.write_pivot(workbook, rows, "город", "По городам", categories)
    base.write_all_reviews(workbook, rows)
    base.write_city_fill(workbook, rows)
    add_address_audit_sheet(workbook, audit_rows)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(OUTPUT_FILE)

    method_counts = Counter(row["метод"] for row in audit_rows)
    print(f"rows={len(rows)}")
    print(f"unique_addresses_before={len(set(row['адрес_исходный'] for row in audit_rows))}")
    print(f"unique_addresses_after={len(set(row['адрес_нормализованный'] for row in audit_rows))}")
    print(f"unknown_after={sum(1 for row in rows if row.get('адрес') == 'Неизвестно')}")
    print(f"methods={dict(method_counts)}")
    print(f"fuzzy_redirects={len(redirects)}")
    print(f"output={OUTPUT_FILE}")


if __name__ == "__main__":
    main()
