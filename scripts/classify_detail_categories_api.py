"""Classify complaint detail categories through the VibeCode API."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import openpyxl
import requests


BASE_URL = "https://vibecode.bitrix24.tech/v1"
MODEL = "bitrix/bitrixgpt-5.5"
BATCH_SIZE = 20

PROJECT_DIR = Path(__file__).resolve().parents[1]
LEGACY_SCRIPT = PROJECT_DIR / "analyze_reviews.py"
INPUT_FILE = Path(
    r"C:\Users\dns\Documents\Codex\2026-06-30\new-chat\outputs\Отзывы покупателей единый реестр.xlsx"
)
CHECKPOINT_FILE = PROJECT_DIR / "outputs" / "detail_categories_api_checkpoint.json"
RESULTS_FILE = PROJECT_DIR / "outputs" / "detail_categories_api_results.json"

DETAIL_CATEGORIES = [
    "Низкое качество еды",
    "Невежливое обслуживание",
    "Несвежие продукты",
    "Проблемы с чистотой",
    "Недостаток ассортимента",
    "Ошибки при расчетах",
    "Наличие вредителей",
    "Плохая упаковка товаров",
    "Другое",
]


def load_api_key() -> str:
    env_key = os.getenv("VIBECODE_API_KEY")
    if env_key:
        return env_key
    text = LEGACY_SCRIPT.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r'API_KEY\s*=\s*["\']([^"\']+)["\']', text)
    if not match:
        raise RuntimeError("API key was not found")
    return match.group(1)


def ai_complete(prompt: str, api_key: str, max_tokens: int = 2500) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    attempt = 0
    while True:
        try:
            response = requests.post(
                f"{BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            if response.status_code == 429:
                wait = int(response.headers.get("Retry-After", 20))
                print(f"  rate limit, waiting {wait}s")
                time.sleep(wait)
                continue
            if response.status_code >= 500:
                wait = min(300, 15 + attempt * 15)
                print(f"  server error {response.status_code}, waiting {wait}s")
                time.sleep(wait)
                attempt += 1
                continue
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except requests.RequestException as exc:
            wait = min(300, 15 + attempt * 15)
            print(f"  request error, waiting {wait}s: {exc}")
            time.sleep(wait)
            attempt += 1


def strip_json(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def normalize_category(value: str) -> str:
    value = str(value or "").strip()
    return value if value in DETAIL_CATEGORIES else "Другое"


def headers_map(sheet: openpyxl.worksheet.worksheet.Worksheet) -> dict[str, int]:
    return {str(c.value).strip(): i for i, c in enumerate(sheet[1], start=1) if c.value}


def collect_complaints(input_file: Path) -> list[dict[str, Any]]:
    wb = openpyxl.load_workbook(input_file, read_only=True, data_only=True)
    ws = wb["Единый реестр отзывов"]
    h = headers_map(ws)
    rows = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        review_type = str(row[h["тип_отзыва"] - 1] or "").strip()
        if review_type != "Жалоба":
            continue
        source_sheet = str(row[h["исходный_лист"] - 1] or "").strip()
        source_row = str(row[h["исходная_строка"] - 1] or "").strip()
        rows.append(
            {
                "id": f"{source_sheet}|{source_row}",
                "category": str(row[h["категория_нормализованная"] - 1] or "").strip(),
                "text": str(row[h["текст_отзыва"] - 1] or "").strip(),
                "city": str(row[h["город"] - 1] or "").strip(),
                "address": str(row[h["адрес"] - 1] or "").strip(),
            }
        )
    return rows


def classify_batch(records: list[dict[str, Any]], api_key: str) -> list[str]:
    categories = "\n".join(f"- {category}" for category in DETAIL_CATEGORIES)
    lines = []
    for index, record in enumerate(records, start=1):
        text = record["text"].replace("\n", " ")[:900]
        lines.append(
            f"{index}. базовая_категория={record['category']}; город={record['city']}; "
            f"адрес={record['address']}; текст={text}"
        )

    prompt = f"""Ты классификатор жалоб покупателей сети пекарен.
Для каждой жалобы выбери ОДНУ наиболее подходящую детальную категорию из справочника.

Справочник:
{categories}

Критерии:
- "Невежливое обслуживание": грубость, хамство, игнорирование, конфликт с кассиром/продавцом, долгое обслуживание.
- "Низкое качество еды": невкусно, сухо, мало начинки, сырое, плохая рецептура, ухудшилось качество.
- "Несвежие продукты": просрочка, плесень, кислый/прокисший вкус, запах, подозрение на несвежесть.
- "Проблемы с чистотой": грязь, антисанитария, грязные столы/полы, санитарные замечания без явной просрочки.
- "Недостаток ассортимента": нет товара, закончилась выпечка, мало выбора.
- "Ошибки при расчетах": цена, чек, сдача, обсчитали, ценник не совпал.
- "Наличие вредителей": тараканы, насекомые, мыши, крысы.
- "Плохая упаковка товаров": пакет/упаковка испортили товар, товар развалился из-за упаковки.
- "Другое": если не подходит ни одна категория.

Верни только JSON-массив строк в том же порядке. Без пояснений.

Жалобы:
{chr(10).join(lines)}
"""
    raw = ai_complete(prompt, api_key, max_tokens=max(1200, len(records) * 40))
    try:
        parsed = json.loads(strip_json(raw))
        if not isinstance(parsed, list):
            raise ValueError("not a list")
    except Exception as exc:
        print(f"  parse failed: {exc}")
        parsed = []

    result = []
    for index in range(len(records)):
        value = parsed[index] if index < len(parsed) else "Другое"
        result.append(normalize_category(value))
    return result


def load_checkpoint(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run(limit: int | None) -> None:
    api_key = load_api_key()
    records = collect_complaints(INPUT_FILE)
    if limit is not None:
        records = records[:limit]
    checkpoint = load_checkpoint(CHECKPOINT_FILE)
    todo = [record for record in records if record["id"] not in checkpoint]
    print(f"complaints={len(records)} already={len(checkpoint)} remaining={len(todo)}")
    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo[start : start + BATCH_SIZE]
        categories = classify_batch(batch, api_key)
        for record, category in zip(batch, categories):
            checkpoint[record["id"]] = category
        save_json(CHECKPOINT_FILE, checkpoint)
        print(f"  classified {min(start + len(batch), len(todo))}/{len(todo)}")
        time.sleep(0.2)
    save_json(RESULTS_FILE, [{"id": key, "детальная_категория_api": value} for key, value in checkpoint.items()])
    print(f"saved={RESULTS_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args.limit)


if __name__ == "__main__":
    main()
