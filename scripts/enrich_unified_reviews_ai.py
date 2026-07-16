"""Enrich the unified buyer reviews registry with AI-derived analytical fields."""

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
BATCH_SIZE = 8

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path(
    r"C:\Users\dns\Documents\Codex\2026-06-30\new-chat\outputs\Отзывы покупателей единый реестр.xlsx"
)
DEFAULT_CHECKPOINT = PROJECT_DIR / "outputs" / "unified_reviews_ai_enrichment_checkpoint.json"
DEFAULT_RESULTS = PROJECT_DIR / "outputs" / "unified_reviews_ai_enrichment_results.json"
LEGACY_SCRIPT = PROJECT_DIR / "analyze_reviews.py"

FIELDS = [
    "подкатегория_нормализованная",
    "подкатегория_описание",
    "критичность",
    "риск_здоровью",
    "продукт",
    "краткая_суть",
    "требует_реакции",
]

SUBCATEGORY_OPTIONS = [
    "Свежесть продукта",
    "Плесень",
    "Прокисший продукт",
    "Инородный предмет",
    "Плохой вкус",
    "Сухая выпечка",
    "Мало начинки",
    "Размер или вес продукта",
    "Цена и ценность",
    "Ассортимент",
    "Упаковка",
    "Грубость персонала",
    "Невнимательность персонала",
    "Долгое обслуживание",
    "Ошибки в заказе",
    "Отказ решить проблему",
    "Чистота помещения",
    "Санитарный риск",
    "Шум или запах",
    "Условия помещения",
    "Похвала продукции",
    "Похвала персонала",
    "Похвала сервиса",
    "Похвала атмосферы",
    "Похвала ассортимента",
    "Похвала заказа на праздник",
    "Отзыв сотрудника",
    "Другое",
]


def load_api_key() -> str:
    env_key = os.getenv("VIBECODE_API_KEY")
    if env_key:
        return env_key

    if LEGACY_SCRIPT.exists():
        text = LEGACY_SCRIPT.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r'API_KEY\s*=\s*["\']([^"\']+)["\']', text)
        if match:
            return match.group(1)

    raise RuntimeError("VIBECODE_API_KEY is not set and API_KEY was not found")


def ai_complete(prompt: str, api_key: str, max_tokens: int = 2500) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }

    last_error: Exception | str | None = None
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
                retry = int(response.headers.get("Retry-After", 8))
                print(f"  rate limit, waiting {retry}s")
                time.sleep(retry)
                attempt += 1
                continue
            if 500 <= response.status_code < 600:
                last_error = f"{response.status_code}: {response.text[:300]}"
                wait = min(300, 10 + attempt * 10)
                print(f"  server error {response.status_code}, waiting {wait}s")
                time.sleep(wait)
                attempt += 1
                continue

            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except requests.RequestException as exc:
            last_error = exc
            wait = min(300, 10 + attempt * 10)
            print(f"  request error, waiting {wait}s: {exc}")
            time.sleep(wait)
            attempt += 1


def strip_json_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def normalize_item(value: dict[str, Any]) -> dict[str, str]:
    result = {field: str(value.get(field, "") or "").strip() for field in FIELDS}

    if result["подкатегория_нормализованная"] not in SUBCATEGORY_OPTIONS:
        result["подкатегория_нормализованная"] = "Другое"
    if result["критичность"] not in {"низкая", "средняя", "высокая"}:
        result["критичность"] = "средняя"
    if result["риск_здоровью"] not in {"да", "нет"}:
        result["риск_здоровью"] = "нет"
    if result["требует_реакции"] not in {"да", "нет"}:
        result["требует_реакции"] = "да" if result["критичность"] in {"средняя", "высокая"} else "нет"

    for field in FIELDS:
        if not result[field]:
            result[field] = "не указано"
    return result


def parse_response(raw: str, expected: int) -> list[dict[str, str]]:
    try:
        data = json.loads(strip_json_fence(raw))
        if not isinstance(data, list):
            raise ValueError("JSON response is not a list")
        items = [normalize_item(item if isinstance(item, dict) else {}) for item in data]
    except Exception as exc:
        print(f"  could not parse JSON response: {exc}")
        items = []

    while len(items) < expected:
        items.append(
            {
                "подкатегория_нормализованная": "Другое",
                "подкатегория_описание": "не удалось разобрать ответ модели",
                "критичность": "средняя",
                "риск_здоровью": "нет",
                "продукт": "не указано",
                "краткая_суть": "не удалось разобрать ответ модели",
                "требует_реакции": "да",
            }
        )
    return items[:expected]


def classify_batch(records: list[dict[str, Any]], api_key: str) -> list[dict[str, str]]:
    lines = []
    for idx, record in enumerate(records, start=1):
        text = str(record["текст_отзыва"]).replace("\n", " ")[:700]
        lines.append(
            f"{idx}. тип={record['тип_отзыва']}; категория={record['категория_нормализованная']}; "
            f"текст={text}"
        )

    options = "\n".join(f"- {item}" for item in SUBCATEGORY_OPTIONS)
    prompt = f"""Ты аналитик отзывов покупателей о сети пекарен.
Для каждого отзыва верни JSON-массив объектов строго в том же порядке.

Поля каждого объекта:
- "подкатегория_нормализованная": выбери РОВНО ОДНО значение из справочника ниже.
- "подкатегория_описание": конкретная причина/тема в 2-5 словах. Для благодарностей: за что хвалят.
- "критичность": одно из "низкая", "средняя", "высокая".
- "риск_здоровью": "да" только если есть плесень, отравление, просрочка, инородный предмет, прокисший/несвежий продукт или явный санитарный риск; иначе "нет".
- "продукт": конкретный продукт, если указан; иначе "не указано".
- "краткая_суть": короткое резюме до 18 слов.
- "требует_реакции": "да" если это жалоба, санитарный риск, конфликт, потеря клиента или нужно проверить точку; иначе "нет".

Справочник подкатегорий:
{options}

Не добавляй пояснений. Только валидный JSON-массив.

Отзывы:
{chr(10).join(lines)}
"""
    raw = ai_complete(prompt, api_key, max_tokens=max(1200, len(records) * 120))
    return parse_response(raw, len(records))


def load_records(input_path: Path) -> list[dict[str, Any]]:
    workbook = openpyxl.load_workbook(input_path, read_only=True, data_only=True)
    sheet = workbook.worksheets[0]
    headers = [str(cell.value).strip() for cell in sheet[1]]

    records: list[dict[str, Any]] = []
    for row in sheet.iter_rows(min_row=2, values_only=True):
        data = dict(zip(headers, row))
        text = str(data.get("текст_отзыва") or "").strip()
        if not text:
            continue
        source_sheet = str(data.get("исходный_лист") or "").strip()
        source_row = str(data.get("исходная_строка") or "").strip()
        records.append(
            {
                "id": f"{source_sheet}|{source_row}",
                "тип_отзыва": str(data.get("тип_отзыва") or "").strip(),
                "категория_нормализованная": str(data.get("категория_нормализованная") or "").strip(),
                "текст_отзыва": text,
            }
        )
    return records


def load_checkpoint(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        key: value
        for key, value in data.items()
        if isinstance(value, dict) and "подкатегория_нормализованная" in value
    }


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_chunk_results(results_path: Path, checkpoint: dict[str, dict[str, str]], chunk_number: int) -> None:
    chunk_dir = results_path.parent / "ai_enrichment_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = chunk_dir / f"chunk_{chunk_number:05d}.json"
    save_json(chunk_path, checkpoint)


def run(input_path: Path, checkpoint_path: Path, results_path: Path, limit: int | None) -> None:
    api_key = load_api_key()
    records = load_records(input_path)
    if limit is not None:
        records = records[:limit]

    checkpoint = load_checkpoint(checkpoint_path)
    todo = [record for record in records if record["id"] not in checkpoint]

    print(f"Reviews found: {len(records)}")
    print(f"Already enriched: {len(checkpoint)}")
    print(f"Remaining: {len(todo)}")

    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo[start : start + BATCH_SIZE]
        enriched = classify_batch(batch, api_key)
        for record, item in zip(batch, enriched):
            checkpoint[record["id"]] = item
        save_json(checkpoint_path, checkpoint)
        save_chunk_results(results_path, checkpoint, len(checkpoint) // BATCH_SIZE)
        print(f"  enriched {min(start + len(batch), len(todo))}/{len(todo)} new rows")
        time.sleep(0.2)

    results = []
    for record in records:
        item = checkpoint.get(record["id"])
        if not item:
            continue
        row = {"id": record["id"], **item}
        results.append(row)

    save_json(results_path, results)
    print(f"Saved results: {results_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich unified reviews with AI fields")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for testing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.input, args.checkpoint, args.results, args.limit)


if __name__ == "__main__":
    main()
