"""
Простой трекинг экспериментов в JSONL-файл.
Каждый запуск обучения -> одна JSON-строка.
"""

import json
import os
from datetime import datetime

from src.config import REPORTS_DIR

EXPERIMENT_LOG = os.path.join(REPORTS_DIR, "experiment_log.jsonl")


def log_experiment(
    name: str,
    metrics: dict,
    params: dict = None,
    model_path: str = None,
    data_path: str = None,
    notes: str = None,
):
    """
    Записывает результат эксперимента в JSONL-файл.

    Args:
        name: название эксперимента (e.g. "v6_train", "exp_a_no_weather")
        metrics: dict с метриками (mae, wmape, r2, ...)
        params: параметры модели
        model_path: путь к сохраненной модели
        data_path: путь к данным
        notes: произвольные заметки
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "name": name,
        "metrics": metrics,
    }
    if params:
        entry["params"] = params
    if model_path:
        entry["model_path"] = model_path
    if data_path:
        entry["data_path"] = data_path
    if notes:
        entry["notes"] = notes

    os.makedirs(os.path.dirname(EXPERIMENT_LOG), exist_ok=True)

    with open(EXPERIMENT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return entry
