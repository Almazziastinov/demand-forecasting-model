"""
Модуль логирования. Совместим с Windows cp1251 консолью.
"""

import logging
import sys
import os

from src.config import LOGS_DIR


def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Создает логгер с форматированным выводом.
    - Консоль: всегда
    - Файл: если указан log_file (относительно LOGS_DIR)
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler -- utf-8 для корректного вывода кириллицы
    console = logging.StreamHandler(
        stream=open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
    )
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler
    if log_file:
        os.makedirs(LOGS_DIR, exist_ok=True)
        fh = logging.FileHandler(
            os.path.join(LOGS_DIR, log_file), encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
