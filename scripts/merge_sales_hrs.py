"""
Собирает все xlsx-файлы из sales_hrs/ (включая подпапки aug, sept, oct, nove, desem)
в один CSV-файл data/raw/sales_hrs_all.csv
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent / "sales_hrs"
OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "sales_hrs_all.csv"

dfs = []
errors = []

# Собираем все xlsx рекурсивно
files = sorted(BASE.rglob("*.xlsx"))
print(f"Найдено {len(files)} xlsx-файлов", flush=True)

for i, f in enumerate(files, 1):
    try:
        df = pd.read_excel(f)
        dfs.append(df)
        print(f"  [{i}/{len(files)}] {f.relative_to(BASE)}: {len(df):,} строк", flush=True)
    except Exception as e:
        errors.append((f, e))
        print(f"  [{i}/{len(files)}] ОШИБКА {f.relative_to(BASE)}: {e}", flush=True)

if errors:
    print(f"\n{len(errors)} файлов с ошибками!")

result = pd.concat(dfs, ignore_index=True)
print(f"\nИтого: {len(result):,} строк, {result['Дата продажи'].nunique()} уникальных дат")
print(f"Период: {result['Дата продажи'].min()} - {result['Дата продажи'].max()}")
print(f"Колонки: {list(result.columns)}")

# Убираем полные дубликаты (если диапазоны файлов пересекались)
before = len(result)
result = result.drop_duplicates()
after = len(result)
if before != after:
    print(f"Удалено {before - after:,} дубликатов")

result.to_csv(OUT, index=False, encoding='utf-8-sig')
print(f"\nСохранено: {OUT}")
print(f"Размер: {OUT.stat().st_size / 1024 / 1024:.1f} MB")
