"""
Полный пайплайн: preprocessing -> weather -> train -> evaluate.
Запуск: python run_pipeline.py [--skip-preprocess] [--skip-weather] [--merge]
"""

import argparse
import subprocess
import sys
import os

PYTHON = sys.executable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_step(name, cmd):
    print(f"\n{'=' * 60}")
    print(f"  [{name}]")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd, cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"\n  ОШИБКА на шаге [{name}], код: {result.returncode}")
        sys.exit(result.returncode)
    print(f"  [{name}] OK")


def main():
    parser = argparse.ArgumentParser(description="Demand forecasting pipeline")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Пропустить препроцессинг (использовать готовый CSV)")
    parser.add_argument("--skip-weather", action="store_true",
                        help="Пропустить загрузку погоды (использовать готовый enriched CSV)")
    parser.add_argument("--merge", action="store_true",
                        help="Объединить 3_month_data + beigl_data перед препроцессингом")
    args = parser.parse_args()

    print("=" * 60)
    print("  DEMAND FORECASTING PIPELINE")
    print("=" * 60)

    # Step 1: Preprocessing
    if not args.skip_preprocess:
        preprocess_cmd = [PYTHON, "src/preprocessing.py"]
        if args.merge:
            preprocess_cmd.append("--merge")
        run_step("PREPROCESSING", preprocess_cmd)

        # Determine which CSV was produced
        if args.merge:
            csv_path = "data/processed/preprocessed_data_merged.csv"
        else:
            csv_path = "data/processed/preprocessed_data.csv"
    else:
        print("\n  [PREPROCESSING] skipped")
        if args.merge:
            csv_path = "data/processed/preprocessed_data_merged.csv"
        else:
            csv_path = "data/processed/preprocessed_data.csv"

    # Step 2: Weather enrichment
    if not args.skip_weather:
        run_step("WEATHER", [PYTHON, "src/features/fetch_weather.py", csv_path])
    else:
        print("\n  [WEATHER] skipped")

    # Step 3: Train model
    run_step("TRAIN", [PYTHON, "src/models/train_and_save.py"])

    print(f"\n{'=' * 60}")
    print("  PIPELINE COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
