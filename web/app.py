"""
Flask веб-приложение для прогнозирования спроса пекарни.
Фичи автоматически берутся из исторических данных (CSV).
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MODEL_PATH, META_PATH, PROCESSED_DATA_PATH, LOGS_DIR
from src.logger import get_logger

DATA_PATH = PROCESSED_DATA_PATH

app = Flask(__name__)
logger = get_logger("web", log_file="web.log")

# --- Prediction logging ---
PREDICTION_LOG = os.path.join(LOGS_DIR, "predictions.csv")


def _log_prediction(bakery, product, date, prediction, fact, response_time_ms):
    """Append prediction to CSV log."""
    import csv
    from datetime import datetime
    os.makedirs(LOGS_DIR, exist_ok=True)
    write_header = not os.path.exists(PREDICTION_LOG)
    with open(PREDICTION_LOG, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "bakery", "product", "date",
                             "prediction", "fact", "response_time_ms"])
        writer.writerow([
            datetime.now().isoformat(),
            bakery, product, str(date),
            round(prediction, 1), fact, round(response_time_ms, 1),
        ])


# --- Загрузка при старте ---
logger.info("Загрузка модели и данных...")
model = joblib.load(MODEL_PATH)
meta = joblib.load(META_PATH)

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df["Дата"] = pd.to_datetime(df["Дата"])
for col in meta["categorical_cols"]:
    if col in df.columns:
        df[col] = df[col].astype("category")

# Уникальные значения для фильтров
bakeries = sorted(df["Пекарня"].unique().tolist())
products = sorted(df["Номенклатура"].unique().tolist())
dates = sorted(df["Дата"].dt.strftime("%Y-%m-%d").unique().tolist())

# Маппинги
product_category = df.groupby(observed=True, by="Номенклатура")["Категория"].first().to_dict()
bakery_city = df.groupby(observed=True, by="Пекарня")["Город"].first().to_dict()

# Какие продукты есть в каждой пекарне
bakery_products = df.groupby(observed=True, by="Пекарня")["Номенклатура"].apply(lambda x: sorted(x.unique().tolist())).to_dict()

logger.info(f"Готово. Строк: {len(df):,}, пекарен: {len(bakeries)}, продуктов: {len(products)}, дат: {len(dates)}")


@app.route("/")
def index():
    return render_template(
        "index.html",
        bakeries=bakeries,
        dates=dates,
        metrics=meta["metrics"],
        bakery_city=bakery_city,
    )


@app.route("/api/products")
def get_products():
    """Продукты доступные в выбранной пекарне."""
    bakery = request.args.get("bakery", "")
    prods = bakery_products.get(bakery, products)
    return jsonify(prods)


@app.route("/api/dates")
def get_dates():
    """Даты, для которых есть данные по паре пекарня+продукт."""
    bakery = request.args.get("bakery", "")
    product = request.args.get("product", "")
    mask = (df["Пекарня"] == bakery) & (df["Номенклатура"] == product)
    available = sorted(df.loc[mask, "Дата"].dt.strftime("%Y-%m-%d").unique().tolist())
    return jsonify(available)


@app.route("/predict", methods=["POST"])
def predict():
    import time
    t0 = time.perf_counter()
    try:
        data = request.get_json()
        bakery = data["bakery"]
        product = data["product"]
        date = pd.to_datetime(data["date"])

        # Ищем строку в данных
        mask = (df["Пекарня"] == bakery) & (df["Номенклатура"] == product) & (df["Дата"] == date)
        rows = df[mask]

        if rows.empty:
            return jsonify({
                "success": False,
                "error": "Нет данных для этой комбинации пекарня/продукт/дата"
            })

        row = rows.iloc[[0]]

        # Берём фичи из данных (iloc[[0]] сохраняет DataFrame и типы)
        X = row[meta["features"]].copy()
        for col in meta["categorical_cols"]:
            if col in X.columns:
                X[col] = X[col].astype("category")

        prediction = max(float(model.predict(X)[0]), 0)
        row = row.iloc[0]

        # Факт (если есть)
        fact = int(row["Продано"]) if pd.notna(row["Продано"]) else None

        # Ключевые фичи для отображения
        details = {
            "Категория": str(row.get("Категория", "")),
            "Город": str(row.get("Город", "")),
            "Продажи вчера": row.get("sales_lag1"),
            "Продажи 7 дн. назад": row.get("sales_lag7"),
            "Среднее за 7 дн.": round(row.get("sales_roll_mean7", 0), 1),
            "Остаток вчера": row.get("stock_lag1"),
            "Темп. средняя": round(row.get("temp_mean", 0), 1),
            "Осадки": row.get("precipitation"),
            "Выходной": "Да" if row.get("IsWeekend") == 1 else "Нет",
            "Праздник": "Да" if row.get("is_holiday") == 1 else "Нет",
        }

        # --- История за неделю до выбранной даты (включительно) ---
        week_start = date - pd.Timedelta(days=6)
        week_mask = (
            (df["Пекарня"] == bakery)
            & (df["Номенклатура"] == product)
            & (df["Дата"] >= week_start)
            & (df["Дата"] <= date)
        )
        week_rows = df[week_mask].sort_values("Дата")

        chart_data = []
        if not week_rows.empty:
            X_week = week_rows[meta["features"]].copy()
            for col in meta["categorical_cols"]:
                if col in X_week.columns:
                    X_week[col] = X_week[col].astype("category")
            week_preds = np.maximum(model.predict(X_week), 0)

            for i, (_, wr) in enumerate(week_rows.iterrows()):
                chart_data.append({
                    "date": wr["Дата"].strftime("%Y-%m-%d"),
                    "fact": int(wr["Продано"]) if pd.notna(wr["Продано"]) else None,
                    "pred": round(float(week_preds[i]), 1),
                })

        elapsed_ms = (time.perf_counter() - t0) * 1000
        _log_prediction(bakery, product, date, prediction, fact, elapsed_ms)

        return jsonify({
            "success": True,
            "prediction": round(prediction, 1),
            "prediction_int": int(round(prediction)),
            "fact": fact,
            "details": details,
            "chart": chart_data,
        })

    except Exception as e:
        logger.error(f"predict error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/predict_all", methods=["POST"])
def predict_all():
    """Прогноз по всем продуктам пекарни на выбранную дату."""
    import time
    t0 = time.perf_counter()
    try:
        data = request.get_json()
        bakery = data["bakery"]
        date = pd.to_datetime(data["date"])

        mask = (df["Пекарня"] == bakery) & (df["Дата"] == date)
        rows = df[mask]

        if rows.empty:
            return jsonify({"success": False, "error": "Нет данных для этой пекарни на эту дату"})

        # Прогноз по всем строкам разом (быстрее чем по одной)
        X_all = rows[meta["features"]].copy()
        for col in meta["categorical_cols"]:
            if col in X_all.columns:
                X_all[col] = X_all[col].astype("category")

        preds = np.maximum(model.predict(X_all), 0)

        results = []
        for i, (_, row) in enumerate(rows.iterrows()):
            pred = float(preds[i])
            fact = int(row["Продано"]) if pd.notna(row["Продано"]) else None

            results.append({
                "product": str(row["Номенклатура"]),
                "category": str(row["Категория"]),
                "prediction": round(pred, 1),
                "prediction_int": int(round(pred)),
                "fact": fact,
                "error": abs(fact - round(pred)) if fact is not None else None,
            })

        results.sort(key=lambda x: x["prediction"], reverse=True)

        total_pred = sum(r["prediction"] for r in results)
        total_fact = sum(r["fact"] for r in results if r["fact"] is not None)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"predict_all: {bakery} {date.date()} -> {len(results)} items, {elapsed_ms:.0f}ms")

        return jsonify({
            "success": True,
            "results": results,
            "total_pred": round(total_pred, 1),
            "total_fact": total_fact,
            "count": len(results),
        })

    except Exception as e:
        logger.error(f"predict_all error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=False, port=5000)
