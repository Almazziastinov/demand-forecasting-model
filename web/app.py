"""
Flask app for bakery demand forecasting.
Uses V3 model (58 features, quantile P50) and raw XLSX data from web/data/.
"""

import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logger import get_logger
from src.features.fetch_weather import fetch_weather_forecast
from web.data_processing import (
    load_or_process, get_hourly_profile,
    FEATURES_V3, CATEGORICAL_COLS_V2,
)

# --- Paths ---
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "demand_model_v3.pkl")
META_PATH = os.path.join(MODELS_DIR, "model_meta_v3.pkl")
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

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


# --- Load model and data at startup ---
logger.info("Loading V3 model...")
model = joblib.load(MODEL_PATH)
meta = joblib.load(META_PATH)
features = meta["features"]  # 58 features
categorical_cols = meta["categorical_cols"]

logger.info("Processing XLSX data from web/data/...")
df, hourly_df, indicators_df = load_or_process()

# Cast categoricals
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype("category")

# Unique values for filters
bakeries = sorted(df["Пекарня"].unique().tolist())
products = sorted(df["Номенклатура"].unique().tolist())
dates = sorted(df["Дата"].dt.strftime("%Y-%m-%d").unique().tolist())

# Mappings
product_category = df.groupby("Номенклатура", observed=True)["Категория"].first().to_dict()
bakery_city = df.groupby("Пекарня", observed=True)["Город"].first().to_dict()
bakery_products = (df.groupby("Пекарня", observed=True)["Номенклатура"]
                   .apply(lambda x: sorted(x.unique().tolist())).to_dict())

# Metrics from meta
metrics = meta.get("metrics", {})

logger.info(f"Ready. Rows: {len(df):,}, bakeries: {len(bakeries)}, "
            f"products: {len(products)}, dates: {len(dates)}")


# --- Helper: get indicators for a bakery/product/date ---
def _get_indicators(bakery, product, date, is_historical=True):
    """Get indicators from indicators_df."""
    result = {
        "остаток": None,
        "упущено": None,
        "продажи": None,
    }

    if is_historical and not indicators_df.empty:
        mask = ((indicators_df["Пекарня"] == bakery) &
                (indicators_df["Номенклатура"] == product) &
                (indicators_df["Дата"] == date))
        rows = indicators_df[mask]
        if not rows.empty:
            row = rows.iloc[0]
            # Column names: остаток_на_конец_дня, упущено
            result["остаток"] = int(row.get("остаток_на_конец_дня", 0)) if pd.notna(row.get("остаток_на_конец_дня")) else None
            result["упущено"] = int(row.get("упущено", 0)) if pd.notna(row.get("упущено")) else None

    # For historical dates, also get sales
    if is_historical:
        sales_mask = (df["Пекарня"] == bakery) & (df["Номенклатура"] == product) & (df["Дата"] == date)
        sales_rows = df[sales_mask]
        if not sales_rows.empty:
            result["продажи"] = int(sales_rows.iloc[0]["Продано"])

    return result


# --- Routes ---

@app.route("/")
def index():
    return render_template(
        "index.html",
        bakeries=bakeries,
        dates=dates,
        metrics=metrics,
        bakery_city=bakery_city,
    )


@app.route("/api/products")
def get_products():
    """Products available in a bakery."""
    bakery = request.args.get("bakery", "")
    prods = bakery_products.get(bakery, products)
    return jsonify(prods)


@app.route("/api/dates")
def get_dates():
    """Dates with data for a bakery+product pair."""
    bakery = request.args.get("bakery", "")
    product = request.args.get("product", "")
    mask = (df["Пекарня"] == bakery) & (df["Номенклатура"] == product)
    available = sorted(df.loc[mask, "Дата"].dt.strftime("%Y-%m-%d").unique().tolist())
    return jsonify(available)


@app.route("/predict", methods=["POST"])
def predict():
    t0 = time.perf_counter()
    try:
        data = request.get_json()
        bakery = data["bakery"]
        product = data["product"]
        date = pd.to_datetime(data["date"])

        # Check if date is in our data or is a future date
        is_future = date > df["Дата"].max()
        
        if is_future:
            # Build features for future date
            feature_row = _build_features_for_future(bakery, product, date)
            if feature_row is None:
                return jsonify({
                    "success": False,
                    "error": "Нет исторических данных для этого продукта/пекарни"
                })
            
            X = pd.DataFrame([feature_row])[features]
            for col in categorical_cols:
                if col in X.columns:
                    X[col] = X[col].astype("category")
            
            prediction = max(float(model.predict(X)[0]), 0)
            
            # Get details from last known row
            last_mask = (df["Пекарня"] == bakery) & (df["Номенклатура"] == product)
            last_rows = df[last_mask].sort_values("Дата", ascending=False)
            last_row = last_rows.iloc[0] if not last_rows.empty else None
            
            details = {
                "Категория": str(last_row.get("Категория", "")) if last_row is not None else "",
                "Город": str(last_row.get("Город", "")) if last_row is not None else "",
                "Продажи вчера": _safe_num(last_row.get("sales_lag1")) if last_row is not None else None,
                "Продажи 7 дн. назад": _safe_num(last_row.get("sales_lag7")) if last_row is not None else None,
                "Среднее за 7 дн.": _safe_round(last_row.get("sales_roll_mean7"), 1) if last_row is not None else None,
                "Темп. средняя": _safe_round(feature_row.get("temp_mean"), 1) if "temp_mean" in feature_row else None,
                "Осадки": _safe_num(feature_row.get("precipitation")) if "precipitation" in feature_row else None,
                "Выходной": "Да" if feature_row.get("IsWeekend") == 1 else "Нет",
                "Праздник": "Да" if feature_row.get("is_holiday") == 1 else "Нет",
            }
            
            return jsonify({
                "success": True,
                "prediction": round(prediction, 1),
                "prediction_int": int(round(prediction)),
                "fact": None,
                "details": details,
                "is_forecast": True,
            })
        
        # Original logic for known dates
        mask = (df["Пекарня"] == bakery) & (df["Номенклатура"] == product) & (df["Дата"] == date)
        rows = df[mask]

        if rows.empty:
            return jsonify({
                "success": False,
                "error": "Нет данных для этой комбинации пекарня/продукт/дата"
            })

        row = rows.iloc[[0]]

        # Prepare features
        X = row[features].copy()
        for col in categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype("category")

        prediction = max(float(model.predict(X)[0]), 0)
        row = rows.iloc[0]

        # Fact
        fact = int(row["Продано"]) if pd.notna(row.get("Продано")) else None

        # Details
        details = {
            "Категория": str(row.get("Категория", "")),
            "Город": str(row.get("Город", "")),
            "Продажи вчера": _safe_num(row.get("sales_lag1")),
            "Продажи 7 дн. назад": _safe_num(row.get("sales_lag7")),
            "Среднее за 7 дн.": _safe_round(row.get("sales_roll_mean7"), 1),
            "Темп. средняя": _safe_round(row.get("temp_mean"), 1),
            "Осадки": _safe_num(row.get("precipitation")),
            "Выходной": "Да" if row.get("IsWeekend") == 1 else "Нет",
            "Праздник": "Да" if row.get("is_holiday") == 1 else "Нет",
        }

        # Indicators
        ind = _get_indicators(bakery, product, date, is_historical=True)

        # Full history chart for this bakery+product (all available dates up to selected)
        hist_mask = (
            (df["Пекарня"] == bakery)
            & (df["Номенклатура"] == product)
            & (df["Дата"] <= date)
        )
        hist_rows = df[hist_mask].sort_values("Дата")

        chart_data = []
        if not hist_rows.empty:
            X_hist = hist_rows[features].copy()
            for col in categorical_cols:
                if col in X_hist.columns:
                    X_hist[col] = X_hist[col].astype("category")
            hist_preds = np.maximum(model.predict(X_hist), 0)

            for i, (_, wr) in enumerate(hist_rows.iterrows()):
                demand = int(wr["Спрос"]) if "Спрос" in wr.index and pd.notna(wr.get("Спрос")) else None
                chart_data.append({
                    "date": wr["Дата"].strftime("%Y-%m-%d"),
                    "fact": int(wr["Продано"]) if pd.notna(wr.get("Продано")) else None,
                    "demand": demand,
                    "pred": round(float(hist_preds[i]), 1),
                })

        # Last week summary
        last_week_start = date - pd.Timedelta(days=13)
        last_week_end = date - pd.Timedelta(days=7)
        lw_mask = (
            (df["Пекарня"] == bakery)
            & (df["Номенклатура"] == product)
            & (df["Дата"] >= last_week_start)
            & (df["Дата"] <= last_week_end)
        )
        lw_rows = df[lw_mask]
        last_week = None
        if not lw_rows.empty:
            lw_sales = lw_rows["Продано"].sum()
            lw_demand = lw_rows["Спрос"].sum() if "Спрос" in lw_rows.columns else None
            last_week = {
                "sales": int(lw_sales),
                "demand": int(lw_demand) if lw_demand is not None and pd.notna(lw_demand) else None,
                "days": len(lw_rows),
            }

        elapsed_ms = (time.perf_counter() - t0) * 1000
        _log_prediction(bakery, product, date, prediction, fact, elapsed_ms)

        return jsonify({
            "success": True,
            "prediction": round(prediction, 1),
            "prediction_int": int(round(prediction)),
            "fact": fact,
            "details": details,
            "indicators": ind,
            "chart": chart_data,
            "last_week": last_week,
        })

    except Exception as e:
        logger.error(f"predict error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/predict_all", methods=["POST"])
def predict_all():
    """Predict all products for a bakery on a date."""
    t0 = time.perf_counter()
    try:
        data = request.get_json()
        bakery = data["bakery"]
        date = pd.to_datetime(data["date"])

        is_historical = date <= df["Дата"].max()

        if is_historical:
            mask = (df["Пекарня"] == bakery) & (df["Дата"] == date)
            rows = df[mask]

            if rows.empty:
                return jsonify({"success": False, "error": "Нет данных для этой пекарни на эту дату"})

            X_all = rows[features].copy()
            for col in categorical_cols:
                if col in X_all.columns:
                    X_all[col] = X_all[col].astype("category")

            preds = np.maximum(model.predict(X_all), 0)

            results = []
            for i, (_, row) in enumerate(rows.iterrows()):
                pred = float(preds[i])
                fact = int(row["Продано"]) if pd.notna(row.get("Продано")) else None
                product = str(row["Номенклатура"])

                results.append({
                    "product": product,
                    "category": str(row["Категория"]),
                    "prediction": round(pred, 1),
                    "prediction_int": int(round(pred)),
                    "fact": fact,
                    "error": abs(fact - round(pred)) if fact is not None else None,
                })

            results.sort(key=lambda x: x["prediction"], reverse=True)

            total_pred = sum(r["prediction"] for r in results)
            total_fact = sum(r["fact"] for r in results if r["fact"] is not None)

            return jsonify({
                "success": True,
                "results": results,
                "total_pred": round(total_pred, 1),
                "total_fact": total_fact,
                "count": len(results),
            })
        else:
            # Future date - need to predict for all products that have history
            # Get all products for this bakery from historical data
            bakery_products_mask = df["Пекарня"] == bakery
            all_products = df[bakery_products_mask]["Номенклатура"].unique()

            if len(all_products) == 0:
                return jsonify({"success": False, "error": "Нет исторических данных для этой пекарни"})

            results = []
            for product in all_products:
                # Build features for future
                feature_row = _build_features_for_future(bakery, product, date)
                if feature_row is None:
                    continue

                X = pd.DataFrame([feature_row])[features]
                for col in categorical_cols:
                    if col in X.columns:
                        X[col] = X[col].astype("category")

                pred = max(float(model.predict(X)[0]), 0)

                # Get category from last known
                last_mask = (df["Пекарня"] == bakery) & (df["Номенклатура"] == product)
                last_rows = df[last_mask].sort_values("Дата", ascending=False)
                category = str(last_rows.iloc[0].get("Категория", "")) if not last_rows.empty else "Прочее"

                results.append({
                    "product": str(product),
                    "category": category,
                    "prediction": round(pred, 1),
                    "prediction_int": int(round(pred)),
                    "fact": None,
                    "error": None,
                })

            results.sort(key=lambda x: x["prediction"], reverse=True)

            total_pred = sum(r["prediction"] for r in results)

            return jsonify({
                "success": True,
                "results": results,
                "total_pred": round(total_pred, 1),
                "total_fact": None,
                "count": len(results),
                "is_forecast": True,
            })

    except Exception as e:
        logger.error(f"predict_all error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/daily_plan", methods=["POST"])
def daily_plan():
    """Get hourly demand plan for a bakery+product."""
    try:
        data = request.get_json()
        bakery = data["bakery"]
        product = data["product"]
        date = pd.to_datetime(data["date"])

        # Check if historical or future
        is_historical = date <= df["Дата"].max()

        if is_historical:
            # Get model prediction from data
            mask = (df["Пекарня"] == bakery) & (df["Номенклатура"] == product) & (df["Дата"] == date)
            rows = df[mask]

            if rows.empty:
                return jsonify({"success": False, "error": "Нет данных"})

            row = rows.iloc[[0]]
            X = row[features].copy()
            for col in categorical_cols:
                if col in X.columns:
                    X[col] = X[col].astype("category")

            daily_total = max(float(model.predict(X)[0]), 0)
            category = str(rows.iloc[0].get("Категория", ""))
        else:
            # Future date - build features on the fly
            feature_row = _build_features_for_future(bakery, product, date)
            if feature_row is None:
                return jsonify({"success": False, "error": "Нет исторических данных"})

            X = pd.DataFrame([feature_row])[features]
            for col in categorical_cols:
                if col in X.columns:
                    X[col] = X[col].astype("category")

            daily_total = max(float(model.predict(X)[0]), 0)

            # Get category from last known
            last_mask = (df["Пекарня"] == bakery) & (df["Номенклатура"] == product)
            last_rows = df[last_mask].sort_values("Дата", ascending=False)
            category = str(last_rows.iloc[0].get("Категория", "")) if not last_rows.empty else ""

        # Get hourly profile (works for both historical and future)
        profile = get_hourly_profile(hourly_df, bakery, product, category)

        # Build hourly plan: prediction x share
        hourly_plan = []
        for hour in range(6, 23):
            share = profile.get(hour, 0)
            qty = round(daily_total * share, 1)
            hourly_plan.append({
                "hour": hour,
                "label": f"{hour:02d}:00",
                "predicted": qty,
                "share": round(share * 100, 1),
            })

        # Add actual hourly sales if available for this historical date
        if is_historical and not hourly_df.empty:
            actual_mask = (
                (hourly_df["Пекарня"] == bakery) &
                (hourly_df["Номенклатура"] == product) &
                (hourly_df["Дата"] == date)
            )
            actual = hourly_df[actual_mask]
            if not actual.empty:
                actual_by_hour = actual.set_index("Час")["Кол-во"].to_dict()
                for item in hourly_plan:
                    item["actual"] = int(actual_by_hour.get(item["hour"], 0))

        return jsonify({
            "success": True,
            "daily_total": round(daily_total, 1),
            "plan": hourly_plan,
            "is_forecast": not is_historical,
        })

    except Exception as e:
        logger.error(f"daily_plan error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/refresh", methods=["POST"])
def refresh_data():
    """Force re-process all XLSX data."""
    global df, hourly_df, indicators_df, bakeries, products, dates
    global product_category, bakery_city, bakery_products

    try:
        logger.info("Refreshing data...")
        df_new, hourly_new, ind_new = load_or_process(force=True)

        for col in categorical_cols:
            if col in df_new.columns:
                df_new[col] = df_new[col].astype("category")

        df = df_new
        hourly_df = hourly_new
        indicators_df = ind_new

        bakeries = sorted(df["Пекарня"].unique().tolist())
        products = sorted(df["Номенклатура"].unique().tolist())
        dates = sorted(df["Дата"].dt.strftime("%Y-%m-%d").unique().tolist())
        product_category = df.groupby("Номенклатура", observed=True)["Категория"].first().to_dict()
        bakery_city = df.groupby("Пекарня", observed=True)["Город"].first().to_dict()
        bakery_products = (df.groupby("Пекарня", observed=True)["Номенклатура"]
                           .apply(lambda x: sorted(x.unique().tolist())).to_dict())

        logger.info(f"Refresh done. Rows: {len(df):,}")
        return jsonify({"success": True, "rows": len(df), "dates": len(dates)})

    except Exception as e:
        logger.error(f"refresh error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400


# --- Utility ---

def _safe_num(val):
    """Convert to int/float safely, return None for NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return int(val) if float(val) == int(float(val)) else round(float(val), 1)


def _safe_round(val, decimals=1):
    """Round safely, return None for NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return round(float(val), decimals)


def _build_features_for_future(bakery, product, date):
    """Build feature row for a future date using last known values."""
    from src.experiments_v2.common import CITY_COORDS
    
    # Get last known row for this bakery/product
    last_mask = (df["Пекарня"] == bakery) & (df["Номенклатура"] == product)
    last_rows = df[last_mask].sort_values("Дата", ascending=False)
    
    if last_rows.empty:
        return None
    
    last_row = last_rows.iloc[0]
    last_date = last_row["Дата"]
    days_ahead = (date - last_date).days
    
    # Copy all features from last known row
    feature_row = {}
    for col in features:
        if col in last_row.index:
            feature_row[col] = last_row[col]
        else:
            feature_row[col] = 0
    
    # Update calendar features for new date
    feature_row["ДеньНедели"] = date.dayofweek
    feature_row["День"] = date.day
    feature_row["IsWeekend"] = 1 if date.dayofweek >= 5 else 0
    feature_row["Месяц"] = date.month
    
    # Russian holidays
    RU_HOLIDAYS = {(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(2,23),(3,8),(5,1),(5,9),(6,12),(11,4)}
    feature_row["is_holiday"] = 1 if (date.month, date.day) in RU_HOLIDAYS else 0
    
    next_day = date + pd.Timedelta(days=1)
    prev_day = date - pd.Timedelta(days=1)
    feature_row["is_pre_holiday"] = 1 if (next_day.month, next_day.day) in RU_HOLIDAYS else 0
    feature_row["is_post_holiday"] = 1 if (prev_day.month, prev_day.day) in RU_HOLIDAYS else 0
    feature_row["is_payday_week"] = 1 if date.day in [4,5,6,19,20,21] else 0
    feature_row["is_month_start"] = 1 if date.day <= 5 else 0
    feature_row["is_month_end"] = 1 if date.day >= 25 else 0
    
    # Shift lag features based on days ahead
    for lag in [1, 2, 3, 7, 14, 30]:
        new_lag_val = None
        if lag == days_ahead:
            new_lag_val = last_row.get("Продано", 0)
        elif lag > days_ahead:
            old_lag_col = f"sales_lag{days_ahead}" if lag == days_ahead + 1 else None
            if old_lag_col and old_lag_col in last_row.index:
                new_lag_val = last_row[old_lag_col]
        
        if new_lag_val is not None:
            feature_row[f"sales_lag{lag}"] = new_lag_val
            feature_row[f"demand_lag{lag}"] = new_lag_val
    
    # Get city for weather
    city = str(last_row.get("Город", "Казань"))
    
    # Try to get weather forecast
    try:
        weather_df = fetch_weather_forecast(CITY_COORDS, days_ahead=14)
        if not weather_df.empty:
            weather_row = weather_df[(weather_df["Город"] == city) & (weather_df["Дата"] == date)]
            if not weather_row.empty:
                wr = weather_row.iloc[0]
                weather_cols = ["temp_max", "temp_min", "temp_mean", "temp_range", 
                               "precipitation", "rain", "snowfall", "windspeed_max",
                               "is_rainy", "is_snowy", "is_cold", "is_warm", "is_windy", 
                               "is_bad_weather", "weather_cat_code"]
                for wc in weather_cols:
                    if wc in wr.index:
                        feature_row[wc] = wr[wc]
    except Exception as e:
        logger.warning(f"Weather fetch failed: {e}")
    
    return feature_row


@app.route("/api/future_dates", methods=["GET"])
def get_future_dates():
    """Get available future dates for forecasting (next 14 days from last known)."""
    if df.empty:
        return jsonify([])
    
    last_date = df["Дата"].max()
    future_dates = []
    for i in range(1, 15):
        future_date = last_date + pd.Timedelta(days=i)
        future_dates.append(future_date.strftime("%Y-%m-%d"))
    
    return jsonify(future_dates)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
