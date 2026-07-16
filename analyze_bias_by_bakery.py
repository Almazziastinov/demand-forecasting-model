"""
Analyze bias in test period (Apr-Jun 2026) by bakery and by month.
Uses saved predictions from analyze_weather_seasonal.py approach — retrains quickly.
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

out = sys.stdout.buffer
def p(s): out.write((s + '\n').encode('utf-8')); out.flush()

ROOT = Path(__file__).resolve().parent

FEATURES_STG = [
    "Пекарня", "Номенклатура", "Категория", "Город",
    "ДеньНедели", "День", "IsWeekend", "Месяц", "НомерНедели",
    "sales_lag1", "sales_lag2", "sales_lag3", "sales_lag7",
    "sales_lag14", "sales_lag30",
    "sales_roll_mean3", "sales_roll_mean7", "sales_roll_std7",
    "sales_roll_mean14", "sales_roll_mean30",
    "is_holiday", "is_pre_holiday", "is_post_holiday", "is_payday_week",
    "is_month_start", "is_month_end",
    "temp_mean", "temp_range",
    "precipitation", "rain", "snowfall", "windspeed_max",
    "is_rainy", "is_snowy", "is_cold", "is_warm",
    "is_windy", "is_bad_weather", "weather_cat_code",
]
CATEGORICAL_COLS = ["Пекарня", "Номенклатура", "Категория", "Город", "Месяц"]
TARGET = "Продано"
MODEL_PARAMS = dict(
    n_estimators=1000, learning_rate=0.05, num_leaves=127,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1,
    objective="quantile", alpha=0.5,
    n_jobs=-1, random_state=42, verbose=-1,
)

p("Loading data...")
df = pd.read_csv(ROOT / "data/processed/daily_sales_stg.csv", encoding="utf-8-sig", low_memory=False)
df["Дата"] = pd.to_datetime(df["Дата"])
df["YM"] = df["Дата"].dt.to_period("M").astype(str)

dates_sorted = sorted(df["Дата"].unique())
cutoff_idx = int(len(dates_sorted) * 0.8)
cutoff = dates_sorted[cutoff_idx]

train = df[df["Дата"] < cutoff].copy()
test  = df[df["Дата"] >= cutoff].copy()
p(f"  train: {train['Дата'].min().date()} .. {train['Дата'].max().date()}  ({len(train):,})")
p(f"  test:  {test['Дата'].min().date()} .. {test['Дата'].max().date()}  ({len(test):,})")

p("Training model...")
t0 = time.time()
X_tr = train[FEATURES_STG].copy()
X_te = test[FEATURES_STG].copy()
for col in CATEGORICAL_COLS:
    X_tr[col] = X_tr[col].astype("category")
    X_te[col] = X_te[col].astype("category")
model = LGBMRegressor(**MODEL_PARAMS)
model.fit(X_tr, train[TARGET], categorical_feature=CATEGORICAL_COLS)
pred = np.clip(model.predict(X_te), 0, None)
p(f"  done in {time.time()-t0:.1f}s")

test = test.copy()
test["pred"] = pred
test["resid"] = test[TARGET] - pred       # positive = under-forecast
test["bias_item"] = pred - test[TARGET]   # positive = over-forecast

# ── 1. По месяцам ──────────────────────────────────────────────────────────────
p("\n" + "=" * 70)
p("1. Смещение по месяцам (тест-период)")
p("=" * 70)
p(f"  {'Месяц':<10} {'n':>8}  {'MAE':>7}  {'Bias':>8}  {'Bias%':>8}  {'mean_actual':>12}  {'mean_pred':>10}")
p(f"  {'-'*10} {'-'*8}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*12}  {'-'*10}")
for ym, g in test.groupby("YM"):
    mae  = mean_absolute_error(g[TARGET], g["pred"])
    bias = float(g["bias_item"].mean())
    ma   = float(g[TARGET].mean())
    mp   = float(g["pred"].mean())
    bpct = bias / ma * 100 if ma > 0 else 0
    p(f"  {ym:<10} {len(g):>8,}  {mae:>7.4f}  {bias:>+8.2f}  {bpct:>+7.1f}%  {ma:>12.2f}  {mp:>10.2f}")

# ── 2. По пекарням (топ-20 по абсолютному bias) ────────────────────────────────
p("\n" + "=" * 70)
p("2. Смещение по пекарням — топ-20 по абсолютному bias")
p("=" * 70)
bak = (
    test.groupby("Пекарня")
    .agg(
        n        = (TARGET, "count"),
        actual   = (TARGET, "mean"),
        pred     = ("pred", "mean"),
        bias     = ("bias_item", "mean"),
        mae      = ("resid", lambda x: np.abs(x).mean()),
    )
    .reset_index()
)
bak["bias_pct"] = bak["bias"] / bak["actual"] * 100
bak["abs_bias"] = bak["bias"].abs()
bak = bak.sort_values("abs_bias", ascending=False)

p(f"  {'Пекарня':<40} {'n':>6}  {'actual':>7}  {'pred':>7}  {'bias':>7}  {'bias%':>7}")
p(f"  {'-'*40} {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
for _, row in bak.head(20).iterrows():
    name = str(row["Пекарня"])[:40]
    p(f"  {name:<40} {int(row['n']):>6,}  {row['actual']:>7.1f}  {row['pred']:>7.1f}  {row['bias']:>+7.2f}  {row['bias_pct']:>+6.1f}%")

# ── 3. Распределение bias по пекарням ─────────────────────────────────────────
p("\n" + "=" * 70)
p("3. Распределение bias% по пекарням")
p("=" * 70)
pcts = bak["bias_pct"]
p(f"  Медиана bias%:  {pcts.median():+.1f}%")
p(f"  Среднее  bias%: {pcts.mean():+.1f}%")
p(f"  Пекарни с bias > +10%:  {(pcts > 10).sum()} из {len(bak)}")
p(f"  Пекарни с bias < -10%:  {(pcts < -10).sum()} из {len(bak)}")
p(f"  Пекарни с |bias| < 5%:  {(pcts.abs() < 5).sum()} из {len(bak)}")

# ── 4. Bias по городам ─────────────────────────────────────────────────────────
p("\n" + "=" * 70)
p("4. Смещение по городам")
p("=" * 70)
city = (
    test.groupby("Город")
    .agg(n=(TARGET,"count"), actual=(TARGET,"mean"),
         pred=("pred","mean"), bias=("bias_item","mean"))
    .reset_index()
)
city["bias_pct"] = city["bias"] / city["actual"] * 100
for _, row in city.sort_values("bias_pct").iterrows():
    p(f"  {str(row['Город']):<25} n={int(row['n']):>7,}  actual={row['actual']:>6.1f}  bias={row['bias']:>+6.2f}  ({row['bias_pct']:>+5.1f}%)")

p("\nDone.")
