"""
Analyze weather impact on residuals by season.
Train baseline model -> compute residuals -> breakdown by season x weather.
Also fit proportional correction slopes per season (like exp 69 but split).
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

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
TEST_DAYS = 7
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
df["is_warm_season"] = df["Месяц"].isin([4,5,6,7,8,9]).astype(int)
p(f"  {len(df):,} rows  {df['Дата'].min().date()} .. {df['Дата'].max().date()}")

# Use ALL data for analysis (not just last 7 days test)
# Train on first 80% of dates, analyze residuals on remaining 20%
dates_sorted = sorted(df["Дата"].unique())
cutoff_idx = int(len(dates_sorted) * 0.8)
cutoff = dates_sorted[cutoff_idx]
train = df[df["Дата"] < cutoff].copy()
test  = df[df["Дата"] >= cutoff].copy()
p(f"  train: {len(train):,} rows ({dates_sorted[0].date()} .. {cutoff.date()})")
p(f"  test:  {len(test):,} rows  ({cutoff.date()} .. {dates_sorted[-1].date()})")
p(f"  test warm rows: {(test['is_warm_season']==1).sum():,}  cold rows: {(test['is_warm_season']==0).sum():,}")

# Train baseline
p("\nTraining baseline model...")
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

actual = test[TARGET].values
resid = actual - pred   # positive = under-forecast, negative = over-forecast
ratio = actual / np.where(pred > 0.5, pred, 0.5)  # actual/pred, clipped pred floor

test = test.copy()
test["pred"] = pred
test["resid"] = resid
test["ratio"] = ratio.clip(0.2, 5.0)

mae_all = mean_absolute_error(actual, pred)
bias_all = float(np.mean(pred - actual))
p(f"\n  Overall MAE={mae_all:.4f}  Bias={bias_all:+.2f}")

# ── 1. Season breakdown ────────────────────────────────────────────────────────
p("\n" + "=" * 70)
p("1. Season breakdown")
p("=" * 70)
for season, label in [(1, "WARM (Apr-Sep)"), (0, "COLD (Oct-Mar)")]:
    m = test["is_warm_season"] == season
    if not m.any(): continue
    mae = mean_absolute_error(actual[m], pred[m])
    bias = float(np.mean(pred[m] - actual[m]))
    p(f"  {label}: n={m.sum():,}  MAE={mae:.4f}  Bias={bias:+.2f}")

# ── 2. Season x Weather breakdown ─────────────────────────────────────────────
p("\n" + "=" * 70)
p("2. Season x Weather breakdown")
p("=" * 70)
p(f"  {'Season':<12} {'Weather':<25} {'n':>7}  {'MAE':>7}  {'Bias':>7}  {'mean_ratio':>10}")
p(f"  {'-'*12} {'-'*25} {'-'*7}  {'-'*7}  {'-'*7}  {'-'*10}")

weather_groups = [
    ("clear",    test["precipitation"] < 0.5),
    ("light",    (test["precipitation"] >= 0.5) & (test["precipitation"] < 3)),
    ("moderate", (test["precipitation"] >= 3) & (test["precipitation"] < 8)),
    ("heavy",    test["precipitation"] >= 8),
]
for season, slabel in [(1, "WARM"), (0, "COLD")]:
    sm = test["is_warm_season"] == season
    for wlabel, wm in weather_groups:
        m = sm & wm
        if m.sum() < 10: continue
        mae = mean_absolute_error(actual[m], pred[m])
        bias = float(np.mean(pred[m] - actual[m]))
        mr = float(test.loc[m, "ratio"].mean())
        p(f"  {slabel:<12} {wlabel:<25} {m.sum():>7,}  {mae:>7.4f}  {bias:>+7.2f}  {mr:>10.4f}")

# ── 3. Precipitation bins x season (for proportional correction) ───────────────
p("\n" + "=" * 70)
p("3. Proportional correction: mean ratio by precip bin x season")
p("=" * 70)
p("  (ratio = actual/pred; >1 means model under-forecasts)")
bins = [0, 0.5, 1, 2, 4, 6, 8, 12, 20, 50]
labels_b = ["0", "0-1", "1-2", "2-4", "4-6", "6-8", "8-12", "12-20", "20+"]
test["precip_bin"] = pd.cut(test["precipitation"], bins=bins, labels=labels_b, right=False)

p(f"\n  {'precip_bin':<12} {'WARM n':>8} {'WARM ratio':>11} {'COLD n':>8} {'COLD ratio':>11}")
p(f"  {'-'*12} {'-'*8} {'-'*11} {'-'*8} {'-'*11}")
for lb in labels_b:
    bm = test["precip_bin"] == lb
    warm_m = bm & (test["is_warm_season"] == 1)
    cold_m = bm & (test["is_warm_season"] == 0)
    wn = warm_m.sum(); cn = cold_m.sum()
    wr = f"{test.loc[warm_m,'ratio'].mean():.4f}" if wn > 10 else "  --"
    cr = f"{test.loc[cold_m,'ratio'].mean():.4f}" if cn > 10 else "  --"
    p(f"  {lb:<12} {wn:>8,} {wr:>11} {cn:>8,} {cr:>11}")

# ── 4. Fit linear slopes (like exp 69) per season ─────────────────────────────
p("\n" + "=" * 70)
p("4. Linear correction slope: CF = 1 + slope * precip  (per season)")
p("=" * 70)

for season, slabel in [(1, "WARM (Apr-Sep)"), (0, "COLD (Oct-Mar)")]:
    sm = test["is_warm_season"] == season
    sub = test[sm & (test["precipitation"] > 0.5)].copy()
    if len(sub) < 50:
        p(f"  {slabel}: not enough data")
        continue
    # Winsorize ratio
    lo, hi = sub["ratio"].quantile(0.02), sub["ratio"].quantile(0.98)
    sub["ratio_w"] = sub["ratio"].clip(lo, hi)
    # Fit: (ratio - 1) ~ precip * slope, no intercept
    X_fit = sub["precipitation"].values.reshape(-1, 1)
    y_fit = (sub["ratio_w"] - 1).values
    reg = LinearRegression(fit_intercept=False).fit(X_fit, y_fit)
    slope = reg.coef_[0]
    precip_ref = float(sub["precipitation"].median())
    cf_ref = 1 + slope * precip_ref
    p(f"\n  {slabel}:")
    p(f"    slope={slope:+.5f}  precip_median={precip_ref:.1f}mm  CF@median={cf_ref:.4f}")
    p(f"    CF table: 0mm->1.00  2mm->{1+slope*2:.3f}  4mm->{1+slope*4:.3f}  "
      f"8mm->{1+slope*8:.3f}  15mm->{1+slope*15:.3f}")
    p(f"    Interpretation: {'rain depresses demand (CF>1 = under-forecast)' if slope > 0 else 'precip does NOT depress demand (model already over-forecasts)'}")

p("\nDone.")
