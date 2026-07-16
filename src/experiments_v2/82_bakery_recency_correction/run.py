"""
Experiment 82: Bakery-level recency correction variants.

Split:
  train  : first 75% of dates  (model training)
  recent : next ~14 days        (simulate "we have recent actuals" for CF calibration)
  test   : last ~10% of dates   (evaluation)

Variants tested:
  0. baseline         - raw LightGBM, no correction
  1. additive_static  - additive bias from recent window (mean(actual-pred))
  2. additive_dynamic - same as 1, but soft-blend: bias = alpha*recent + (1-alpha)*0
  3. multiplicative   - CF = winsorized_mean(actual/pred) per bakery
  4. soft_blend_cf    - CF with alpha = min(0.8, obs/14); CF_final = alpha*CF + (1-alpha)*1.0
  5. dow_aware_cf     - separate CF for weekday / weekend

Input:  data/processed/daily_sales_stg.csv
Output: src/experiments_v2/82_bakery_recency_correction/metrics.json
        src/experiments_v2/82_bakery_recency_correction/variant_summary.csv
"""

import sys, json, time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

out = sys.stdout.buffer
def p(s): out.write((s + '\n').encode('utf-8')); out.flush()

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent

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
BAKERY_COL = "Пекарня"
TARGET = "Продано"
RECENT_DAYS = 14

MODEL_PARAMS = dict(
    n_estimators=1000, learning_rate=0.05, num_leaves=127,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1,
    objective="quantile", alpha=0.5,
    n_jobs=-1, random_state=42, verbose=-1,
)


def prep_X(df, features):
    X = df[features].copy()
    for col in CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X


def metrics(actual, pred, label):
    mae  = mean_absolute_error(actual, pred)
    bias = float(np.mean(pred - actual))
    wm   = float(np.sum(np.abs(actual - pred)) / np.sum(actual)) * 100
    p(f"  {label:<30} MAE={mae:.4f}  Bias={bias:+.3f}  WMAPE={wm:.2f}%")
    return {"label": label, "mae": round(mae, 4), "bias": round(bias, 4), "wmape": round(wm, 2)}


# ── Load data ─────────────────────────────────────────────────────────────────
p("Loading data...")
df = pd.read_csv(ROOT / "data/processed/daily_sales_stg.csv", encoding="utf-8-sig", low_memory=False)
df["Дата"] = pd.to_datetime(df["Дата"])
df = df.sort_values("Дата").reset_index(drop=True)

dates = sorted(df["Дата"].unique())
n = len(dates)
cut_train  = dates[int(n * 0.75)]
cut_recent = dates[min(int(n * 0.75) + RECENT_DAYS, n - 1)]

train  = df[df["Дата"] <  cut_train].copy()
recent = df[(df["Дата"] >= cut_train) & (df["Дата"] < cut_recent)].copy()
test   = df[df["Дата"] >= cut_recent].copy()

p(f"  train : {train['Дата'].min().date()} .. {train['Дата'].max().date()}  ({len(train):,} rows)")
p(f"  recent: {recent['Дата'].min().date()} .. {recent['Дата'].max().date()}  ({len(recent):,} rows, {recent['Дата'].nunique()} days)")
p(f"  test  : {test['Дата'].min().date()} .. {test['Дата'].max().date()}  ({len(test):,} rows)")

# ── Train model ───────────────────────────────────────────────────────────────
p("\nTraining model...")
t0 = time.time()
model = LGBMRegressor(**MODEL_PARAMS)
model.fit(prep_X(train, FEATURES_STG), train[TARGET], categorical_feature=CATEGORICAL_COLS)
p(f"  done in {time.time()-t0:.1f}s")

# ── Predict on recent + test ──────────────────────────────────────────────────
recent_pred = np.clip(model.predict(prep_X(recent, FEATURES_STG)), 0, None)
test_pred   = np.clip(model.predict(prep_X(test,   FEATURES_STG)), 0, None)

recent = recent.copy(); recent["pred"] = recent_pred
test   = test.copy();   test["pred"]   = test_pred

actual_test = test[TARGET].values
p(f"\n{'='*70}")
p("RESULTS")
p(f"{'='*70}")

results = []

# ── Variant 0: baseline ───────────────────────────────────────────────────────
results.append(metrics(actual_test, test_pred, "0_baseline"))

# ── Variant 1: additive static bias ──────────────────────────────────────────
# bias[bakery] = mean(actual - pred) in recent window (no damping)
bias1 = (
    recent.groupby(BAKERY_COL)
    .apply(lambda g: float(np.mean(g[TARGET] - g["pred"])))
    .rename("bias")
    .reset_index()
)
test1 = test.merge(bias1, on=BAKERY_COL, how="left")
test1["bias"] = test1["bias"].fillna(0.0)
pred1 = np.clip(test_pred + test1["bias"].values, 0, None)
results.append(metrics(actual_test, pred1, "1_additive_static"))

# ── Variant 2: additive dynamic (soft-blend toward 0) ────────────────────────
# alpha = min(0.8, obs / RECENT_DAYS)
obs2 = recent.groupby(BAKERY_COL)["Дата"].nunique().rename("obs").reset_index()
bias2 = bias1.merge(obs2, on=BAKERY_COL, how="left")
bias2["alpha"] = np.minimum(0.8, bias2["obs"] / RECENT_DAYS)
bias2["bias_damped"] = bias2["alpha"] * bias2["bias"]  # blend toward 0
test2 = test.merge(bias2[[BAKERY_COL, "bias_damped"]], on=BAKERY_COL, how="left")
test2["bias_damped"] = test2["bias_damped"].fillna(0.0)
pred2 = np.clip(test_pred + test2["bias_damped"].values, 0, None)
results.append(metrics(actual_test, pred2, "2_additive_dynamic"))

# ── Variant 3: multiplicative CF ─────────────────────────────────────────────
# CF[bakery] = winsorized_mean(actual / pred) in recent window
def winsor_mean(s, lo=0.05, hi=0.95):
    q_lo, q_hi = s.quantile(lo), s.quantile(hi)
    return float(s.clip(q_lo, q_hi).mean())

recent3 = recent.copy()
recent3["ratio"] = recent3[TARGET] / np.where(recent3["pred"] > 0.5, recent3["pred"], 0.5)

cf3 = (
    recent3.groupby(BAKERY_COL)["ratio"]
    .apply(winsor_mean)
    .rename("cf")
    .reset_index()
)
cf3["cf"] = cf3["cf"].clip(0.5, 2.0)
test3 = test.merge(cf3, on=BAKERY_COL, how="left")
test3["cf"] = test3["cf"].fillna(1.0)
pred3 = np.clip(test_pred * test3["cf"].values, 0, None)
results.append(metrics(actual_test, pred3, "3_multiplicative_cf"))

# ── Variant 4: soft blend CF (alpha by obs count) ────────────────────────────
obs4 = recent.groupby(BAKERY_COL)["Дата"].nunique().rename("obs").reset_index()
cf4 = cf3.merge(obs4, on=BAKERY_COL, how="left")
cf4["alpha"] = np.minimum(0.8, cf4["obs"] / RECENT_DAYS)
cf4["cf_blended"] = cf4["alpha"] * cf4["cf"] + (1 - cf4["alpha"]) * 1.0
test4 = test.merge(cf4[[BAKERY_COL, "cf_blended"]], on=BAKERY_COL, how="left")
test4["cf_blended"] = test4["cf_blended"].fillna(1.0)
pred4 = np.clip(test_pred * test4["cf_blended"].values, 0, None)
results.append(metrics(actual_test, pred4, "4_soft_blend_cf"))

# ── Variant 5: DOW-aware CF (weekday / weekend separate) ─────────────────────
recent5 = recent.copy()
recent5["is_weekend"] = recent5["ДеньНедели"].isin([5, 6]).astype(int)
recent5["ratio"] = recent5[TARGET] / np.where(recent5["pred"] > 0.5, recent5["pred"], 0.5)

cf5 = (
    recent5.groupby([BAKERY_COL, "is_weekend"])["ratio"]
    .apply(winsor_mean)
    .rename("cf")
    .reset_index()
)
cf5["cf"] = cf5["cf"].clip(0.5, 2.0)

# obs per bakery-weekpart
obs5 = recent5.groupby([BAKERY_COL, "is_weekend"])["Дата"].nunique().rename("obs").reset_index()
cf5 = cf5.merge(obs5, on=[BAKERY_COL, "is_weekend"], how="left")
cf5["alpha"] = np.minimum(0.8, cf5["obs"] / (RECENT_DAYS / 2))
cf5["cf_blended"] = cf5["alpha"] * cf5["cf"] + (1 - cf5["alpha"]) * 1.0

test5 = test.copy()
test5["is_weekend"] = test5["ДеньНедели"].isin([5, 6]).astype(int)
test5 = test5.merge(cf5[[BAKERY_COL, "is_weekend", "cf_blended"]], on=[BAKERY_COL, "is_weekend"], how="left")
test5["cf_blended"] = test5["cf_blended"].fillna(1.0)
pred5 = np.clip(test_pred * test5["cf_blended"].values, 0, None)
results.append(metrics(actual_test, pred5, "5_dow_aware_cf"))

# ── Summary ───────────────────────────────────────────────────────────────────
p(f"\n{'='*70}")
p("SUMMARY vs baseline")
p(f"{'='*70}")
base_mae = results[0]["mae"]
for r in results[1:]:
    delta = r["mae"] - base_mae
    sign = "✓" if delta < 0 else "✗"
    p(f"  {r['label']:<30} delta={delta:+.4f}  {sign}")

# ── Bakery-level breakdown for best variant ───────────────────────────────────
best = min(results[1:], key=lambda r: r["mae"])
p(f"\nBest variant: {best['label']}")

# CF distribution for multiplicative variants
if "cf" in cf3.columns:
    p(f"\nCF distribution (variant 3):")
    p(f"  median={cf3['cf'].median():.3f}  mean={cf3['cf'].mean():.3f}  "
      f"  p10={cf3['cf'].quantile(0.1):.3f}  p90={cf3['cf'].quantile(0.9):.3f}")
    p(f"  bakeries with CF>1.1: {(cf3['cf']>1.1).sum()}  CF<0.9: {(cf3['cf']<0.9).sum()}  total: {len(cf3)}")

# ── Save ─────────────────────────────────────────────────────────────────────
EXP_DIR.mkdir(parents=True, exist_ok=True)
summary_df = pd.DataFrame(results)
summary_df.to_csv(EXP_DIR / "variant_summary.csv", index=False)

out_json = {
    "experiment": "82_bakery_recency_correction",
    "dataset": "daily_sales_stg.csv",
    "recent_days": RECENT_DAYS,
    "train_dates": f"{train['Дата'].min().date()} .. {train['Дата'].max().date()}",
    "recent_dates": f"{recent['Дата'].min().date()} .. {recent['Дата'].max().date()}",
    "test_dates": f"{test['Дата'].min().date()} .. {test['Дата'].max().date()}",
    "variants": {r["label"]: {k: v for k, v in r.items() if k != "label"} for r in results},
}
with open(EXP_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(out_json, f, ensure_ascii=False, indent=2)
p(f"\nSaved -> {EXP_DIR / 'metrics.json'}")
