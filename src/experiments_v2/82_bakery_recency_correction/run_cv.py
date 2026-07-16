"""
Exp 82 CV: Rolling-window cross-validation for bakery recency correction.

3 folds, fixed 13-month training window:
  Fold 1: train Feb 2025-Mar 2026 | recent N days | test Apr 2026
  Fold 2: train Mar 2025-Apr 2026 | recent N days | test May 2026
  Fold 3: train Apr 2025-May 2026 | recent N days | test Jun 2026

Recent window sizes: 14 and 28 days.
Variants per fold: 0_baseline, 1_additive_static, 2_additive_dynamic,
                   3_multiplicative_cf, 4_soft_blend_cf, 5_dow_aware_cf
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

MODEL_PARAMS = dict(
    n_estimators=1000, learning_rate=0.05, num_leaves=127,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1,
    objective="quantile", alpha=0.5,
    n_jobs=-1, random_state=42, verbose=-1,
)

FOLDS = [
    {"name": "fold1_apr", "train_from": "2025-02-01", "train_to": "2026-03-31",
     "test_from": "2026-04-01", "test_to": "2026-04-30"},
    {"name": "fold2_may", "train_from": "2025-03-01", "train_to": "2026-04-30",
     "test_from": "2026-05-01", "test_to": "2026-05-31"},
    {"name": "fold3_jun", "train_from": "2025-04-01", "train_to": "2026-05-31",
     "test_from": "2026-06-01", "test_to": "2026-06-30"},
]
RECENT_WINDOWS = [14, 28]


def prep_X(df):
    X = df[FEATURES_STG].copy()
    for col in CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X


def winsor_mean(s, lo=0.05, hi=0.95):
    q_lo, q_hi = s.quantile(lo), s.quantile(hi)
    return float(s.clip(q_lo, q_hi).mean())


def apply_variants(test, test_pred, recent, recent_pred, recent_days):
    actual = test[TARGET].values
    results = {}

    # 0: baseline
    results["0_baseline"] = test_pred.copy()

    # recent with predictions
    rec = recent.copy()
    rec["pred"] = recent_pred
    rec["ratio"] = rec[TARGET] / np.where(rec["pred"] > 0.5, rec["pred"], 0.5)

    # obs per bakery (unique days)
    obs = rec.groupby(BAKERY_COL)["Дата"].nunique().rename("obs").reset_index()

    # 1: additive static
    bias1 = (
        rec.groupby(BAKERY_COL)
        .apply(lambda g: float(np.mean(g[TARGET] - g["pred"])), include_groups=False)
        .rename("bias").reset_index()
    )
    t1 = test.merge(bias1, on=BAKERY_COL, how="left")
    t1["bias"] = t1["bias"].fillna(0.0)
    results["1_additive_static"] = np.clip(test_pred + t1["bias"].values, 0, None)

    # 2: additive dynamic (alpha blend toward 0)
    b2 = bias1.merge(obs, on=BAKERY_COL, how="left")
    b2["alpha"] = np.minimum(0.8, b2["obs"] / recent_days)
    b2["bias_d"] = b2["alpha"] * b2["bias"]
    t2 = test.merge(b2[[BAKERY_COL, "bias_d"]], on=BAKERY_COL, how="left")
    t2["bias_d"] = t2["bias_d"].fillna(0.0)
    results["2_additive_dynamic"] = np.clip(test_pred + t2["bias_d"].values, 0, None)

    # 3: multiplicative CF
    cf3 = (
        rec.groupby(BAKERY_COL)["ratio"]
        .apply(winsor_mean).rename("cf").reset_index()
    )
    cf3["cf"] = cf3["cf"].clip(0.5, 2.0)
    t3 = test.merge(cf3, on=BAKERY_COL, how="left")
    t3["cf"] = t3["cf"].fillna(1.0)
    results["3_multiplicative_cf"] = np.clip(test_pred * t3["cf"].values, 0, None)

    # 4: soft blend CF
    cf4 = cf3.merge(obs, on=BAKERY_COL, how="left")
    cf4["alpha"] = np.minimum(0.8, cf4["obs"] / recent_days)
    cf4["cf_b"] = cf4["alpha"] * cf4["cf"] + (1 - cf4["alpha"]) * 1.0
    t4 = test.merge(cf4[[BAKERY_COL, "cf_b"]], on=BAKERY_COL, how="left")
    t4["cf_b"] = t4["cf_b"].fillna(1.0)
    results["4_soft_blend_cf"] = np.clip(test_pred * t4["cf_b"].values, 0, None)

    # 5: DOW-aware CF
    rec5 = rec.copy()
    rec5["is_weekend"] = rec5["ДеньНедели"].isin([5, 6]).astype(int)
    cf5 = (
        rec5.groupby([BAKERY_COL, "is_weekend"])["ratio"]
        .apply(winsor_mean).rename("cf").reset_index()
    )
    cf5["cf"] = cf5["cf"].clip(0.5, 2.0)
    obs5 = rec5.groupby([BAKERY_COL, "is_weekend"])["Дата"].nunique().rename("obs").reset_index()
    cf5 = cf5.merge(obs5, on=[BAKERY_COL, "is_weekend"], how="left")
    cf5["alpha"] = np.minimum(0.8, cf5["obs"] / (recent_days / 2))
    cf5["cf_b"] = cf5["alpha"] * cf5["cf"] + (1 - cf5["alpha"]) * 1.0
    t5 = test.copy()
    t5["is_weekend"] = t5["ДеньНедели"].isin([5, 6]).astype(int)
    t5 = t5.merge(cf5[[BAKERY_COL, "is_weekend", "cf_b"]], on=[BAKERY_COL, "is_weekend"], how="left")
    t5["cf_b"] = t5["cf_b"].fillna(1.0)
    results["5_dow_aware_cf"] = np.clip(test_pred * t5["cf_b"].values, 0, None)

    # Compute metrics
    metrics = {}
    for name, pred in results.items():
        mae  = mean_absolute_error(actual, pred)
        bias = float(np.mean(pred - actual))
        wm   = float(np.sum(np.abs(actual - pred)) / np.sum(actual)) * 100
        metrics[name] = {"mae": round(mae, 4), "bias": round(bias, 4), "wmape": round(wm, 2)}
    return metrics


# ── Load data ──────────────────────────────────────────────────────────────────
p("Loading data...")
df = pd.read_csv(ROOT / "data/processed/daily_sales_stg.csv", encoding="utf-8-sig", low_memory=False)
df["Дата"] = pd.to_datetime(df["Дата"])
p(f"  {len(df):,} rows  {df['Дата'].min().date()} .. {df['Дата'].max().date()}")

all_results = {}

for fold in FOLDS:
    p(f"\n{'='*70}")
    p(f"FOLD: {fold['name']}  train {fold['train_from']}..{fold['train_to']}  test {fold['test_from']}..{fold['test_to']}")
    p(f"{'='*70}")

    train = df[(df["Дата"] >= fold["train_from"]) & (df["Дата"] <= fold["train_to"])].copy()
    test  = df[(df["Дата"] >= fold["test_from"])  & (df["Дата"] <= fold["test_to"])].copy()

    p(f"  train: {len(train):,} rows  test: {len(test):,} rows")

    t0 = time.time()
    model = LGBMRegressor(**MODEL_PARAMS)
    model.fit(prep_X(train), train[TARGET], categorical_feature=CATEGORICAL_COLS)
    p(f"  trained in {time.time()-t0:.1f}s")

    test_pred = np.clip(model.predict(prep_X(test)), 0, None)

    for rdays in RECENT_WINDOWS:
        recent_end   = pd.Timestamp(fold["test_from"]) - pd.Timedelta(days=1)
        recent_start = recent_end - pd.Timedelta(days=rdays - 1)
        recent = df[(df["Дата"] >= recent_start) & (df["Дата"] <= recent_end)].copy()
        recent_pred = np.clip(model.predict(prep_X(recent)), 0, None)

        p(f"\n  recent_{rdays}d: {recent_start.date()} .. {recent_end.date()}  ({len(recent):,} rows)")

        fold_metrics = apply_variants(test, test_pred, recent, recent_pred, rdays)

        key = f"{fold['name']}_recent{rdays}d"
        all_results[key] = fold_metrics

        base_mae = fold_metrics["0_baseline"]["mae"]
        for vname, m in fold_metrics.items():
            delta = m["mae"] - base_mae
            sign = "✓" if delta < -0.001 else ("=" if abs(delta) <= 0.001 else "✗")
            p(f"    {vname:<30} MAE={m['mae']:.4f}  Bias={m['bias']:+.3f}  WMAPE={m['wmape']:.1f}%  delta={delta:+.4f} {sign}")

# ── Aggregate across folds ────────────────────────────────────────────────────
p(f"\n{'='*70}")
p("AGGREGATE MAE across 3 folds (mean)")
p(f"{'='*70}")

variants = list(next(iter(all_results.values())).keys())
for rdays in RECENT_WINDOWS:
    p(f"\n  recent_window = {rdays} days:")
    fold_keys = [k for k in all_results if f"recent{rdays}d" in k]
    for v in variants:
        maes = [all_results[k][v]["mae"] for k in fold_keys]
        biases = [all_results[k][v]["bias"] for k in fold_keys]
        mean_mae = np.mean(maes)
        mean_bias = np.mean(biases)
        base_maes = [all_results[k]["0_baseline"]["mae"] for k in fold_keys]
        delta = mean_mae - np.mean(base_maes)
        sign = "✓" if delta < -0.001 else ("=" if abs(delta) <= 0.001 else "✗")
        p(f"    {v:<30} MAE={mean_mae:.4f}  Bias={mean_bias:+.3f}  delta={delta:+.4f} {sign}")

# Save
EXP_DIR.mkdir(parents=True, exist_ok=True)
with open(EXP_DIR / "cv_metrics.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
p(f"\nSaved -> {EXP_DIR / 'cv_metrics.json'}")
