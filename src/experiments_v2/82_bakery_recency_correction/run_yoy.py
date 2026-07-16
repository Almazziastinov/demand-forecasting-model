"""
Exp 82 YoY: Year-over-year correction for bakery-day forecasts.

Hypothesis: same month last year is a better calibration signal than
recent 14-28 days, because seasonal patterns repeat and the model's
systematic errors in e.g. April repeat year-over-year.

Two experiments:
  A. YoY correction factor:
     CF[bakery] = winsorized_mean(actual/pred) in same month last year
     Apply to current month forecast.

  B. YoY feature in model:
     Add sales_lag365 (same day last year) to FEATURES_STG.
     Check feature importance gain%.

Same 3-fold CV structure as run_cv.py:
  Fold 1: train Feb 2025-Mar 2026 | yoy_ref Apr 2025 | test Apr 2026
  Fold 2: train Mar 2025-Apr 2026 | yoy_ref May 2025 | test May 2026
  Fold 3: train Apr 2025-May 2026 | yoy_ref Jun 2025 | test Jun 2026
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

FEATURES_BASE = [
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
     "test_from": "2026-04-01", "test_to": "2026-04-30",
     "yoy_from":  "2025-04-01", "yoy_to":   "2025-04-30"},
    {"name": "fold2_may", "train_from": "2025-03-01", "train_to": "2026-04-30",
     "test_from": "2026-05-01", "test_to": "2026-05-31",
     "yoy_from":  "2025-05-01", "yoy_to":   "2025-05-31"},
    {"name": "fold3_jun", "train_from": "2025-04-01", "train_to": "2026-05-31",
     "test_from": "2026-06-01", "test_to": "2026-06-30",
     "yoy_from":  "2025-06-01", "yoy_to":   "2025-06-30"},
]


def prep_X(df, features):
    X = df[features].copy()
    for col in CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X


def winsor_mean(s, lo=0.05, hi=0.95):
    q_lo, q_hi = s.quantile(lo), s.quantile(hi)
    return float(s.clip(q_lo, q_hi).mean())


def add_lag365(df_all, df_target):
    """Add sales_lag365: actual sales same day last year for each bakery+product."""
    ref = df_all[["Дата", BAKERY_COL, "Номенклатура", TARGET]].copy()
    ref["Дата_next"] = ref["Дата"] + pd.DateOffset(years=1)
    ref = ref.rename(columns={TARGET: "sales_lag365"})[["Дата_next", BAKERY_COL, "Номенклатура", "sales_lag365"]]
    ref = ref.rename(columns={"Дата_next": "Дата"})
    merged = df_target.merge(ref, on=["Дата", BAKERY_COL, "Номенклатура"], how="left")
    merged["sales_lag365"] = merged["sales_lag365"].fillna(merged["sales_roll_mean30"])
    return merged


# ── Load data ──────────────────────────────────────────────────────────────────
p("Loading data...")
df = pd.read_csv(ROOT / "data/processed/daily_sales_stg.csv", encoding="utf-8-sig", low_memory=False)
df["Дата"] = pd.to_datetime(df["Дата"])
p(f"  {len(df):,} rows  {df['Дата'].min().date()} .. {df['Дата'].max().date()}")

all_results = {}
importances = {}

for fold in FOLDS:
    p(f"\n{'='*70}")
    p(f"FOLD: {fold['name']}  train {fold['train_from']}..{fold['train_to']}  test {fold['test_from']}..{fold['test_to']}")
    p(f"  YoY reference: {fold['yoy_from']} .. {fold['yoy_to']}")
    p(f"{'='*70}")

    train = df[(df["Дата"] >= fold["train_from"]) & (df["Дата"] <= fold["train_to"])].copy()
    test  = df[(df["Дата"] >= fold["test_from"])  & (df["Дата"] <= fold["test_to"])].copy()
    yoy   = df[(df["Дата"] >= fold["yoy_from"])   & (df["Дата"] <= fold["yoy_to"])].copy()

    p(f"  train: {len(train):,}  test: {len(test):,}  yoy_ref: {len(yoy):,} rows")

    # ── Part A: YoY correction (baseline model, no lag365) ─────────────────
    p("\n  [A] Training baseline model (no lag365)...")
    t0 = time.time()
    model_base = LGBMRegressor(**MODEL_PARAMS)
    model_base.fit(prep_X(train, FEATURES_BASE), train[TARGET],
                   categorical_feature=CATEGORICAL_COLS)
    p(f"      trained in {time.time()-t0:.1f}s")

    test_pred  = np.clip(model_base.predict(prep_X(test, FEATURES_BASE)), 0, None)
    yoy_pred   = np.clip(model_base.predict(prep_X(yoy,  FEATURES_BASE)), 0, None)

    actual_test = test[TARGET].values
    mae_base = mean_absolute_error(actual_test, test_pred)
    bias_base = float(np.mean(test_pred - actual_test))
    p(f"      0_baseline: MAE={mae_base:.4f}  Bias={bias_base:+.3f}")

    # YoY CF per bakery
    yoy_w = yoy.copy()
    yoy_w["pred"] = yoy_pred
    yoy_w["ratio"] = yoy_w[TARGET] / np.where(yoy_w["pred"] > 0.5, yoy_w["pred"], 0.5)

    cf_yoy = (
        yoy_w.groupby(BAKERY_COL)["ratio"]
        .apply(winsor_mean).rename("cf").reset_index()
    )
    cf_yoy["cf"] = cf_yoy["cf"].clip(0.5, 2.0)
    obs_yoy = yoy_w.groupby(BAKERY_COL)["Дата"].nunique().rename("obs").reset_index()
    cf_yoy = cf_yoy.merge(obs_yoy, on=BAKERY_COL, how="left")

    p(f"\n      YoY CF distribution:")
    p(f"        median={cf_yoy['cf'].median():.3f}  mean={cf_yoy['cf'].mean():.3f}"
      f"  p10={cf_yoy['cf'].quantile(0.1):.3f}  p90={cf_yoy['cf'].quantile(0.9):.3f}")
    p(f"        CF>1.1: {(cf_yoy['cf']>1.1).sum()}  CF<0.9: {(cf_yoy['cf']<0.9).sum()}  total: {len(cf_yoy)}")

    # Variant A1: raw YoY CF
    t_a1 = test.merge(cf_yoy[[BAKERY_COL, "cf"]], on=BAKERY_COL, how="left")
    t_a1["cf"] = t_a1["cf"].fillna(1.0)
    pred_a1 = np.clip(test_pred * t_a1["cf"].values, 0, None)
    mae_a1  = mean_absolute_error(actual_test, pred_a1)
    bias_a1 = float(np.mean(pred_a1 - actual_test))
    p(f"\n      A1_yoy_cf_raw:    MAE={mae_a1:.4f}  Bias={bias_a1:+.3f}  delta={mae_a1-mae_base:+.4f} {'✓' if mae_a1<mae_base else '✗'}")

    # Variant A2: soft blend (alpha by obs count, max 28 days in yoy month)
    cf_yoy2 = cf_yoy.copy()
    cf_yoy2["alpha"] = np.minimum(0.8, cf_yoy2["obs"] / 28)
    cf_yoy2["cf_b"]  = cf_yoy2["alpha"] * cf_yoy2["cf"] + (1 - cf_yoy2["alpha"]) * 1.0
    t_a2 = test.merge(cf_yoy2[[BAKERY_COL, "cf_b"]], on=BAKERY_COL, how="left")
    t_a2["cf_b"] = t_a2["cf_b"].fillna(1.0)
    pred_a2 = np.clip(test_pred * t_a2["cf_b"].values, 0, None)
    mae_a2  = mean_absolute_error(actual_test, pred_a2)
    bias_a2 = float(np.mean(pred_a2 - actual_test))
    p(f"      A2_yoy_cf_blend:  MAE={mae_a2:.4f}  Bias={bias_a2:+.3f}  delta={mae_a2-mae_base:+.4f} {'✓' if mae_a2<mae_base else '✗'}")

    # ── Part B: model WITH sales_lag365 ────────────────────────────────────
    p("\n  [B] Adding sales_lag365 feature...")
    FEATURES_YOY = FEATURES_BASE + ["sales_lag365"]
    train_b = add_lag365(df, train)
    test_b  = add_lag365(df, test)

    lag365_coverage = test_b["sales_lag365"].notna().mean()
    p(f"      lag365 coverage in test: {lag365_coverage:.1%}")

    t0 = time.time()
    model_yoy = LGBMRegressor(**MODEL_PARAMS)
    model_yoy.fit(prep_X(train_b, FEATURES_YOY), train_b[TARGET],
                  categorical_feature=CATEGORICAL_COLS)
    p(f"      trained in {time.time()-t0:.1f}s")

    pred_b = np.clip(model_yoy.predict(prep_X(test_b, FEATURES_YOY)), 0, None)
    mae_b  = mean_absolute_error(actual_test, pred_b)
    bias_b = float(np.mean(pred_b - actual_test))
    p(f"      B_model_lag365:   MAE={mae_b:.4f}  Bias={bias_b:+.3f}  delta={mae_b-mae_base:+.4f} {'✓' if mae_b<mae_base else '✗'}")

    # Feature importance for lag365
    imp = pd.Series(model_yoy.booster_.feature_importance(importance_type="gain"),
                    index=FEATURES_YOY)
    imp_pct = imp / imp.sum() * 100
    top5 = imp_pct.nlargest(5)
    lag365_pct = float(imp_pct.get("sales_lag365", 0))
    p(f"\n      sales_lag365 importance (gain): {lag365_pct:.2f}%")
    p(f"      Top-5 features:")
    for feat, val in top5.items():
        marker = " <<" if feat == "sales_lag365" else ""
        p(f"        {feat:<35} {val:.2f}%{marker}")

    importances[fold["name"]] = {f: round(float(imp_pct[f]), 3) for f in FEATURES_YOY}

    fold_res = {
        "0_baseline":      {"mae": round(mae_base, 4), "bias": round(bias_base, 4)},
        "A1_yoy_cf_raw":   {"mae": round(mae_a1,   4), "bias": round(bias_a1,   4), "delta": round(mae_a1-mae_base, 4)},
        "A2_yoy_cf_blend": {"mae": round(mae_a2,   4), "bias": round(bias_a2,   4), "delta": round(mae_a2-mae_base, 4)},
        "B_model_lag365":  {"mae": round(mae_b,    4), "bias": round(bias_b,    4), "delta": round(mae_b-mae_base,  4)},
    }
    all_results[fold["name"]] = fold_res

# ── Aggregate ─────────────────────────────────────────────────────────────────
p(f"\n{'='*70}")
p("AGGREGATE across 3 folds (mean MAE)")
p(f"{'='*70}")
variants = ["0_baseline", "A1_yoy_cf_raw", "A2_yoy_cf_blend", "B_model_lag365"]
for v in variants:
    maes   = [all_results[f["name"]][v]["mae"]  for f in FOLDS]
    biases = [all_results[f["name"]][v]["bias"] for f in FOLDS]
    mean_mae  = np.mean(maes)
    mean_bias = np.mean(biases)
    base_mean = np.mean([all_results[f["name"]]["0_baseline"]["mae"] for f in FOLDS])
    delta = mean_mae - base_mean
    sign = "✓" if delta < -0.001 else ("=" if abs(delta) <= 0.001 else "✗")
    p(f"  {v:<30} MAE={mean_mae:.4f}  Bias={mean_bias:+.3f}  delta={delta:+.4f} {sign}")

p(f"\n  avg lag365 importance across folds:")
for fold in FOLDS:
    val = importances[fold["name"]].get("sales_lag365", 0)
    p(f"    {fold['name']}: {val:.2f}%")

# Save
EXP_DIR.mkdir(parents=True, exist_ok=True)
with open(EXP_DIR / "yoy_metrics.json", "w", encoding="utf-8") as f:
    json.dump({"results": all_results, "importances": importances}, f, ensure_ascii=False, indent=2)
p(f"\nSaved -> {EXP_DIR / 'yoy_metrics.json'}")
