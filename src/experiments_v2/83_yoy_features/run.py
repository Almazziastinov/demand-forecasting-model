"""
Exp 83: YoY feature enrichment for bakery-day forecasts.

Problem: lag365 shifts DOW (Tue 2026 -> Mon 2025), contaminating the seasonal signal
with day-of-week noise.

Three new features, on top of existing lag365:
  lag364        -- shift 364d = 52 exact weeks, always same DOW
  roll_mean4w_yoy -- mean of [date-378 .. date-350]: 4-week window centered at date-364
  yoy_month_mean  -- mean of same (bakery, month) in prior year

CV structure (same as exp 82 YoY):
  Fold 1: train Feb 2025-Mar 2026 | test Apr 2026
  Fold 2: train Mar 2025-Apr 2026 | test May 2026
  Fold 3: train Apr 2025-May 2026 | test Jun 2026
"""

import sys, json, time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

out = sys.stdout.buffer
def p(s): out.write((s + "\n").encode("utf-8")); out.flush()

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent

DATASET_PATH = ROOT / "data/processed/stg_daily_v1/bakery_daily_sales.csv"

BAKERY_ID_COL = "bakery_id"
DATE_COL      = "date"
TARGET_COL    = "bakery_sales"

BASE_FEATURES = [
    "bakery_id", "city",
    "dow", "day", "month", "iso_week", "is_weekend",
    "is_month_start", "is_month_end", "is_payday_week",
    "bakery_sales_lag1", "bakery_sales_lag2", "bakery_sales_lag3",
    "bakery_sales_lag7", "bakery_sales_lag14", "bakery_sales_lag30",
    "bakery_sales_lag365",
    "bakery_sales_roll_mean3", "bakery_sales_roll_mean7",
    "bakery_sales_roll_mean14", "bakery_sales_roll_mean30",
    "bakery_sales_roll_std7",
]
CATEGORICAL_COLS = ["bakery_id", "city", "month"]

YOY_NEW = ["lag364", "roll_mean4w_yoy", "yoy_month_mean"]

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


# ── YoY feature computation ────────────────────────────────────────────────────

def add_yoy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag364, roll_mean4w_yoy, yoy_month_mean to a sorted bakery panel."""
    work = df.sort_values([BAKERY_ID_COL, DATE_COL]).copy()
    grp = work.groupby(BAKERY_ID_COL)[TARGET_COL]

    # lag364: same DOW, 52 weeks back
    work["lag364"] = grp.shift(364)

    # roll_mean4w_yoy: mean of window [date-378 .. date-350]
    # = shift(350) then rolling(29) — window covers 29 days ending 350d ago
    # centering: midpoint at shift 364 ✓
    work["roll_mean4w_yoy"] = grp.transform(
        lambda x: x.shift(350).rolling(window=29, min_periods=10).mean()
    )

    # yoy_month_mean: mean of bakery sales in same month, prior year
    # Computed from observed data only; merged by (bakery_id, year-1, month)
    tmp = work[[BAKERY_ID_COL, DATE_COL, TARGET_COL]].copy()
    tmp["year"]  = tmp[DATE_COL].dt.year
    tmp["month"] = tmp[DATE_COL].dt.month
    month_means = (
        tmp.groupby([BAKERY_ID_COL, "year", "month"])[TARGET_COL]
        .mean()
        .reset_index()
        .rename(columns={TARGET_COL: "yoy_month_mean", "year": "prev_year"})
    )
    work["year"]  = work[DATE_COL].dt.year
    work["month"] = work[DATE_COL].dt.month
    work["prev_year"] = work["year"] - 1
    work = work.merge(
        month_means,
        left_on=[BAKERY_ID_COL, "prev_year", "month"],
        right_on=[BAKERY_ID_COL, "prev_year", "month"],
        how="left",
    )
    work = work.drop(columns=["year", "prev_year", "month"])
    # restore month column (it was in work before the merge)
    work["month"] = work[DATE_COL].dt.month

    # fill NaN with lag365 as fallback
    for col in ["lag364", "roll_mean4w_yoy", "yoy_month_mean"]:
        work[col] = work[col].fillna(work.get("bakery_sales_lag365", np.nan))

    return work


def prep_X(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = df[features].copy()
    for col in CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X


# ── Load & enrich ──────────────────────────────────────────────────────────────

p("Loading dataset...")
df = pd.read_csv(DATASET_PATH, encoding="utf-8-sig", low_memory=False)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
p(f"  {len(df):,} rows  {df[DATE_COL].min().date()} .. {df[DATE_COL].max().date()}")

p("Adding YoY features (lag364, roll_mean4w_yoy, yoy_month_mean)...")
df = add_yoy_features(df)

# coverage check
for col in YOY_NEW:
    cov = df[col].notna().mean()
    p(f"  {col} coverage: {cov:.1%}")

all_results = {}
importances = {}

VARIANTS = {
    "A_baseline_lag365":     BASE_FEATURES,
    "B_+lag364":             BASE_FEATURES + ["lag364"],
    "C_+roll_mean4w_yoy":    BASE_FEATURES + ["roll_mean4w_yoy"],
    "D_+yoy_month_mean":     BASE_FEATURES + ["yoy_month_mean"],
    "E_+all_three":          BASE_FEATURES + YOY_NEW,
}

for fold in FOLDS:
    p(f"\n{'='*70}")
    p(f"FOLD: {fold['name']}  train {fold['train_from']}..{fold['train_to']}"
      f"  test {fold['test_from']}..{fold['test_to']}")
    p(f"{'='*70}")

    train = df[(df[DATE_COL] >= fold["train_from"]) & (df[DATE_COL] <= fold["train_to"])].copy()
    test  = df[(df[DATE_COL] >= fold["test_from"])  & (df[DATE_COL] <= fold["test_to"])].copy()
    actual = pd.to_numeric(test[TARGET_COL], errors="coerce").values

    p(f"  train: {len(train):,}  test: {len(test):,} rows")

    fold_res = {}
    fold_imp = {}

    for vname, features in VARIANTS.items():
        t0 = time.time()
        model = LGBMRegressor(**MODEL_PARAMS)
        model.fit(prep_X(train, features), train[TARGET_COL],
                  categorical_feature=CATEGORICAL_COLS)
        elapsed = time.time() - t0

        pred = np.clip(model.predict(prep_X(test, features)), 0, None)
        mae  = mean_absolute_error(actual, pred)
        bias = float(np.mean(pred - actual))

        base_mae = fold_res.get("A_baseline_lag365", {}).get("mae", mae)
        delta = mae - base_mae
        sign = "✓" if delta < -0.001 else ("=" if abs(delta) <= 0.001 else "✗")
        p(f"  {vname:<28} MAE={mae:.4f}  Bias={bias:+.3f}  delta={delta:+.4f} {sign}  [{elapsed:.0f}s]")

        fold_res[vname] = {"mae": round(mae, 4), "bias": round(bias, 4), "delta": round(delta, 4)}

        # importance for new features only
        imp = pd.Series(model.booster_.feature_importance(importance_type="gain"), index=features)
        imp_pct = imp / imp.sum() * 100
        fold_imp[vname] = {f: round(float(imp_pct.get(f, 0)), 3) for f in YOY_NEW if f in features}

    all_results[fold["name"]] = fold_res
    importances[fold["name"]] = fold_imp

# ── Aggregate ──────────────────────────────────────────────────────────────────
p(f"\n{'='*70}")
p("AGGREGATE across 3 folds (mean MAE)")
p(f"{'='*70}")
for vname in VARIANTS:
    maes   = [all_results[f["name"]][vname]["mae"]  for f in FOLDS]
    biases = [all_results[f["name"]][vname]["bias"] for f in FOLDS]
    base_maes = [all_results[f["name"]]["A_baseline_lag365"]["mae"] for f in FOLDS]
    mean_mae  = np.mean(maes)
    mean_bias = np.mean(biases)
    delta = mean_mae - np.mean(base_maes)
    sign = "✓" if delta < -0.001 else ("=" if abs(delta) <= 0.001 else "✗")
    p(f"  {vname:<28} MAE={mean_mae:.4f}  Bias={mean_bias:+.3f}  delta={delta:+.4f} {sign}")

p(f"\nYoY feature importances по фолдам (gain %):")
for fold in FOLDS:
    p(f"  {fold['name']}:")
    for vname in ["B_+lag364", "C_+roll_mean4w_yoy", "D_+yoy_month_mean", "E_+all_three"]:
        if vname in importances[fold["name"]]:
            vals = importances[fold["name"]][vname]
            parts = "  ".join(f"{k}={v:.2f}%" for k, v in vals.items())
            p(f"    {vname:<28} {parts}")

EXP_DIR.mkdir(parents=True, exist_ok=True)
with open(EXP_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump({"results": all_results, "importances": importances}, f, ensure_ascii=False, indent=2)
p(f"\nSaved -> {EXP_DIR / 'metrics.json'}")
