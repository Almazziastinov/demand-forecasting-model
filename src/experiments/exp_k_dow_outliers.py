"""
Experiment K: DOW-adjusted local outlier treatment.

Idea: for each bakery-product pair, compute expected sales using
      bakery-product mean * product-level DOW ratio. Flag points where
      actual sales deviate significantly from expected (without external
      causes like holidays). Winsorize those points to boundary values.

Strategies:
  K1: Winsorize at 2*std from expected (product DOW ratio method)
  K2: Winsorize at 2.5*std
  K3: Winsorize at 3*std
  K4: Replace with expected value (aggressive)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from experiments.common import (
    load_data, FEATURES, TARGET, CATEGORICAL_COLS, MODEL_PARAMS,
    TEST_DAYS, train_lgbm, predict_clipped, print_metrics,
    print_category_metrics,
)


def compute_dow_expected(df):
    """Compute DOW-adjusted expected sales for each row.

    Method: expected = bakery_product_mean * product_dow_ratio
    where product_dow_ratio = product_dow_mean / product_overall_mean
    """
    product_col = "Номенклатура"
    bakery_col = "Пекарня"
    dow_col = "ДеньНедели"

    # Product-level DOW ratios (robust, computed across all bakeries)
    prod_overall = df.groupby(product_col)[TARGET].mean()
    prod_dow = df.groupby([product_col, dow_col])[TARGET].mean()

    # dow_ratio = product_dow_mean / product_overall_mean
    dow_ratio = prod_dow / prod_overall
    dow_ratio = dow_ratio.rename("dow_ratio")

    # Bakery-product mean
    bp_mean = df.groupby([bakery_col, product_col])[TARGET].mean()
    bp_mean = bp_mean.rename("bp_mean")

    # Bakery-product std
    bp_std = df.groupby([bakery_col, product_col])[TARGET].std()
    bp_std = bp_std.rename("bp_std")

    # Merge
    df2 = df.merge(dow_ratio, left_on=[product_col, dow_col], right_index=True, how="left")
    df2 = df2.merge(bp_mean, left_on=[bakery_col, product_col], right_index=True, how="left")
    df2 = df2.merge(bp_std, left_on=[bakery_col, product_col], right_index=True, how="left")

    df2["expected"] = df2["bp_mean"] * df2["dow_ratio"]
    df2["residual"] = df2[TARGET] - df2["expected"]

    return df2


def winsorize_outliers(df, train_mask, threshold_std=2.0,
                       replace_with_expected=False, categories=None):
    """Winsorize outliers in training data only.

    Outlier = |residual| > threshold_std * bp_std, excluding holidays.
    categories: if set, only apply to these categories (list of strings).
    """
    df2 = df.copy()

    has_external = (
        (df2["is_holiday"] == 1) |
        (df2["is_pre_holiday"] == 1) |
        (df2["is_post_holiday"] == 1)
    )

    bp_std_safe = df2["bp_std"].clip(lower=1)
    is_outlier = (
        train_mask &
        ~has_external &
        (df2["residual"].abs() > threshold_std * bp_std_safe)
    )

    if categories is not None:
        cat_mask = df2["Категория"].isin(categories)
        is_outlier = is_outlier & cat_mask
        print(f"    (tolko kategorii: {categories})")

    n_outliers = is_outlier.sum()
    n_train = train_mask.sum()

    if replace_with_expected:
        df2.loc[is_outlier, TARGET] = df2.loc[is_outlier, "expected"].clip(lower=0).round()
    else:
        upper = df2["expected"] + threshold_std * bp_std_safe
        lower = (df2["expected"] - threshold_std * bp_std_safe).clip(lower=0)

        too_high = is_outlier & (df2[TARGET] > upper)
        too_low = is_outlier & (df2[TARGET] < lower)

        df2.loc[too_high, TARGET] = upper[too_high].round()
        df2.loc[too_low, TARGET] = lower[too_low].round()

    print(f"    Vybrosy: {n_outliers} iz {n_train} ({100*n_outliers/n_train:.2f}%)")
    print(f"    (vyshe ozhidaemogo: {(is_outlier & (df2['residual'] > 0)).sum()}, "
          f"nizhe: {(is_outlier & (df2['residual'] < 0)).sum()})")

    return df2


def run_experiment(strategy_name, df, train_mask, test_mask, features,
                   threshold_std=2.0, replace_with_expected=False,
                   categories=None):
    """Run one strategy: winsorize train, retrain, evaluate on original test."""
    print(f"\n--- {strategy_name} ---")

    df_mod = winsorize_outliers(df, train_mask, threshold_std,
                                replace_with_expected, categories)

    X_train = df_mod.loc[train_mask, features]
    y_train = df_mod.loc[train_mask, TARGET]
    X_test = df.loc[test_mask, features]  # original test!
    y_test = df.loc[test_mask, TARGET]

    model = train_lgbm(X_train, y_train)
    y_pred = predict_clipped(model, X_test)

    mae, wm, bias = print_metrics(strategy_name, y_test, y_pred, baseline_mae=None)

    categories = df.loc[test_mask, "Категория"].values
    print_category_metrics(y_test, y_pred, categories)

    return mae, wm, bias


def main():
    print("=" * 60)
    print("Experiment K: DOW-adjusted local outlier treatment")
    print("=" * 60)

    # Load data
    df, X_train, y_train, X_test, y_test, features = load_data()

    # Compute DOW-expected on full data
    df = compute_dow_expected(df)

    test_start = df["Дата"].max() - pd.Timedelta(days=TEST_DAYS - 1)
    train_mask = df["Дата"] < test_start
    test_mask = df["Дата"] >= test_start

    # 0. Baseline (no changes)
    print("\n--- Baseline (bez izmenenij) ---")
    model = train_lgbm(X_train, y_train)
    y_pred = predict_clipped(model, X_test)
    baseline_mae, _, _ = print_metrics("Baseline", y_test, y_pred)

    categories = df.loc[test_mask, "Категория"].values
    print_category_metrics(y_test, y_pred, categories)

    # Show outlier distribution in train
    train_df = df[train_mask]
    bp_std_safe = train_df["bp_std"].clip(lower=1)
    has_ext = (
        (train_df["is_holiday"] == 1) |
        (train_df["is_pre_holiday"] == 1) |
        (train_df["is_post_holiday"] == 1)
    )
    for thresh in [1.5, 2.0, 2.5, 3.0]:
        n = ((train_df["residual"].abs() > thresh * bp_std_safe) & ~has_ext).sum()
        print(f"    Vybrosy pri threshold={thresh}: {n} ({100*n/len(train_df):.2f}%)")

    results = {}

    # K1: Winsorize at 2*std
    mae, wm, bias = run_experiment(
        "K1: Winsorize 2*std", df, train_mask, test_mask, features,
        threshold_std=2.0)
    results["K1_2std"] = mae

    # K2: Winsorize at 2.5*std
    mae, wm, bias = run_experiment(
        "K2: Winsorize 2.5*std", df, train_mask, test_mask, features,
        threshold_std=2.5)
    results["K2_2.5std"] = mae

    # K3: Winsorize at 3*std
    mae, wm, bias = run_experiment(
        "K3: Winsorize 3*std", df, train_mask, test_mask, features,
        threshold_std=3.0)
    results["K3_3std"] = mae

    # K4: Replace with expected (all categories)
    mae, wm, bias = run_experiment(
        "K4: Replace with expected", df, train_mask, test_mask, features,
        threshold_std=2.0, replace_with_expected=True)
    results["K4_replace"] = mae

    HIGH_DEMAND_CATS = ["Выпечка сытная", "Фастфуд"]

    # K5: Replace with expected, only high-demand categories
    mae, wm, bias = run_experiment(
        "K5: Replace, syt+fast only", df, train_mask, test_mask, features,
        threshold_std=2.0, replace_with_expected=True,
        categories=HIGH_DEMAND_CATS)
    results["K5_replace_high"] = mae

    # K6: Replace with expected, only high-demand, threshold 2.5
    mae, wm, bias = run_experiment(
        "K6: Replace, syt+fast, 2.5std", df, train_mask, test_mask, features,
        threshold_std=2.5, replace_with_expected=True,
        categories=HIGH_DEMAND_CATS)
    results["K6_replace_high_2.5"] = mae

    # K7: Winsorize 2*std, only high-demand categories
    mae, wm, bias = run_experiment(
        "K7: Winsorize 2std, syt+fast", df, train_mask, test_mask, features,
        threshold_std=2.0, replace_with_expected=False,
        categories=HIGH_DEMAND_CATS)
    results["K7_wins_high"] = mae

    # Summary
    print("\n" + "=" * 60)
    print("ITOGI:")
    print(f"  {'Strategiya':<30} {'MAE':>8} {'Delta':>8}")
    print(f"  {'-'*46}")
    print(f"  {'Baseline':<30} {baseline_mae:>8.4f} {'--':>8}")
    for name, mae in sorted(results.items(), key=lambda x: x[1]):
        delta = mae - baseline_mae
        marker = " <-- luchshe" if delta < -0.005 else ""
        print(f"  {name:<30} {mae:>8.4f} {delta:>+8.4f}{marker}")


if __name__ == "__main__":
    main()
