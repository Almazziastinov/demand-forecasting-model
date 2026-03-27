"""
Experiment I: Combined Best
Reads summary CSVs from F, G, H to pick winners, then applies all
winning transforms together on a single model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd

from src.experiments.common import (
    load_data, FEATURES, CATEGORICAL_COLS, TARGET, MODEL_PARAMS, BASELINE_MAE,
    DATA_PATH,
    wmape, print_metrics, print_category_metrics,
    train_lgbm, predict_clipped, save_predictions,
)


def read_best_strategy(summary_path, all_results_path):
    """Read best strategy from experiment results. Returns (name, mae, delta) or None."""
    if os.path.exists(all_results_path):
        df = pd.read_csv(all_results_path)
        best = df.sort_values("mae").iloc[0]
        return best["strategy"], best["mae"], best["delta"]
    elif os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        row = df.iloc[0]
        return row["experiment"], row["mae"], row["delta"]
    return None, None, None


def main():
    print("=" * 65)
    print("  EXPERIMENT I: COMBINED BEST (F + G + H)")
    print("=" * 65)

    # --- Read winners from F, G, H ---
    print("\n--- Chtenie rezul'tatov eksperimentov F, G, H ---")

    f_name, f_mae, f_delta = read_best_strategy(
        "reports/exp_f_summary.csv", "reports/exp_f_all_results.csv")
    g_name, g_mae, g_delta = read_best_strategy(
        "reports/exp_g_summary.csv", "reports/exp_g_all_results.csv")
    h_name, h_mae, h_delta = read_best_strategy(
        "reports/exp_h_summary.csv", "reports/exp_h_all_results.csv")

    print(f"  F best: {f_name} (MAE={f_mae}, delta={f_delta})")
    print(f"  G best: {g_name} (MAE={g_mae}, delta={g_delta})")
    print(f"  H best: {h_name} (MAE={h_mae}, delta={h_delta})")

    # Decide which transforms to apply (only if they improved over baseline)
    use_f = f_delta is not None and f_delta < 0
    use_g = g_delta is not None and g_delta < 0
    use_h = h_delta is not None and h_delta < 0

    print(f"\n  Primenyaem F (outlier clipping): {'DA' if use_f else 'NET'}")
    print(f"  Primenyaem G (noise filtering):   {'DA' if use_g else 'NET'}")
    print(f"  Primenyaem H (censored demand):   {'DA' if use_h else 'NET'}")

    if not (use_f or use_g or use_h):
        print("\n  Ni odin eksperiment ne uluchshil baseline.")
        print("  Zapuskayem baseline dlya sravneniya...")

    # --- Load data ---
    # Determine if we need median rolling (G2)
    use_median_rolling = use_g and g_name is not None and "G2" in str(g_name)
    use_false_zero_filter = use_g and g_name is not None and ("G1" in str(g_name) or "G3" in str(g_name))

    # If G3 won, both G1 and G2 apply
    if use_g and g_name is not None and "G3" in str(g_name):
        use_median_rolling = True
        use_false_zero_filter = True

    if use_median_rolling:
        print("\n--- Zagruzka s median rolling ---")
        df = pd.read_csv(DATA_PATH)
        df["Дата"] = pd.to_datetime(df["Дата"])

        grouped = df.groupby(["Пекарня", "Номенклатура"])
        df["sales_roll_median3"] = grouped["Продано"].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).median()
        )
        df["sales_roll_median7"] = grouped["Продано"].transform(
            lambda x: x.shift(1).rolling(7, min_periods=1).median()
        )

        features = []
        for f in FEATURES:
            if f == "sales_roll_mean3":
                features.append("sales_roll_median3")
            elif f == "sales_roll_mean7":
                features.append("sales_roll_median7")
            else:
                features.append(f)

        for col in CATEGORICAL_COLS:
            if col in df.columns:
                df[col] = df[col].astype("category")

        features = [f for f in features if f in df.columns]
    else:
        df, _, _, _, _, features = load_data()

    # Split
    test_start = df["Дата"].max() - pd.Timedelta(days=2)
    train_full = df[df["Дата"] < test_start].copy()
    test = df[df["Дата"] >= test_start].copy()

    print(f"  Train initial: {len(train_full):,} strok")
    print(f"  Test: {len(test):,} strok")

    # --- Apply transforms to train ---
    train_working = train_full.copy()
    y_col = TARGET

    # G1: Remove false zeros
    if use_false_zero_filter:
        false_zero_mask = (train_working["Продано"] == 0) & (train_working["Выпуск"] == 0)
        n_removed = false_zero_mask.sum()
        train_working = train_working[~false_zero_mask]
        print(f"  G1: Udaleno lozhnykh nulej: {n_removed:,}")

    # F: Outlier clipping
    if use_f and f_name is not None:
        y_vals = train_working[TARGET].values.copy().astype(float)
        if "p99" in str(f_name):
            threshold = np.percentile(y_vals, 99)
            y_vals = np.clip(y_vals, 0, threshold)
            print(f"  F: Clip at p99 ({threshold:.1f})")
        elif "per_cat" in str(f_name):
            cats = train_working["Категория"].values
            tmp_df = pd.DataFrame({"y": y_vals, "cat": cats})
            cat_p95 = tmp_df.groupby("cat")["y"].quantile(0.95).to_dict()
            y_vals = np.array([min(v, cat_p95.get(c, np.percentile(y_vals, 95)))
                               for v, c in zip(y_vals, cats)])
            print(f"  F: Clip per-category p95")
        else:  # p95 or winsorize_p95
            threshold = np.percentile(y_vals, 95)
            y_vals = np.clip(y_vals, 0, threshold)
            print(f"  F: Clip at p95 ({threshold:.1f})")
        train_working["__target_adj"] = y_vals
        y_col = "__target_adj"

    # H: Censored demand correction
    if use_h and h_name is not None:
        if y_col == "__target_adj":
            y_vals = train_working[y_col].values.copy()
        else:
            y_vals = train_working[TARGET].values.copy().astype(float)

        censored_mask = train_working["Остаток"].values == 0

        if "exclude" in str(h_name):
            # H2: remove censored rows
            keep_mask = ~censored_mask
            train_working = train_working[keep_mask]
            if y_col == "__target_adj":
                # y_col column already in train_working
                pass
            n_removed = (~keep_mask).sum()
            print(f"  H2: Udaleno tsenzurirovannykh: {n_removed:,}")
        elif "deficit" in str(h_name):
            # H3: deficit correction
            deficit = train_working["stock_deficit"].values
            y_vals[censored_mask] = y_vals[censored_mask] + \
                np.nan_to_num(deficit[censored_mask], nan=0.0) * 0.5
            train_working["__target_adj"] = y_vals
            y_col = "__target_adj"
            print(f"  H3: Deficit correction applied")
        else:
            # H1: demand_correction (max of actual vs rolling mean7)
            roll_mean7 = train_working["sales_roll_mean7"].values if "sales_roll_mean7" in train_working.columns \
                else np.zeros(len(train_working))
            y_vals[censored_mask] = np.maximum(
                y_vals[censored_mask],
                np.nan_to_num(roll_mean7[censored_mask], nan=0.0)
            )
            train_working["__target_adj"] = y_vals
            y_col = "__target_adj"
            print(f"  H1: Demand correction applied")

    # --- Prepare final train ---
    X_train_final = train_working[features]
    if y_col == "__target_adj":
        y_train_final = train_working["__target_adj"].values
    else:
        y_train_final = train_working[TARGET].values

    X_test_final = test[features]
    y_test_final = test[TARGET].values

    print(f"\n  Final train: {len(X_train_final):,} strok")

    # --- Baseline for comparison ---
    print("\n--- Baseline (no transforms) ---")
    df_bl, X_tr_bl, y_tr_bl, X_te_bl, y_te_bl, feat_bl = load_data()
    bl_model = train_lgbm(X_tr_bl, y_tr_bl)
    bl_pred = predict_clipped(bl_model, X_te_bl)
    bl_mae, _, _ = print_metrics("Baseline", y_te_bl, bl_pred)

    # --- Combined model ---
    print("\n--- Combined model (F+G+H) ---")
    model_comb = train_lgbm(X_train_final, y_train_final)
    pred_comb = predict_clipped(model_comb, X_test_final)

    mae_comb, wm_comb, bias_comb = print_metrics("I_combined", y_test_final, pred_comb)
    print_category_metrics(y_test_final, pred_comb, test["Категория"])

    # --- Summary ---
    print("\n" + "=" * 65)
    print("  ITOGI EKSPERIMENTA I")
    print("=" * 65)

    transforms_used = []
    if use_f:
        transforms_used.append(f"F({f_name})")
    if use_g:
        transforms_used.append(f"G({g_name})")
    if use_h:
        transforms_used.append(f"H({h_name})")
    transforms_str = " + ".join(transforms_used) if transforms_used else "none"

    delta = mae_comb - BASELINE_MAE
    print(f"\n  Transforms: {transforms_str}")
    print(f"  MAE = {mae_comb:.4f} (delta vs baseline: {delta:+.4f})")

    if delta < 0:
        improvement_pct = abs(delta) / BASELINE_MAE * 100
        print(f"  Uluchshenie: {improvement_pct:.2f}% vs baseline")
    else:
        print(f"  Baseline ostaetsya luchshim")

    # Compare with individual bests
    print(f"\n  Sravnenie:")
    print(f"    Baseline:  {BASELINE_MAE:.4f}")
    if f_mae is not None:
        print(f"    F best:    {f_mae:.4f} ({f_delta:+.4f})")
    if g_mae is not None:
        print(f"    G best:    {g_mae:.4f} ({g_delta:+.4f})")
    if h_mae is not None:
        print(f"    H best:    {h_mae:.4f} ({h_delta:+.4f})")
    print(f"    Combined:  {mae_comb:.4f} ({delta:+.4f})")

    # Save
    save_predictions(X_test_final, y_test_final, pred_comb, "reports/exp_i_predictions.csv")

    summary = pd.DataFrame([{
        "experiment": f"I_combined({transforms_str})",
        "mae": mae_comb,
        "wmape": wm_comb,
        "bias": bias_comb,
        "baseline_mae": BASELINE_MAE,
        "delta": delta,
    }])
    summary.to_csv("reports/exp_i_summary.csv", index=False)

    print(f"\n  Summary: reports/exp_i_summary.csv")
    print("\nGotovo!")


if __name__ == "__main__":
    main()
