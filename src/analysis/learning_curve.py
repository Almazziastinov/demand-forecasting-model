"""
Learning Curve: MAE vs количество дней обучения.
Фиксированные параметры (v6-best), без тюнинга.
Строит график + power-law экстраполяцию на 6/9/12 месяцев.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

DATA_PATH = "data/processed/preprocessed_data_3month_enriched.csv"

FEATURES = [
    "Пекарня", "Номенклатура", "Категория", "Город",
    "ДеньНедели", "День", "IsWeekend", "Месяц", "НомерНедели",
    "sales_lag1", "sales_lag2", "sales_lag3", "sales_lag7",
    "sales_lag14", "sales_lag30",
    "sales_roll_mean3", "sales_roll_mean7", "sales_roll_std7",
    "sales_roll_mean14", "sales_roll_mean30",
    "stock_lag1", "stock_sales_ratio", "stock_deficit",
    "is_holiday", "is_pre_holiday", "is_post_holiday", "is_payday_week",
    "is_month_start", "is_month_end",
    "temp_mean", "temp_range", "precipitation",
    "is_cold", "is_bad_weather", "weather_cat_code",
]
TARGET = "Продано"

# v6 symmetric best params (fixed, no tuning)
MODEL_PARAMS = {
    "n_estimators": 2304,
    "learning_rate": 0.016,
    "num_leaves": 151,
    "max_depth": 7,
    "min_child_samples": 5,
    "subsample": 0.80,
    "colsample_bytree": 0.67,
    "reg_alpha": 6.63,
    "reg_lambda": 2.2e-6,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(y_true) + 1e-8) * 100


def power_law(x, a, b, c):
    """MAE = a * x^(-b) + c  -- standard learning curve form"""
    return a * np.power(x, -b) + c


def main():
    print("=" * 60)
    print("  LEARNING CURVE: MAE vs TRAINING DAYS")
    print("=" * 60)

    # -- Load data --
    df = pd.read_csv(DATA_PATH)
    df["Дата"] = pd.to_datetime(df["Дата"])

    for col in ["Пекарня", "Номенклатура", "Категория", "Город", "Месяц"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    available = [f for f in FEATURES if f in df.columns]
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"  [!] Missing features: {missing}")

    # Test = last 2 days (same as tune_v6)
    test_start = df["Дата"].max() - pd.Timedelta(days=2)
    test = df[df["Дата"] >= test_start]
    train_all = df[df["Дата"] < test_start]

    X_test = test[available]
    y_test = test[TARGET]

    all_train_dates = sorted(train_all["Дата"].unique())
    total_train_days = len(all_train_dates)

    print(f"  Total train days available: {total_train_days}")
    print(f"  Test: {test['Дата'].min().date()} -- {test['Дата'].max().date()} ({len(test)} rows)")

    # -- Build day points to evaluate --
    # Min 31 days: lag30 + rolling_mean30 need 30 days of history,
    # features are precomputed in CSV so shorter windows are "honest"
    # only if they start AFTER the lag warmup period.
    # But since our CSV already has features baked in from full history,
    # we must start from 31+ to avoid data leakage.
    MIN_DAYS = 31
    day_points = sorted(set(
        list(range(MIN_DAYS, total_train_days + 1, 2)) +
        [total_train_days]
    ))

    print(f"  Points to evaluate: {len(day_points)} ({day_points[0]}..{day_points[-1]} days)")
    print()
    print(f"  {'Days':>6} {'Rows':>8} {'MAE':>8} {'WMAPE':>8}")
    print(f"  {'-' * 34}")

    results = []
    for n_days in day_points:
        cutoff_date = all_train_dates[-n_days]
        train_subset = train_all[train_all["Дата"] >= cutoff_date]

        X_tr = train_subset[available]
        y_tr = train_subset[TARGET]

        model = LGBMRegressor(**MODEL_PARAMS)
        model.fit(X_tr, y_tr)

        pred = np.maximum(model.predict(X_test), 0)
        mae = mean_absolute_error(y_test, pred)
        wm = wmape(y_test.values, pred)

        results.append({"days": n_days, "rows": len(train_subset), "MAE": mae, "WMAPE": wm})
        print(f"  {n_days:>6} {len(train_subset):>8,} {mae:>8.4f} {wm:>7.2f}%")

    res_df = pd.DataFrame(results)

    # -- Fit power law: MAE = a * days^(-b) + c --
    x_data = res_df["days"].values.astype(float)
    y_data = res_df["MAE"].values

    try:
        popt, pcov = curve_fit(
            power_law, x_data, y_data,
            p0=[10.0, 0.3, 3.0],
            bounds=([0, 0.01, 0], [200, 3.0, np.max(y_data)]),
            maxfev=10000,
        )
        a, b, c = popt
        perr = np.sqrt(np.diag(pcov))

        # R-squared of fit
        y_fit = power_law(x_data, *popt)
        ss_res = np.sum((y_data - y_fit) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r2_fit = 1 - ss_res / ss_tot

        print(f"\n  Power-law fit: MAE = {a:.2f} * days^(-{b:.3f}) + {c:.3f}")
        print(f"  R2 of fit: {r2_fit:.4f}")
        print(f"  Asymptote (min MAE): {c:.3f}")
        fit_ok = True
    except Exception as e:
        print(f"\n  [!] Curve fit failed: {e}")
        fit_ok = False

    # -- Extrapolation --
    future = {
        f"Current (~{total_train_days} days)": total_train_days,
        "6 months (~180 days)": 180,
        "9 months (~270 days)": 270,
        "12 months (~365 days)": 365,
    }

    if fit_ok:
        print(f"\n  {'':=<55}")
        print(f"  EXTRAPOLATION")
        print(f"  {'':=<55}")
        print(f"  {'Horizon':<28} {'Est. MAE':>10} {'Delta':>10}")
        print(f"  {'-' * 50}")
        current_mae = power_law(total_train_days, *popt)
        for label, d in future.items():
            est_mae = power_law(d, *popt)
            delta = est_mae - current_mae
            marker = " <-- now" if d == total_train_days else ""
            print(f"  {label:<28} {est_mae:>10.3f} {delta:>+10.3f}{marker}")

        print(f"\n  Theoretical minimum (asymptote): {c:.3f}")
        remaining_gain = current_mae - c
        print(f"  Remaining potential gain: {remaining_gain:.3f} MAE")

    # -- Plot --
    fig, ax = plt.subplots(figsize=(11, 7))

    # Actual data
    ax.scatter(x_data, y_data, color="royalblue", s=60, zorder=5,
               edgecolors="white", linewidth=0.5, label="Measured MAE")
    ax.plot(x_data, y_data, color="royalblue", linewidth=1.2, alpha=0.5)

    if fit_ok:
        # Fit on observed range
        x_fit = np.linspace(x_data.min(), x_data.max(), 300)
        ax.plot(x_fit, power_law(x_fit, *popt),
                color="darkorange", linewidth=2.5, linestyle="--",
                label=f"Power-law fit (R2={r2_fit:.3f})")

        # Extrapolation zone
        x_extra = np.linspace(x_data.max(), 400, 300)
        y_extra = power_law(x_extra, *popt)
        ax.plot(x_extra, y_extra,
                color="red", linewidth=2.5, linestyle=":",
                label="Extrapolation")

        # Confidence band (approximate using parameter uncertainty)
        # Simple approach: +/- on asymptote c
        if perr[2] > 0:
            y_upper = power_law(x_extra, a, b, c + 2 * perr[2])
            y_lower = power_law(x_extra, a, b, max(c - 2 * perr[2], 0))
            ax.fill_between(x_extra, y_lower, y_upper, color="red", alpha=0.08,
                            label="95% CI (approx)")

        # Mark future milestones
        colors_future = ["#2E7D32", "#E65100", "#6A1B9A"]
        idx = 0
        for label, d in future.items():
            if d > total_train_days:
                est_mae = power_law(d, *popt)
                ax.scatter(d, est_mae, color=colors_future[idx % 3],
                           s=150, zorder=6, marker="*", edgecolors="black", linewidth=0.5)
                ax.annotate(
                    f"{label}\nMAE ~ {est_mae:.2f}",
                    xy=(d, est_mae),
                    xytext=(d + 8, est_mae + 0.06),
                    fontsize=9, color=colors_future[idx % 3], fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=colors_future[idx % 3], lw=1.2),
                )
                idx += 1

        # Asymptote line
        ax.axhline(y=c, color="gray", linestyle="-.", alpha=0.5, linewidth=1.5,
                    label=f"Asymptote = {c:.3f}")

    # Mark current point
    ax.axvline(x=total_train_days, color="royalblue", linestyle="--", alpha=0.4, linewidth=1)
    ax.annotate(f"Current\n({total_train_days} days)",
                xy=(total_train_days, y_data[-1]),
                xytext=(total_train_days - 15, y_data[-1] + 0.15),
                fontsize=9, color="royalblue", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="royalblue", lw=1))

    ax.set_xlabel("Training days", fontsize=13)
    ax.set_ylabel("MAE (test set)", fontsize=13)
    ax.set_title("Learning Curve: MAE vs Training Data Volume\n"
                 "How much will more data improve the model?",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0, right=420)

    plt.tight_layout()
    out_path = "reports/learning_curve.png"
    fig.savefig(out_path, dpi=150)
    print(f"\n  Plot saved: {out_path}")
    plt.close()

    # Save raw data
    res_df.to_csv("reports/learning_curve_data.csv", index=False)
    print(f"  Data saved: reports/learning_curve_data.csv")
    print("\nDone!")


if __name__ == "__main__":
    main()
