from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.experiments_v2.bakery_day_forecast import DATE_COL
from src.experiments_v2.bakery_day_forecast import TARGET_COL
from src.experiments_v2.bakery_day_forecast import build_model_frame
from src.experiments_v2.bakery_day_forecast import load_dataset


DEFAULT_DATASET_PATH = ROOT / "data" / "processed" / "bakery_daily_sales.csv"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "holiday_behavior"
DEFAULT_PRE_DAYS = 10
DEFAULT_POST_DAYS = 5


FIXED_HOLIDAY_LABELS = {
    "01-01": "new_year_day",
    "01-02": "new_year_holiday_2",
    "01-03": "new_year_holiday_3",
    "01-04": "new_year_holiday_4",
    "01-05": "new_year_holiday_5",
    "01-06": "new_year_holiday_6",
    "01-07": "christmas",
    "01-08": "new_year_holiday_8",
    "02-23": "defender_day",
    "03-08": "womens_day",
    "05-01": "spring_labor_day",
    "05-09": "victory_day",
    "06-12": "russia_day",
    "08-30": "tatarstan_day",
    "11-04": "unity_day",
    "11-06": "constitution_day_rt",
}


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def canonical_holiday_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        return ""
    return FIXED_HOLIDAY_LABELS.get(name, name)


def build_network_daily(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby(DATE_COL, as_index=False)
        .agg(
            actual_sales=(TARGET_COL, "sum"),
            n_bakeries=("bakery_id", "nunique"),
            holiday_name=("holiday_name", lambda s: next((x for x in s if x), "")),
            is_holiday=("is_holiday", "max"),
            dow=("dow", "first"),
        )
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )
    daily["holiday_name"] = daily["holiday_name"].map(canonical_holiday_name)
    daily["same_dow_baseline"] = (
        daily.groupby("dow", sort=False)["actual_sales"]
        .transform(lambda s: s.shift(1).rolling(8, min_periods=4).mean())
    )
    daily["rolling_baseline_14d"] = daily["actual_sales"].shift(1).rolling(14, min_periods=7).mean()
    daily["baseline_sales"] = daily["same_dow_baseline"].fillna(daily["rolling_baseline_14d"])
    daily["baseline_sales"] = daily["baseline_sales"].fillna(daily["actual_sales"].expanding().mean())
    daily["pct_vs_baseline"] = (daily["actual_sales"] / (daily["baseline_sales"] + 1e-8) - 1.0) * 100.0
    daily["abs_delta_vs_baseline"] = daily["actual_sales"] - daily["baseline_sales"]
    return daily


def build_event_windows(network_daily: pd.DataFrame, pre_days: int, post_days: int) -> pd.DataFrame:
    holiday_dates = network_daily[network_daily["is_holiday"] == 1][[DATE_COL, "holiday_name"]].copy()
    rows: list[dict] = []

    for event_idx, event_row in holiday_dates.reset_index(drop=True).iterrows():
        event_date = pd.Timestamp(event_row[DATE_COL])
        holiday_name = event_row["holiday_name"]
        window = network_daily[
            (network_daily[DATE_COL] >= event_date - pd.Timedelta(days=pre_days))
            & (network_daily[DATE_COL] <= event_date + pd.Timedelta(days=post_days))
        ].copy()
        if window.empty:
            continue
        window["event_id"] = f"{event_date.date()}__{holiday_name}"
        window["event_date"] = event_date
        window["holiday_name"] = holiday_name
        window["day_offset"] = (window[DATE_COL] - event_date).dt.days
        rows.append(window)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_event_occurrences(event_windows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for event_id, group in event_windows.groupby("event_id", sort=False):
        group = group.sort_values("day_offset").copy()
        holiday_name = group["holiday_name"].iloc[0]
        event_date = pd.Timestamp(group["event_date"].iloc[0])
        peak_row = group.loc[group["pct_vs_baseline"].idxmax()]
        trough_row = group.loc[group["pct_vs_baseline"].idxmin()]
        pre = group[group["day_offset"] < 0]
        post = group[group["day_offset"] > 0]
        rows.append(
            {
                "event_id": event_id,
                "holiday_name": holiday_name,
                "event_date": str(event_date.date()),
                "window_days": int(len(group)),
                "event_day_pct_vs_baseline": float(group.loc[group["day_offset"] == 0, "pct_vs_baseline"].mean()),
                "pre_window_mean_pct": float(pre["pct_vs_baseline"].mean()) if not pre.empty else 0.0,
                "post_window_mean_pct": float(post["pct_vs_baseline"].mean()) if not post.empty else 0.0,
                "peak_offset": int(peak_row["day_offset"]),
                "peak_pct": float(peak_row["pct_vs_baseline"]),
                "trough_offset": int(trough_row["day_offset"]),
                "trough_pct": float(trough_row["pct_vs_baseline"]),
                "pre_positive_days": int((pre["pct_vs_baseline"] > 0).sum()),
                "post_positive_days": int((post["pct_vs_baseline"] > 0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["holiday_name", "event_date"]).reset_index(drop=True)


def build_event_profile_matrix(event_windows: pd.DataFrame) -> pd.DataFrame:
    profile = (
        event_windows.groupby(["holiday_name", "day_offset"], as_index=False)
        .agg(
            mean_pct_vs_baseline=("pct_vs_baseline", "mean"),
            median_pct_vs_baseline=("pct_vs_baseline", "median"),
            mean_abs_delta_vs_baseline=("abs_delta_vs_baseline", "mean"),
            n_occurrences=("event_id", "nunique"),
        )
        .sort_values(["holiday_name", "day_offset"])
        .reset_index(drop=True)
    )
    return profile


def cluster_event_profiles(event_profile: pd.DataFrame, n_clusters: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = event_profile.pivot(index="holiday_name", columns="day_offset", values="mean_pct_vs_baseline").fillna(0.0)
    n_samples = len(wide)
    if n_samples == 0:
        return pd.DataFrame(), wide.reset_index()

    k = max(1, min(n_clusters, n_samples))
    scaler = StandardScaler()
    x = scaler.fit_transform(wide.to_numpy(dtype=float))
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(x)

    cluster_df = wide.reset_index()
    cluster_df["cluster_id"] = labels
    cluster_centers = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_),
        columns=wide.columns,
    )
    cluster_centers.insert(0, "cluster_id", range(k))
    return cluster_df.sort_values("cluster_id").reset_index(drop=True), cluster_centers


def build_cluster_summary(cluster_df: pd.DataFrame, event_occurrences: pd.DataFrame) -> pd.DataFrame:
    if cluster_df.empty:
        return pd.DataFrame()
    summary = (
        cluster_df.merge(event_occurrences.groupby("holiday_name", as_index=False).agg(n_events=("event_id", "count")), on="holiday_name", how="left")
        .groupby("cluster_id", as_index=False)
        .agg(
            holidays=("holiday_name", lambda s: ", ".join(sorted(s))),
            n_holidays=("holiday_name", "nunique"),
            total_event_occurrences=("n_events", "sum"),
        )
    )
    return summary.sort_values("cluster_id").reset_index(drop=True)


def plot_event_profiles(event_profile: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for holiday_name, group in event_profile.groupby("holiday_name", sort=False):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axhline(0.0, color="black", linewidth=1)
        ax.plot(group["day_offset"], group["mean_pct_vs_baseline"], marker="o")
        ax.set_title(f"{holiday_name}: mean uplift vs baseline")
        ax.set_xlabel("Day offset")
        ax.set_ylabel("Pct vs baseline")
        plt.tight_layout()
        fig.savefig(output_dir / f"{holiday_name}.png", dpi=150)
        plt.close(fig)


def plot_cluster_centers(cluster_centers: pd.DataFrame, output_dir: Path) -> None:
    if cluster_centers.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    x_cols = [col for col in cluster_centers.columns if col != "cluster_id"]
    x = [int(col) for col in x_cols]
    for _, row in cluster_centers.iterrows():
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axhline(0.0, color="black", linewidth=1)
        ax.plot(x, row[x_cols].to_numpy(dtype=float), marker="o")
        ax.set_title(f"Cluster {int(row['cluster_id'])}: mean holiday behavior profile")
        ax.set_xlabel("Day offset")
        ax.set_ylabel("Pct vs baseline")
        plt.tight_layout()
        fig.savefig(output_dir / f"cluster_{int(row['cluster_id'])}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze holiday behavior windows across the full dataset")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--pre-days", type=int, default=DEFAULT_PRE_DAYS)
    parser.add_argument("--post-days", type=int, default=DEFAULT_POST_DAYS)
    parser.add_argument("--n-clusters", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    df = build_model_frame(load_dataset(args.dataset_path))
    network_daily = build_network_daily(df)
    event_windows = build_event_windows(network_daily, pre_days=args.pre_days, post_days=args.post_days)
    event_occurrences = summarize_event_occurrences(event_windows)
    event_profile = build_event_profile_matrix(event_windows)
    cluster_df, cluster_centers = cluster_event_profiles(event_profile, n_clusters=args.n_clusters)
    cluster_summary = build_cluster_summary(cluster_df, event_occurrences)

    save_csv(network_daily, output_dir / "network_daily.csv")
    save_csv(event_windows, output_dir / "event_windows.csv")
    save_csv(event_occurrences, output_dir / "event_occurrences.csv")
    save_csv(event_profile, output_dir / "event_profile_by_holiday.csv")
    save_csv(cluster_df, output_dir / "event_profile_clusters.csv")
    save_csv(cluster_centers, output_dir / "cluster_centers.csv")
    save_csv(cluster_summary, output_dir / "cluster_summary.csv")

    plot_event_profiles(event_profile, output_dir / "plots" / "events")
    plot_cluster_centers(cluster_centers, output_dir / "plots" / "clusters")

    overview = {
        "dataset_path": str(args.dataset_path),
        "date_min": str(network_daily[DATE_COL].min().date()),
        "date_max": str(network_daily[DATE_COL].max().date()),
        "n_network_days": int(len(network_daily)),
        "n_holiday_occurrences": int(event_occurrences["event_id"].nunique()) if not event_occurrences.empty else 0,
        "n_unique_holidays": int(event_occurrences["holiday_name"].nunique()) if not event_occurrences.empty else 0,
        "pre_days": args.pre_days,
        "post_days": args.post_days,
        "n_clusters": int(cluster_summary["cluster_id"].nunique()) if not cluster_summary.empty else 0,
    }
    (output_dir / "overview.json").write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {output_dir}")


if __name__ == "__main__":
    main()
