"""Test causal hierarchical calibration of synthetic lost-demand quantities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.calibrate_post_last_sale_demand import build_cases  # noqa: E402


HOURLY = ROOT / ".codex_tmp/rolling_hourly_sales_20260601_20260823.parquet"
TOP_PRODUCTS = ROOT / "reports/top_loss_product_diagnostic_20260913/top20_product_diagnostic.csv"
OUTPUT = ROOT / "reports/hierarchical_lost_demand_calibration_20260914"
CUTOFFS = [12, 15, 18]
FOLD_STARTS = pd.date_range("2026-06-22", "2026-08-17", freq="7D")
OBSERVED_BINS = [-np.inf, 5, 15, 40, np.inf]
OBSERVED_LABELS = ["<=5", "6-15", "16-40", ">40"]


def add_observed_band(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["observed_band"] = pd.cut(
        result["observed"], bins=OBSERVED_BINS, labels=OBSERVED_LABELS
    ).astype(str)
    return result


def aggregate_multiplier(
    calibration: pd.DataFrame,
    keys: list[str],
    *,
    prior: pd.DataFrame | None = None,
    prior_strength: float = 0.0,
) -> pd.DataFrame:
    grouped = calibration.groupby(keys, as_index=False).agg(
        group_true=("true_hidden", "sum"),
        group_raw=("raw_prediction", "sum"),
        calibration_cases=("date", "size"),
    )
    if prior is None:
        grouped["multiplier"] = grouped["group_true"] / grouped["group_raw"]
        return grouped
    grouped = grouped.merge(prior, on="cutoff", how="left", validate="many_to_one")
    grouped["multiplier"] = (
        grouped["group_true"] + prior_strength * grouped["global_multiplier"]
    ) / (grouped["group_raw"] + prior_strength)
    return grouped


def predict_with_lookup(
    test: pd.DataFrame,
    lookup: pd.DataFrame,
    keys: list[str],
    fallback: pd.DataFrame,
) -> pd.Series:
    columns = keys + ["multiplier"]
    merged = test[keys].merge(lookup[columns], on=keys, how="left", validate="many_to_one")
    merged = merged.merge(fallback, on="cutoff", how="left", validate="many_to_one")
    multiplier = merged["multiplier"].fillna(merged["global_multiplier"])
    return test["raw_prediction"].to_numpy() * multiplier.to_numpy()


def build_fold_predictions(cases: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    for test_start in FOLD_STARTS:
        calibration = cases[
            cases["date"].between(
                test_start - pd.Timedelta(days=21), test_start - pd.Timedelta(days=1)
            )
        ].copy()
        test = cases[
            cases["date"].between(
                test_start, min(test_start + pd.Timedelta(days=6), cases["date"].max())
            )
        ].copy()
        global_lookup = aggregate_multiplier(calibration, ["cutoff"]).rename(
            columns={"multiplier": "global_multiplier"}
        )
        global_fallback = global_lookup[["cutoff", "global_multiplier"]]
        test = test.merge(global_fallback, on="cutoff", how="left", validate="many_to_one")
        test["global"] = test["raw_prediction"] * test["global_multiplier"]

        band_lookup = aggregate_multiplier(
            calibration,
            ["cutoff", "observed_band"],
            prior=global_fallback,
            prior_strength=100.0,
        )
        test["observed_band_shrink100"] = predict_with_lookup(
            test, band_lookup, ["cutoff", "observed_band"], global_fallback
        )

        for strength in [25.0, 50.0, 100.0, 200.0]:
            product_lookup = aggregate_multiplier(
                calibration,
                ["cutoff", "product_id"],
                prior=global_fallback,
                prior_strength=strength,
            )
            test[f"product_shrink{int(strength)}"] = predict_with_lookup(
                test, product_lookup, ["cutoff", "product_id"], global_fallback
            )

        product_band_lookup = aggregate_multiplier(
            calibration,
            ["cutoff", "product_id", "observed_band"],
            prior=global_fallback,
            prior_strength=100.0,
        )
        test["product_band_shrink100"] = predict_with_lookup(
            test,
            product_band_lookup,
            ["cutoff", "product_id", "observed_band"],
            global_fallback,
        )
        test["product_band_blend50_global"] = 0.5 * (
            test["product_band_shrink100"] + test["global"]
        )
        global_total = test.groupby("cutoff")["global"].transform("sum")
        hierarchical_total = test.groupby("cutoff")["product_band_shrink100"].transform("sum")
        test["product_band_rescaled_global"] = test["product_band_shrink100"] * (
            global_total / hierarchical_total.replace(0.0, np.nan)
        ).fillna(1.0)
        test["fold"] = test_start.date().isoformat()
        outputs.append(test)
    return pd.concat(outputs, ignore_index=True)


def metrics(frame: pd.DataFrame, prediction: str) -> dict[str, float | int | str]:
    error = frame[prediction] - frame["true_hidden"]
    true = float(frame["true_hidden"].sum())
    predicted = float(frame[prediction].sum())
    return {
        "variant": prediction,
        "cases": len(frame),
        "true_hidden": true,
        "predicted": predicted,
        "recovery_pct": 100 * predicted / true,
        "bias_pct": 100 * (predicted - true) / true,
        "wape_pct": 100 * float(error.abs().sum()) / true,
        "mae_units": float(error.abs().mean()),
        "within_1_unit_pct": 100 * float(error.abs().le(1.0).mean()),
        "within_3_units_pct": 100 * float(error.abs().le(3.0).mean()),
        "underpredicted_cases_pct": 100 * float(error.lt(0.0).mean()),
        "case_correlation": frame[prediction].corr(frame["true_hidden"]),
    }


def summarize(frame: pd.DataFrame, variants: list[str]) -> pd.DataFrame:
    return pd.DataFrame([metrics(frame, variant) for variant in variants]).sort_values(
        ["wape_pct", "mae_units"]
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = add_observed_band(build_cases(pd.read_parquet(HOURLY), CUTOFFS))
    predictions = build_fold_predictions(cases)
    variants = [
        "global",
        "observed_band_shrink100",
        "product_shrink25",
        "product_shrink50",
        "product_shrink100",
        "product_shrink200",
        "product_band_shrink100",
        "product_band_blend50_global",
        "product_band_rescaled_global",
    ]
    overall = summarize(predictions, variants)
    overall.to_csv(OUTPUT / "overall.csv", index=False, encoding="utf-8-sig")

    by_cutoff = []
    by_fold = []
    for cutoff, group in predictions.groupby("cutoff"):
        part = summarize(group, variants)
        part.insert(0, "cutoff", cutoff)
        by_cutoff.append(part)
    for fold, group in predictions.groupby("fold"):
        part = summarize(group, variants)
        part.insert(0, "fold", fold)
        by_fold.append(part)
    pd.concat(by_cutoff, ignore_index=True).to_csv(
        OUTPUT / "by_cutoff.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(by_fold, ignore_index=True).to_csv(
        OUTPUT / "by_fold.csv", index=False, encoding="utf-8-sig"
    )

    top_ids = set(pd.read_csv(TOP_PRODUCTS, encoding="utf-8-sig")["product_id"].astype(int))
    top20 = summarize(predictions[predictions["product_id"].isin(top_ids)], variants)
    top20.to_csv(OUTPUT / "top20.csv", index=False, encoding="utf-8-sig")
    predictions.to_parquet(OUTPUT / "case_predictions.parquet", index=False)
    print("Overall")
    print(overall.to_string(index=False))
    print("\nTop 20")
    print(top20.to_string(index=False))


if __name__ == "__main__":
    main()
