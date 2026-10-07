"""Offline audit of possible shelf-absence signals using stg sales, not mart labels.

The same-day release and movement fields establish only daily flow. They do not
timestamp shelf availability, so the output is a candidate-signal audit, not
ground-truth stockout labels or a reconstructed-demand training target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PILOT_IDS = {16, 20, 21, 22, 28, 80, 89, 107, 221, 222, 257}
BAKEABLE_CATEGORIES = {
    "Пироги сытные",
    "Пироги сладкие",
    "Выпечка сытная",
    "Выпечка сладкая",
    "Фастфуд",
}
KEYS = ["date", "bakery_id", "product_id"]
HOURS = list(range(7, 21))
TRAIN_END = pd.Timestamp("2026-07-05")
TEST_START = pd.Timestamp("2026-07-06")


def passes_frequency_guard(
    *,
    daily_sales: float,
    historical_median_daily_sales: float,
    same_window_days: float,
    same_window_zeros: float,
    all_window_days: float,
    all_window_zeros: float,
) -> bool:
    """Conservative activity filter, not an availability classifier."""
    return bool(
        daily_sales >= 10
        and historical_median_daily_sales >= 10
        and same_window_days >= 5
        and all_window_days >= 20
        and same_window_zeros / same_window_days <= 0.2
        and all_window_zeros / all_window_days <= 0.1
    )


def load_stg(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, set[int]]:
    sku_parts: list[pd.DataFrame] = []
    bakery_parts: list[pd.DataFrame] = []
    product_ids: set[int] = set()
    columns = [
        "check_datetime",
        "check_date",
        "cash_event_type",
        "quantity",
        "bakery_id",
        "product_id",
        "category_name",
    ]
    for chunk in pd.read_csv(path, usecols=columns, chunksize=250_000):
        chunk = chunk[
            (chunk["cash_event_type"] == "Продажа") & chunk["bakery_id"].isin(PILOT_IDS)
        ].copy()
        if chunk.empty:
            continue
        chunk["date"] = pd.to_datetime(chunk["check_date"]).dt.normalize()
        chunk["hour"] = (
            pd.to_datetime(chunk["check_datetime"], utc=True)
            .dt.tz_convert("Europe/Moscow")
            .dt.hour
        )
        chunk["sold"] = (
            pd.to_numeric(chunk["quantity"], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
        )
        bakery_parts.append(
            chunk.groupby(["date", "bakery_id", "hour"], as_index=False)["sold"].sum()
        )
        bakeable = chunk[chunk["category_name"].isin(BAKEABLE_CATEGORIES)]
        product_ids.update(
            int(value) for value in bakeable["product_id"].dropna().unique()
        )
        sku_parts.append(
            bakeable.groupby([*KEYS, "hour"], as_index=False)["sold"].sum()
        )
    sku = pd.concat(sku_parts).groupby([*KEYS, "hour"], as_index=False)["sold"].sum()
    bakery = (
        pd.concat(bakery_parts)
        .groupby(["date", "bakery_id", "hour"], as_index=False)["sold"]
        .sum()
    )
    return sku, bakery.rename(columns={"sold": "bakery_hour_sales"}), product_ids


def build_hourly(
    panel_path: Path, stg_path: Path
) -> tuple[pd.DataFrame, dict[str, int]]:
    sku, bakery, product_ids = load_stg(stg_path)
    columns = [
        *KEYS,
        "observed_sales_qty",
        "release_qty",
        "incoming_move_qty",
        "outgoing_move_qty",
        "written_off_qty",
    ]
    panel = pd.read_parquet(panel_path, columns=columns)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel = panel[
        panel["date"].between("2026-04-30", "2026-07-19")
        & panel["bakery_id"].isin(PILOT_IDS)
        & panel["product_id"].isin(product_ids)
    ].copy()
    if panel.duplicated(KEYS).any() or sku.duplicated([*KEYS, "hour"]).any():
        raise ValueError("Duplicate keys in prepared inputs")
    daily_stg = (
        sku.groupby(KEYS, as_index=False)["sold"]
        .sum()
        .rename(columns={"sold": "stg_sales"})
    )
    check = panel.merge(daily_stg, on=KEYS, how="left", validate="one_to_one")
    check["stg_sales"] = check["stg_sales"].fillna(0.0)
    reconcile = {
        "panel_sku_days": int(len(check)),
        "fct_stg_difference_gt1": int(
            (check["observed_sales_qty"] - check["stg_sales"]).abs().gt(1).sum()
        ),
    }
    frame = panel.merge(pd.DataFrame({"hour": HOURS}), how="cross")
    frame = frame.merge(sku, on=[*KEYS, "hour"], how="left", validate="one_to_one")
    frame = frame.merge(
        bakery, on=["date", "bakery_id", "hour"], how="left", validate="many_to_one"
    )
    frame[["sold", "bakery_hour_sales"]] = frame[["sold", "bakery_hour_sales"]].fillna(
        0.0
    )
    frame["dow"] = frame["date"].dt.dayofweek
    return frame, reconcile


def add_historical_reference(
    frame: pd.DataFrame,
    *,
    train_end: pd.Timestamp = TRAIN_END,
    test_start: pd.Timestamp = TEST_START,
) -> pd.DataFrame:
    training = frame[frame["date"] <= train_end]
    reference = (
        training.groupby(["bakery_id", "product_id", "dow", "hour"])
        .agg(expected=("sold", "median"), reference_days=("date", "nunique"))
        .reset_index()
    )
    holdout = frame[frame["date"] >= test_start].merge(
        reference,
        on=["bakery_id", "product_id", "dow", "hour"],
        how="left",
        validate="many_to_one",
    )
    holdout["expected"] = holdout["expected"].fillna(0.0)
    holdout["reference_days"] = holdout["reference_days"].fillna(0).astype(int)
    # A two-hour zero is informative only if such zeros are historically rare
    # for the same SKU and hour. Count past windows with an active bakery and
    # positive daily flow; include possibly censored days conservatively.
    historical = training.sort_values([*KEYS, "hour"]).copy()
    historical["next_sold"] = historical.groupby(KEYS)["sold"].shift(-1)
    historical["next_bakery_sales"] = historical.groupby(KEYS)[
        "bakery_hour_sales"
    ].shift(-1)
    historical["daily_sales"] = historical.groupby(KEYS)["sold"].transform("sum")
    historical["daily_supply"] = (
        historical["release_qty"]
        + historical["incoming_move_qty"]
        - historical["outgoing_move_qty"]
    )
    historical = historical[
        historical["hour"].between(10, 17)
        & historical["daily_supply"].gt(0)
        & (historical["bakery_hour_sales"] + historical["next_bakery_sales"]).ge(20)
    ].copy()
    historical["window_zero"] = (historical["sold"] + historical["next_sold"]).eq(0)
    same = (
        historical.groupby(["bakery_id", "product_id", "dow", "hour"])
        .agg(
            same_window_days=("date", "nunique"),
            same_window_zeros=("window_zero", "sum"),
        )
        .reset_index()
    )
    all_days = (
        historical.groupby(["bakery_id", "product_id", "hour"])
        .agg(
            all_window_days=("date", "nunique"),
            all_window_zeros=("window_zero", "sum"),
            historical_median_daily_sales=("daily_sales", "median"),
        )
        .reset_index()
    )
    holdout = holdout.merge(
        same,
        on=["bakery_id", "product_id", "dow", "hour"],
        how="left",
        validate="many_to_one",
    ).merge(
        all_days,
        on=["bakery_id", "product_id", "hour"],
        how="left",
        validate="many_to_one",
    )
    return holdout


def evaluate_signals(
    holdout: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    cases: list[dict[str, object]] = []
    known_sales_true: list[float] = []
    known_sales_pred: list[float] = []
    zero_with_supply = 0
    eligible_days = 0
    for key, group in holdout.groupby(KEYS, sort=False):
        group = group.set_index("hour").reindex(HOURS)
        sold = group["sold"].to_numpy(dtype=float)
        bakery = group["bakery_hour_sales"].to_numpy(dtype=float)
        expected = group["expected"].to_numpy(dtype=float)
        ref_days = group["reference_days"].to_numpy(dtype=int)
        supply = float(
            group["release_qty"].iloc[0]
            + group["incoming_move_qty"].iloc[0]
            - group["outgoing_move_qty"].iloc[0]
        )
        writeoff = float(group["written_off_qty"].iloc[0])
        if supply <= 0:
            continue
        eligible_days += 1
        if sold.sum() == 0:
            zero_with_supply += 1
            continue
        mid_candidates = []
        known_sales_candidates = []
        for hour in range(10, 18):
            index = hour - HOURS[0]
            if not (sold[index - 1] > 0 and sold[index + 2] > 0):
                continue
            if (
                bakery[index : index + 2].sum() < 20
                or expected[index : index + 2].sum() < 4
            ):
                continue
            if min(ref_days[index : index + 2]) < 3:
                continue
            observed = sold[index : index + 2].sum()
            if observed == 0:
                mid_candidates.append((hour, expected[index : index + 2].sum()))
            elif observed >= 4:
                known_sales_candidates.append(
                    (hour, expected[index : index + 2].sum(), observed)
                )
        if mid_candidates:
            hour, estimate = max(mid_candidates, key=lambda item: item[1])
            start = group.loc[hour]
            same_days = float(start.get("same_window_days", 0) or 0)
            all_days = float(start.get("all_window_days", 0) or 0)
            same_zeros = float(start.get("same_window_zeros", 0) or 0)
            all_zeros = float(start.get("all_window_zeros", 0) or 0)
            median_daily = float(start.get("historical_median_daily_sales", 0) or 0)
            frequency_guard = passes_frequency_guard(
                daily_sales=float(sold.sum()),
                historical_median_daily_sales=median_daily,
                same_window_days=same_days,
                same_window_zeros=same_zeros,
                all_window_days=all_days,
                all_window_zeros=all_zeros,
            )
            cases.append(
                {
                    "date": str(key[0].date()),
                    "bakery_id": key[1],
                    "product_id": key[2],
                    "signal": "interior_gap",
                    "start_hour": hour,
                    "stg_daily_sales": sold.sum(),
                    "daily_supply": supply,
                    "written_off_qty": writeoff,
                    "reference_missing_sales": estimate,
                    "historical_median_daily_sales": median_daily,
                    "same_window_days": same_days,
                    "same_window_zeros": same_zeros,
                    "all_window_days": all_days,
                    "all_window_zeros": all_zeros,
                    "frequency_guard_pass": frequency_guard,
                }
            )
        if known_sales_candidates:
            _, estimate, actual = max(known_sales_candidates, key=lambda item: item[1])
            known_sales_pred.append(estimate)
            known_sales_true.append(actual)
        positive = np.flatnonzero(sold > 0)
        last = int(positive[-1])
        if last <= 17 - HOURS[0]:
            remaining = slice(last + 1, None)
            tail_expectation = expected[remaining].sum()
            if (
                bakery[remaining].sum() >= 50
                and tail_expectation >= 6
                and np.all(ref_days[remaining] >= 3)
            ):
                cases.append(
                    {
                        "date": str(key[0].date()),
                        "bakery_id": key[1],
                        "product_id": key[2],
                        "signal": "early_tail",
                        "start_hour": HOURS[last] + 1,
                        "stg_daily_sales": sold.sum(),
                        "daily_supply": supply,
                        "written_off_qty": writeoff,
                        "reference_missing_sales": tail_expectation,
                        "historical_median_daily_sales": float("nan"),
                        "same_window_days": float("nan"),
                        "same_window_zeros": float("nan"),
                        "all_window_days": float("nan"),
                        "all_window_zeros": float("nan"),
                        "frequency_guard_pass": False,
                    }
                )
    result = pd.DataFrame(cases)
    truth = np.array(known_sales_true)
    pred = np.array(known_sales_pred)
    near_supply = pd.Series(False, index=result.index)
    if len(result):
        near_supply = result["stg_daily_sales"].div(result["daily_supply"]).between(
            0.9, 1.1
        ) & result["written_off_qty"].eq(0)
    summary: dict[str, float | int] = {
        "holdout_sku_days_with_positive_daily_supply": eligible_days,
        "zero_sales_despite_daily_supply_review_only": zero_with_supply,
        "candidate_sku_days": int(len(result.drop_duplicates(KEYS)))
        if len(result)
        else 0,
        "interior_gap_candidates": int((result["signal"] == "interior_gap").sum())
        if len(result)
        else 0,
        "interior_gap_frequency_guard_pass": int(result["frequency_guard_pass"].sum())
        if len(result)
        else 0,
        "early_tail_candidates": int((result["signal"] == "early_tail").sum())
        if len(result)
        else 0,
        "candidates_with_writeoffs": int(result["written_off_qty"].gt(0).sum())
        if len(result)
        else 0,
        "candidates_near_daily_supply_without_writeoffs": int(near_supply.sum()),
        "known_positive_sales_windows": len(truth),
        "known_positive_expected_to_actual_ratio": float(pred.sum() / truth.sum())
        if truth.sum()
        else float("nan"),
        "known_positive_wmape": float(np.abs(pred - truth).sum() / truth.sum())
        if truth.sum()
        else float("nan"),
    }
    return result, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        type=Path,
        default=ROOT / "reports/reconstructed_demand_full_20261005/panel.parquet",
    )
    parser.add_argument(
        "--stg",
        type=Path,
        default=ROOT / "data/raw/pilot_stg_check_lines_2026-04-30_2026-07-19.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports/stg_intraday_absence_20261005",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    frame, reconcile = build_hourly(args.panel, args.stg)
    holdout = add_historical_reference(frame)
    cases, summary = evaluate_signals(holdout)
    summary.update(reconcile)
    summary["holdout_bakeries"] = int(holdout["bakery_id"].nunique())
    summary.update(
        {
            "train_end": str(TRAIN_END.date()),
            "holdout_start": str(TEST_START.date()),
            "holdout_end": "2026-07-19",
            "source": "stg_check_lines plus deduplicated daily flow export",
            "interpretation": (
                "probabilistic anomaly candidates, not verified shelf absence"
            ),
        }
    )
    args.output_dir.mkdir(parents=True)
    cases.to_csv(args.output_dir / "candidates.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
