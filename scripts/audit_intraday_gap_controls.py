"""Compare stg two-hour gap candidates with other eligible windows offline.

Daily supply and write-offs are weak contextual evidence, not ground-truth
availability labels. This script never produces reconstructed demand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_stg_intraday_absence import (
    HOURS,
    KEYS,
    ROOT,
    add_historical_reference,
    build_hourly,
    passes_frequency_guard,
)

REFERENCE_END = pd.Timestamp("2026-06-15")
CONTROL_START = pd.Timestamp("2026-06-16")
TEST_START = pd.Timestamp("2026-07-06")


def flow_band(daily_sales: float, daily_supply: float, writeoffs: float) -> str:
    """Daily-flow grouping for descriptive controls, not shelf status."""
    if daily_supply <= 0:
        return "unknown"
    ratio = daily_sales / daily_supply
    if 0.9 <= ratio <= 1.1 and writeoffs == 0:
        return "near_supply_no_writeoff"
    if ratio <= 0.7 and writeoffs > 0:
        return "flow_surplus_with_writeoff"
    return "other"


def eligible_windows(holdout: pd.DataFrame) -> pd.DataFrame:
    """Enumerate every window meeting the same frequency guard as a gap."""
    rows: list[dict[str, object]] = []
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
        if supply <= 0:
            continue
        daily_sales = float(sold.sum())
        writeoffs = float(group["written_off_qty"].iloc[0])
        for hour in range(10, 18):
            index = hour - HOURS[0]
            if not (sold[index - 1] > 0 and sold[index + 2] > 0):
                continue
            if bakery[index : index + 2].sum() < 20:
                continue
            if expected[index : index + 2].sum() < 4:
                continue
            if min(ref_days[index : index + 2]) < 3:
                continue
            start = group.loc[hour]
            same_days = float(start["same_window_days"])
            all_days = float(start["all_window_days"])
            same_zeros = float(start["same_window_zeros"])
            all_zeros = float(start["all_window_zeros"])
            median_daily = float(start["historical_median_daily_sales"])
            if not passes_frequency_guard(
                daily_sales=daily_sales,
                historical_median_daily_sales=median_daily,
                same_window_days=same_days,
                same_window_zeros=same_zeros,
                all_window_days=all_days,
                all_window_zeros=all_zeros,
            ):
                continue
            rows.append(
                {
                    "date": key[0],
                    "bakery_id": key[1],
                    "product_id": key[2],
                    "hour": hour,
                    "window_sold": float(sold[index : index + 2].sum()),
                    "daily_sales": daily_sales,
                    "daily_supply": supply,
                    "writeoffs": writeoffs,
                    "flow_band": flow_band(daily_sales, supply, writeoffs),
                    "period": "control" if key[0] < TEST_START else "test",
                    "next_two_hours_sold": float(sold[index + 2 : index + 4].sum()),
                    "next_two_hours_expected": float(
                        expected[index + 2 : index + 4].sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def eligible_fixed_tails(holdout: pd.DataFrame) -> pd.DataFrame:
    """Use the same 18:00-20:00 tail for control and test dates."""
    rows: list[dict[str, object]] = []
    first = 18 - HOURS[0]
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
        if supply <= 0 or sold[:first].sum() < 10:
            continue
        if bakery[first:].sum() < 50 or expected[first:].sum() < 6:
            continue
        if min(ref_days[first:]) < 3:
            continue
        writeoffs = float(group["written_off_qty"].iloc[0])
        daily_sales = float(sold.sum())
        rows.append(
            {
                "date": key[0],
                "bakery_id": key[1],
                "product_id": key[2],
                "period": "control" if key[0] < TEST_START else "test",
                "flow_band": flow_band(daily_sales, supply, writeoffs),
                "zero_tail": bool(sold[first:].sum() == 0),
                "tail_sales": float(sold[first:].sum()),
                "expected_tail_sales": float(expected[first:].sum()),
            }
        )
    return pd.DataFrame(rows)


def attach_peer_sales(cases: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    """Peer SKU sales during the candidate gap in the ten-bakery panel."""
    hourly = holdout[[*KEYS, "hour", "sold"]]
    peers = hourly.groupby(["date", "product_id", "hour"], as_index=False).agg(
        all_bakery_sales=("sold", "sum"),
        selling_bakeries=("sold", lambda value: int(value.gt(0).sum())),
    )
    result = cases.copy()
    parts = []
    for offset in (0, 1):
        segment = result[["date", "bakery_id", "product_id", "hour"]].copy()
        segment["hour"] += offset
        segment = segment.merge(
            peers,
            on=["date", "product_id", "hour"],
            how="left",
            validate="many_to_one",
        )
        parts.append(segment)
    result["peer_sales_during_gap"] = sum(
        part["all_bakery_sales"].fillna(0).to_numpy() for part in parts
    )
    result["peer_bakery_hours_selling"] = sum(
        part["selling_bakeries"].fillna(0).to_numpy() for part in parts
    )
    return result


def summarize(
    windows: pd.DataFrame,
    tails: pd.DataFrame,
    cases: pd.DataFrame,
    zero_windows_with_peers: pd.DataFrame,
    original_strong_cases: int,
) -> dict[str, object]:
    cohort = (
        windows.assign(zero_gap=windows["window_sold"].eq(0))
        .groupby(["period", "flow_band"])
        .agg(
            eligible_windows=("zero_gap", "size"),
            zero_gap_windows=("zero_gap", "sum"),
        )
        .reset_index()
    )
    cohort["observed_zero_rate"] = (
        cohort["zero_gap_windows"] / cohort["eligible_windows"]
    )
    tail_cohort = (
        tails.groupby(["period", "flow_band"])
        .agg(eligible_days=("zero_tail", "size"), zero_tail_days=("zero_tail", "sum"))
        .reset_index()
    )
    tail_cohort["observed_zero_rate"] = (
        tail_cohort["zero_tail_days"] / tail_cohort["eligible_days"]
    )
    peer_cohort = (
        zero_windows_with_peers.groupby(["period", "flow_band"])
        .agg(
            zero_gap_windows=("date", "size"),
            zero_gaps_with_peer_sales=(
                "peer_sales_during_gap",
                lambda value: int(value.gt(0).sum()),
            ),
            median_peer_sales=("peer_sales_during_gap", "median"),
        )
        .reset_index()
    )
    return {
        "scope": "2026-06-16..2026-07-19; 10 pilot bakeries; offline stg sales",
        "frozen_reference_end": str(REFERENCE_END.date()),
        "control_period": "2026-06-16..2026-07-05",
        "test_period": "2026-07-06..2026-07-19",
        "frequency_guard_eligible_windows": int(len(windows)),
        "frequency_guard_zero_windows": int(windows["window_sold"].eq(0).sum()),
        "cohorts": cohort.to_dict(orient="records"),
        "fixed_18_to_20_tail_cohorts": tail_cohort.to_dict(orient="records"),
        "zero_gap_peer_context": peer_cohort.to_dict(orient="records"),
        "original_strong_cases": original_strong_cases,
        "reviewed_strong_cases_surviving_frozen_reference": int(len(cases)),
        "strong_cases_with_peer_sales": int(cases["peer_sales_during_gap"].gt(0).sum()),
        "strong_peer_sales_median": float(cases["peer_sales_during_gap"].median()),
        "strong_next_two_hours_sales_median": float(
            cases["next_two_hours_sold"].median()
        ),
        "interpretation": (
            "Matched window conditions and one frozen historical reference for "
            "both periods. No group confirms physical shelf presence or absence."
        ),
    }


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
        "--candidates",
        type=Path,
        default=ROOT
        / "reports/stg_intraday_absence_20261005_frequency_guard/candidates.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports/stg_intraday_gap_controls_frozen_20261005",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    frame, _ = build_hourly(args.panel, args.stg)
    holdout = add_historical_reference(
        frame, train_end=REFERENCE_END, test_start=CONTROL_START
    )
    windows = eligible_windows(holdout)
    tails = eligible_fixed_tails(holdout)
    raw_cases = pd.read_csv(args.candidates, parse_dates=["date"])
    cases = raw_cases[
        raw_cases["signal"].eq("interior_gap")
        & raw_cases["frequency_guard_pass"]
        & raw_cases["written_off_qty"].eq(0)
        & raw_cases["stg_daily_sales"].div(raw_cases["daily_supply"]).between(0.9, 1.1)
    ].copy()
    original_strong_cases = len(cases)
    cases = cases.rename(columns={"start_hour": "hour"})
    cases = cases.merge(
        windows[[*KEYS, "hour", "next_two_hours_sold", "next_two_hours_expected"]],
        on=[*KEYS, "hour"],
        how="inner",
        validate="one_to_one",
    )
    cases = attach_peer_sales(cases, holdout)
    zero_windows = attach_peer_sales(
        windows[windows["window_sold"].eq(0)].copy(), holdout
    )
    summary = summarize(windows, tails, cases, zero_windows, original_strong_cases)
    args.output_dir.mkdir(parents=True)
    windows.to_csv(args.output_dir / "eligible_windows.csv", index=False)
    tails.to_csv(args.output_dir / "eligible_fixed_tails.csv", index=False)
    zero_windows.to_csv(args.output_dir / "zero_gap_peer_context.csv", index=False)
    cases.to_csv(args.output_dir / "reviewed_cases.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
