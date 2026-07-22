"""Run the approved stockout direction as a local, read-only shadow job."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.record_stockout_prospective_shadow import (  # noqa: E402
    record_candidate_evaluation,
    record_snapshot,
)

DEFAULT_OUTPUT = ROOT / "reports/stockout_direction_shadow"
DEFAULT_MEMBERSHIP_SEED_SUMMARY = (
    ROOT / "reports/demand_adjusted_membership_seed/summary_deltas.csv"
)
MEMBERSHIP_SEED_VARIANT = "demand_adjusted_membership_seed_0.05"
MEMBERSHIP_SEED_REQUIRED_SCOPES = [
    "all_holdout",
    "clean_sku_days",
    "adjusted_pairs_clean_sku_days",
    "new_tier1_member_clean_sku_days",
]


def run_step(command: list[str]) -> None:
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        message = f"Shadow step failed ({result.returncode}): {' '.join(command)}"
        raise RuntimeError(message)


def load_membership_seed_candidate(
    path: str | Path,
    *,
    evaluated_through: str,
    variant: str = MEMBERSHIP_SEED_VARIANT,
) -> dict:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    selected = frame[frame["variant"].eq(variant)].copy()
    if selected.empty:
        raise ValueError(f"Membership seed variant not found: {variant}")
    rows = []
    for scope in MEMBERSHIP_SEED_REQUIRED_SCOPES:
        part = selected[selected["scope"].eq(scope)]
        if part.empty:
            raise ValueError(f"Membership seed scope not found: {scope}")
        rows.append(
            {
                "scope": scope,
                "folds": int(part["cutoff"].nunique()),
                "wins": int(part["delta"].lt(0).sum()),
                "mean_sku_day_wape_delta": float(part["delta"].mean()),
                "mean_underforecast_qty_delta": float(part["under"].mean()),
                "mean_overforecast_qty_delta": float(part["over"].mean()),
            }
        )
    gates = {
        row["scope"]: row["folds"] >= 3 and row["wins"] == row["folds"]
        for row in rows
    }
    return {
        "candidate": "demand_adjusted_membership_seed",
        "seed_weight": 0.05,
        "variant": variant,
        "evaluation_mode": "historical_rolling_holdout",
        "evaluated_through": evaluated_through,
        "bakery_hour_total_preserved": True,
        "metrics": rows,
        "historical_gates": gates,
        "historical_gates_pass": all(gates.values()),
        "prospective_days_observed": 0,
        "status": "historical_pass_pending_prospective",
        "production_write": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--skip-refresh", action="store_true")
    parser.add_argument(
        "--membership-seed-summary",
        default=str(DEFAULT_MEMBERSHIP_SEED_SUMMARY),
    )
    parser.add_argument(
        "--membership-seed-evaluated-through",
        default="2026-07-19",
    )
    args = parser.parse_args()

    if not args.skip_refresh:
        run_step(
            [
                sys.executable,
                "scripts/classify_stockout_mechanisms.py",
                "--env-file",
                args.env_file,
            ]
        )
        run_step(
            [
                sys.executable,
                "scripts/build_demand_adjusted_stockout_history.py",
                "--env-file",
                args.env_file,
            ]
        )
        run_step(
            [
                sys.executable,
                "scripts/analyze_stockout_historical_shadow.py",
            ]
        )
        run_step(
            [
                sys.executable,
                "scripts/experiment_regime_aware_sku_allocation.py",
                "--env-file",
                args.env_file,
            ]
        )
        run_step(
            [
                sys.executable,
                "scripts/run_stockout_direction_combined_replay.py",
                "--env-file",
                args.env_file,
            ]
        )

    classification = json.loads(
        (ROOT / "reports/stockout_mechanism_classification/summary.json").read_text(
            encoding="utf-8"
        )
    )
    adjustment = json.loads(
        (ROOT / "reports/demand_adjusted_stockout_history/summary.json").read_text(
            encoding="utf-8"
        )
    )
    historical = json.loads(
        (ROOT / "reports/stockout_historical_shadow/summary.json").read_text(
            encoding="utf-8"
        )
    )
    allocation = json.loads(
        (
            ROOT / "reports/regime_aware_sku_allocation_experiment/summary.json"
        ).read_text(encoding="utf-8")
    )
    replay = json.loads(
        (ROOT / "reports/stockout_direction_combined_replay/summary.json").read_text(
            encoding="utf-8"
        )
    )
    membership_seed = load_membership_seed_candidate(
        args.membership_seed_summary,
        evaluated_through=args.membership_seed_evaluated_through,
    )
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "offline_read_only_shadow",
        "production_write": False,
        "classification": classification,
        "demand_adjustment": adjustment,
        "historical_walk_forward": historical,
        "regime_aware_allocation": allocation,
        "combined_replay": replay,
        "membership_seed_profile": membership_seed,
        "decision": {
            "shadow_enabled_components": [
                "robust_demand_loss_preprocessing",
                "regime_aware_positive_capacity_allocation",
            ],
            "shadow_rejected_components": [
                "dynamic_walk_forward_allocation",
                "stockout_risk_allocation_due_to_normal_day_mae_regression",
            ],
            "shadow_candidate_components": [
                "demand_adjusted_membership_seed_0.05",
            ],
            "candidate_prospective_counting": (
                "starts only after a new evaluation date later than 2026-07-19"
            ),
        },
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest["prospective_shadow"] = record_snapshot(
        manifest,
        output / "history",
        timezone_name="Europe/Moscow",
        minimum_days=21,
    )
    candidate_shadow = record_candidate_evaluation(
        membership_seed,
        output / "membership_seed_history",
        start_after="2026-07-19",
        minimum_days=21,
    )
    membership_seed["prospective_shadow"] = candidate_shadow
    membership_seed["prospective_days_observed"] = candidate_shadow[
        "prospective_days_observed"
    ]
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
