"""Record one prospective stockout-shadow observation per local calendar day."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def _metric(source: dict, *path: str, default=None):
    value = source
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def record_snapshot(
    manifest: dict,
    history_dir: Path,
    timezone_name: str = "Europe/Moscow",
    minimum_days: int = 21,
) -> dict:
    generated = datetime.fromisoformat(manifest["generated_at"])
    local_date = generated.astimezone(ZoneInfo(timezone_name)).date().isoformat()
    history_dir.mkdir(parents=True, exist_ok=True)
    day_path = history_dir / f"{local_date}.json"
    existing = (
        json.loads(day_path.read_text(encoding="utf-8")) if day_path.exists() else {}
    )

    allocation = manifest.get("regime_aware_allocation", {})
    combined = manifest.get("combined_replay", {})
    normal_mae_delta = _metric(allocation, "best_metrics", "normal_mae_delta")
    new_underforecast = _metric(
        allocation, "best_metrics", "stockout_new_underforecast"
    )
    combined_scenario = next(
        (
            item
            for item in combined.get("scenarios", [])
            if item.get("scenario") == "regime_aware_allocation_plus_demand"
        ),
        {},
    )
    worsened = combined_scenario.get("cases_worsened")
    gates = {
        "normal_day_bias_not_regressed": normal_mae_delta is not None
        and normal_mae_delta <= 0,
        "no_new_underforecast_cases": new_underforecast == 0,
        "no_combined_cases_worsened": worsened == 0,
    }
    record = {
        "shadow_date": local_date,
        "timezone": timezone_name,
        "first_recorded_at": existing.get(
            "first_recorded_at", manifest["generated_at"]
        ),
        "latest_recorded_at": manifest["generated_at"],
        "run_count": int(existing.get("run_count", 0)) + 1,
        "metrics": {
            "normal_day_mae_delta": normal_mae_delta,
            "new_underforecast_cases": new_underforecast,
            "combined_cases_worsened": worsened,
        },
        "gates": gates,
    }
    membership_seed = manifest.get("membership_seed_profile")
    if membership_seed:
        record["candidate_observations"] = {
            "demand_adjusted_membership_seed_0.05": {
                "status": membership_seed.get("status"),
                "evaluated_through": membership_seed.get("evaluated_through"),
                "historical_gates_pass": membership_seed.get(
                    "historical_gates_pass"
                ),
                "counts_as_prospective_evidence": False,
            }
        }
    day_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(history_dir.glob("????-??-??.json"))
    ]
    summary = {
        "timezone": timezone_name,
        "minimum_days": minimum_days,
        "prospective_days_observed": len(records),
        "date_from": records[0]["shadow_date"] if records else None,
        "date_to": records[-1]["shadow_date"] if records else None,
        "minimum_days_met": len(records) >= minimum_days,
        "all_observed_gates_pass": bool(records)
        and all(all(item["gates"].values()) for item in records),
        "latest": records[-1] if records else None,
        "candidate_tracking": (
            records[-1].get("candidate_observations", {}) if records else {}
        ),
    }
    (history_dir / "index.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def record_candidate_evaluation(
    candidate: dict,
    history_dir: Path,
    *,
    start_after: str,
    minimum_days: int = 21,
) -> dict:
    """Count distinct newly evaluated actual dates, never repeated backtests."""
    history_dir.mkdir(parents=True, exist_ok=True)
    evaluation_date = candidate.get("evaluated_through")
    if evaluation_date and evaluation_date > start_after:
        path = history_dir / f"{evaluation_date}.json"
        record = {
            "evaluation_date": evaluation_date,
            "candidate": candidate.get("candidate"),
            "variant": candidate.get("variant"),
            "historical_gates_pass": candidate.get("historical_gates_pass"),
            "metrics": candidate.get("metrics", []),
            "production_write": False,
        }
        path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(history_dir.glob("????-??-??.json"))
    ]
    summary = {
        "start_after": start_after,
        "minimum_days": minimum_days,
        "prospective_days_observed": len(records),
        "date_from": records[0]["evaluation_date"] if records else None,
        "date_to": records[-1]["evaluation_date"] if records else None,
        "minimum_days_met": len(records) >= minimum_days,
        "all_observed_gates_pass": bool(records)
        and all(item["historical_gates_pass"] for item in records),
    }
    (history_dir / "index.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--history-dir", required=True)
    parser.add_argument("--timezone", default="Europe/Moscow")
    parser.add_argument("--minimum-days", type=int, default=21)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    summary = record_snapshot(
        manifest, Path(args.history_dir), args.timezone, args.minimum_days
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
