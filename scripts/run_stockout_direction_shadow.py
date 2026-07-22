"""Run the approved stockout direction as a local, read-only shadow job."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.record_stockout_prospective_shadow import record_snapshot  # noqa: E402

DEFAULT_OUTPUT = ROOT / "reports/stockout_direction_shadow"


def run_step(command: list[str]) -> None:
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        message = f"Shadow step failed ({result.returncode}): {' '.join(command)}"
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

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
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "offline_read_only_shadow",
        "production_write": False,
        "classification": classification,
        "demand_adjustment": adjustment,
        "historical_walk_forward": historical,
        "regime_aware_allocation": allocation,
        "combined_replay": replay,
        "decision": {
            "shadow_enabled_components": [
                "robust_demand_loss_preprocessing",
                "regime_aware_positive_capacity_allocation",
            ],
            "shadow_rejected_components": [
                "dynamic_walk_forward_allocation",
                "stockout_risk_allocation_due_to_normal_day_mae_regression",
            ],
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
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
