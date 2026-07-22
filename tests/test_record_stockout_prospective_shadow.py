from scripts.record_stockout_prospective_shadow import (
    record_candidate_evaluation,
    record_snapshot,
)


def manifest(generated_at: str, normal_delta: float = -0.1) -> dict:
    return {
        "generated_at": generated_at,
        "regime_aware_allocation": {
            "best_metrics": {
                "normal_mae_delta": normal_delta,
                "stockout_new_underforecast": 0,
            }
        },
        "combined_replay": {
            "scenarios": [
                {
                    "scenario": "regime_aware_allocation_plus_demand",
                    "cases_worsened": 0,
                }
            ]
        },
    }


def test_same_local_day_is_counted_once(tmp_path):
    first = record_snapshot(manifest("2026-07-20T22:10:00+00:00"), tmp_path)
    second = record_snapshot(manifest("2026-07-21T10:00:00+00:00"), tmp_path)

    assert first["prospective_days_observed"] == 1
    assert second["prospective_days_observed"] == 1
    assert second["latest"]["shadow_date"] == "2026-07-21"
    assert second["latest"]["run_count"] == 2
    assert second["latest"]["first_recorded_at"] == "2026-07-20T22:10:00+00:00"


def test_distinct_local_days_and_gate_state(tmp_path):
    record_snapshot(manifest("2026-07-21T10:00:00+00:00"), tmp_path)
    result = record_snapshot(
        manifest("2026-07-21T22:00:00+00:00", normal_delta=0.2), tmp_path
    )

    assert result["prospective_days_observed"] == 2
    assert result["date_from"] == "2026-07-21"
    assert result["date_to"] == "2026-07-22"
    assert not result["all_observed_gates_pass"]
    assert not result["minimum_days_met"]


def test_historical_candidate_is_visible_but_not_counted_as_prospective(tmp_path):
    payload = manifest("2026-07-21T10:00:00+00:00")
    payload["membership_seed_profile"] = {
        "status": "historical_pass_pending_prospective",
        "evaluated_through": "2026-07-19",
        "historical_gates_pass": True,
    }

    result = record_snapshot(payload, tmp_path)

    candidate = result["candidate_tracking"][
        "demand_adjusted_membership_seed_0.05"
    ]
    assert candidate["historical_gates_pass"]
    assert not candidate["counts_as_prospective_evidence"]


def test_candidate_counts_only_new_evaluation_dates(tmp_path):
    candidate = {
        "evaluated_through": "2026-07-19",
        "historical_gates_pass": True,
    }
    first = record_candidate_evaluation(
        candidate,
        tmp_path,
        start_after="2026-07-19",
    )
    candidate["evaluated_through"] = "2026-07-20"
    second = record_candidate_evaluation(
        candidate,
        tmp_path,
        start_after="2026-07-19",
    )
    repeated = record_candidate_evaluation(
        candidate,
        tmp_path,
        start_after="2026-07-19",
    )

    assert first["prospective_days_observed"] == 0
    assert second["prospective_days_observed"] == 1
    assert repeated["prospective_days_observed"] == 1
