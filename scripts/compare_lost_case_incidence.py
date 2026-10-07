"""Compare factual and counterfactual lost-demand incidence on paired SKU-days."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/three_prod_models_august_holdout_20260911"
KEYS = ["date", "bakery_id", "product_id"]
VARIANTS = [
    "base_bakery_norm_recent",
    "direct_alpha_025_v1",
    "p50_loss_075_causal_daily",
    "base_raw_uplift_reconstructed",
]
IMPROVEMENT_VARIANTS = {
    variant: OUTPUT / f"economics_{variant}.parquet" for variant in VARIANTS
}
IMPROVEMENT_VARIANTS.update(
    {
        "endogenous_residual_q85": ROOT
        / "reports/direct_endogenous_residual_layer_20260911"
        / "economics_endogenous_residual_q85_q25_cap3.parquet",
        "endogenous_residual_q80": ROOT
        / "reports/direct_endogenous_residual_layer_20260911"
        / "economics_endogenous_residual_q80_q30_cap5.parquet",
    }
)
THRESHOLDS = {
    "any_positive": 1e-9,
    "at_least_one_unit": 1.0,
}


def main() -> None:
    actual = pd.read_parquet(
        OUTPUT / "economics_actual_state.parquet", columns=KEYS + ["lost"]
    ).rename(columns={"lost": "actual_lost"})
    incidence_records = []
    transition_records = []
    improvement_records = []
    total_rows = len(actual)

    for threshold_name, threshold in THRESHOLDS.items():
        actual_case = actual["actual_lost"].ge(threshold)
        actual_cases = int(actual_case.sum())
        incidence_records.append(
            {
                "threshold": threshold_name,
                "variant": "actual_state",
                "rows": total_rows,
                "lost_cases": actual_cases,
                "lost_case_pct": 100 * actual_cases / total_rows,
                "case_delta_vs_actual": 0,
                "relative_case_delta_vs_actual_pct": 0.0,
            }
        )
        for variant in VARIANTS:
            model = pd.read_parquet(
                OUTPUT / f"economics_{variant}.parquet", columns=KEYS + ["lost"]
            ).rename(columns={"lost": "model_lost"})
            paired = actual.merge(model, on=KEYS, validate="one_to_one")
            paired["actual_case"] = paired["actual_lost"].ge(threshold)
            paired["model_case"] = paired["model_lost"].ge(threshold)
            model_cases = int(paired["model_case"].sum())
            incidence_records.append(
                {
                    "threshold": threshold_name,
                    "variant": variant,
                    "rows": total_rows,
                    "lost_cases": model_cases,
                    "lost_case_pct": 100 * model_cases / total_rows,
                    "case_delta_vs_actual": model_cases - actual_cases,
                    "relative_case_delta_vs_actual_pct": (
                        100 * (model_cases - actual_cases) / actual_cases
                    ),
                }
            )
            transition_records.append(
                {
                    "threshold": threshold_name,
                    "variant": variant,
                    "both_have_lost": int(
                        (paired["actual_case"] & paired["model_case"]).sum()
                    ),
                    "actual_only_model_fixed": int(
                        (paired["actual_case"] & ~paired["model_case"]).sum()
                    ),
                    "model_only_new_case": int(
                        (~paired["actual_case"] & paired["model_case"]).sum()
                    ),
                    "neither_has_lost": int(
                        (~paired["actual_case"] & ~paired["model_case"]).sum()
                    ),
                }
            )

    # Business success does not require fully eliminating the lost-demand case.
    # A row is successful when the counterfactual policy reduces lost demand by
    # at least one unit relative to the factual policy.
    actual_case = actual["actual_lost"].ge(1.0)
    actual_cases = int(actual_case.sum())
    for variant, model_path in IMPROVEMENT_VARIANTS.items():
        model = pd.read_parquet(model_path, columns=KEYS + ["lost"]).rename(
            columns={"lost": "model_lost"}
        )
        paired = actual.merge(model, on=KEYS, validate="one_to_one")
        delta = paired["actual_lost"] - paired["model_lost"]
        improved = paired["actual_lost"].ge(1.0) & delta.ge(1.0)
        worsened = delta.le(-1.0)
        fully_closed = improved & paired["model_lost"].lt(1.0)
        partial = improved & ~fully_closed
        improvement_records.append(
            {
                "variant": variant,
                "actual_cases": actual_cases,
                "improved_cases": int(improved.sum()),
                "partial_improvements": int(partial.sum()),
                "fully_closed": int(fully_closed.sum()),
                "worsened_rows": int(worsened.sum()),
                "unchanged_rows": int((~improved & ~worsened).sum()),
                "recovered_units_positive": float(delta.clip(lower=0).sum()),
                "additional_lost_units": float((-delta).clip(lower=0).sum()),
                "net_recovered_units": float(delta.sum()),
                "actual_lost_qty": float(paired["actual_lost"].sum()),
                "model_lost_qty": float(paired["model_lost"].sum()),
                "improved_share_actual_cases_pct": 100
                * int(improved.sum())
                / actual_cases,
            }
        )

    incidence = pd.DataFrame(incidence_records)
    transitions = pd.DataFrame(transition_records)
    improvements = pd.DataFrame(improvement_records)
    incidence.to_csv(
        OUTPUT / "lost_case_incidence.csv", index=False, encoding="utf-8-sig"
    )
    transitions.to_csv(
        OUTPUT / "lost_case_transitions.csv", index=False, encoding="utf-8-sig"
    )
    improvements.to_csv(
        OUTPUT / "lost_improvement_vs_actual.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(
        incidence[incidence["threshold"].eq("at_least_one_unit")].to_string(
            index=False
        )
    )
    print(
        transitions[transitions["threshold"].eq("at_least_one_unit")].to_string(
            index=False
        )
    )
    print(improvements.to_string(index=False))


if __name__ == "__main__":
    main()
