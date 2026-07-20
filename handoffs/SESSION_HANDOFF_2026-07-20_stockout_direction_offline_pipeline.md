# Session Handoff — 2026-07-20 — Stockout Direction Offline Pipeline

## Scope and safety boundary

The user approved the full offline plan for separating stockout-related SKU
allocation errors from lost bakery demand, then left the execution autonomous
with one hard constraint: **do not touch production**.

That boundary was preserved. ClickHouse was used read only. No production
tables, services, timers, forecast runs, profiles, environment variables, or
deployments were changed. Candidate datasets and forecasts were written only
to local `reports/` files.

This handoff is a historical session record. `docs/ops/` remains the current
operational source of truth and takes precedence over this file.

## Agreed problem decomposition

Two mechanisms must have different implementations:

1. **Allocation error:** bakery total is approximately correct, but SKU shares
   are wrong. A candidate allocation model may redistribute volume between
   SKUs, but must preserve the bakery-day total.
2. **Demand loss:** the stockout reduces observed bakery sales. This is a data
   censoring problem, so missing demand must be reconstructed before building
   profiles, lag/rolling features, and bakery targets. Both SKU and bakery
   volume may increase.

Cases that cannot be identified confidently remain unchanged. WMAPE is not a
decision metric for this direction because unchanged production behaviour can
make a valid demand forecast look intentionally high on stockout days.

## Completed work

### 1. Leakage-safe classification

The 397 previously confirmed model-underforecast stockout cases were compared
with a trailing same-weekday bakery counterfactual using only earlier dates.

Primary classification:

| Mechanism | Cases | Confirmed shortfall |
| --- | ---: | ---: |
| Allocation | 240 | 961.2 |
| Demand loss | 57 | 213.5 |
| Uncertain | 100 | 334.6 |

Threshold sensitivity produced a high-precision intersection used downstream:

- 231 robust allocation cases;
- 26 robust demand-loss cases;
- 140 uncertain cases.

No stable mixed class was found under the strict definition. The demand-loss
label means strong co-occurrence of a stockout and abnormally low bakery
volume; it is not causal proof that the single SKU explains the full bakery
gap.

### 2. Demand-adjusted preprocessing

Only the 26 robust demand-loss cases were eligible for reconstruction. The
pipeline uses prior non-stockout weekdays and combines direct SKU-hour demand
with the SKU share of bakery-hour traffic. It requires at least three reference
days, verifies continued bakery activity, and applies per-hour and per-case
caps. Original sales are retained separately for audit.

Result:

- 25/26 cases adjusted;
- 140.313 units reconstructed;
- median adjustment 3.75 units, maximum 15;
- 96.15% reference coverage;
- 17/825 bakery-days and 25/15,724 profile cells affected;
- total target uplift 0.01425% of 984,765.955 observed units;
- maximum profile-share delta 0.17895 percentage points.

This implementation is deliberately conservative and suitable for local
shadow collection, but the current sample is too sparse to justify bakery
model retraining.

### 3. Synthetic reconstruction backtest

Pseudo-stockouts were created by hiding sales from known non-stockout days.
The existing bakery-share method recovered roughly 75–81% of hidden demand.
A guarded hybrid recovered 82–87% for higher-volume SKU ending two or three
hours early, but regressed for low-volume SKU and four-hour gaps. No universal
hybrid was promoted; conservative caps remain in the shadow configuration.

### 4. Dynamic allocation experiment

A strict walk-forward LightGBM model predicted a capped log correction to SKU
share, trained only on earlier confirmed non-stockout rows. All scenarios were
renormalized to preserve the original bakery-day total.

Best candidate: `model_log_ratio_strength_0.25`.

| Metric | Baseline | Candidate |
| --- | ---: | ---: |
| Stockout shortfall | 1,530.29 | 1,553.07 |
| Normal-day MAE | 4.68059 | 4.69620 |
| Cases fixed | — | 2 |
| New underforecast cases | — | 5 |
| Maximum bakery-total delta | — | approximately 0 |

The candidate failed both stockout and normal-day gates and was rejected from
shadow. Stronger model strengths and direct bakery×SKU calibration were worse.

The earlier current-profile replay still improves the historical shortfall by
about 33%, but it contains evaluation-period information and is therefore
diagnostic/look-ahead evidence, not a deployable walk-forward result.

### 5. Combined replay

| Scenario | Shortfall | Fixed | Improved | Worsened |
| --- | ---: | ---: | ---: | ---: |
| Historical baseline | 1,509.285 | 0 | 0 | 0 |
| Demand preprocessing only | 1,453.585 | 23 | 25 | 0 |
| Current profile diagnostic | 1,005.857 | 133 | 282 | 112 |
| Current profile + demand diagnostic | 973.223 | 143 | 286 | 108 |
| Walk-forward dynamic allocation | 1,525.407 | 2 | 39 | 113 |
| Dynamic allocation + demand | 1,469.478 | 25 | 62 | 106 |

Only conservative demand preprocessing improved the confirmed cases without
creating new underforecasts. It is the only mechanism accepted into local
shadow.

## Decision state

Accepted into local shadow:

- robust demand-loss classification;
- conservative demand reconstruction;
- adjusted SKU-day and bakery-day targets;
- adjusted lag/rolling columns and share profiles;
- full case/hour audit.

Rejected from shadow:

- the current walk-forward dynamic allocation model;
- direct bakery×SKU ratio calibration.

Diagnostic only:

- current-profile allocation replay, due to look-ahead.

Deferred:

- production activation;
- bakery model retraining until enough adjusted history accumulates;
- mixed-case restoration;
- cold-start allocation evaluation.

## Code and key artifacts

Implementation:

- `config/stockout_direction_shadow.json`
- `scripts/classify_stockout_mechanisms.py`
- `scripts/build_demand_adjusted_stockout_history.py`
- `scripts/experiment_dynamic_sku_allocation.py`
- `scripts/backtest_pseudo_stockout_reconstruction.py`
- `scripts/run_stockout_direction_combined_replay.py`
- `scripts/run_stockout_direction_shadow.py`

Primary narrative report:

- `docs/stockout_direction_results_20260720.md`

Compact machine-readable artifacts:

- `reports/stockout_mechanism_classification/summary.json`
- `reports/stockout_mechanism_classification/threshold_sensitivity.csv`
- `reports/stockout_mechanism_classification/manual_review_sample.csv`
- `reports/demand_adjusted_stockout_history/summary.json`
- `reports/demand_adjusted_stockout_history/case_adjustments.csv`
- `reports/dynamic_sku_allocation_experiment/summary.json`
- `reports/dynamic_sku_allocation_experiment/scenario_comparison.csv`
- `reports/stockout_direction_combined_replay/summary.json`
- `reports/stockout_direction_combined_replay/scenario_comparison.csv`
- `reports/stockout_direction_shadow/manifest.json`
- `reports/pseudo_stockout_backtest_10/summary.csv`
- `reports/pseudo_stockout_backtest_10/summary.json`

## Reproduction

Run the complete read-only local shadow pipeline:

```powershell
.venv\Scripts\python.exe scripts\run_stockout_direction_shadow.py --env-file .env
```

The runner refreshes classification, demand-adjusted history, and combined
replay, then writes `reports/stockout_direction_shadow/manifest.json`. It has
no ClickHouse write path.

Promotion gates in `config/stockout_direction_shadow.json` require:

- at least 21 shadow days;
- no normal-day bias regression;
- no new underforecast cases;
- manual review before any production proposal.

## Verification and known unrelated failures

The new targeted test set passes (`6 passed`) and Ruff is clean for the added
code. The broader suite status at handoff:

- excluding the pre-existing collection blocker: `257 passed`;
- collection blocker: `tests/test_build_bakeable_products_table.py` imports
  missing `build_bakeable_table` from
  `scripts/build_bakeable_products_table.py`;
- four pre-existing unrelated failures remain: three in
  `tests/test_apply_bakery_profiles_clickhouse_recent.py` and one in
  `tests/test_daily_profile_blending.py::test_blending_downweights_weak_local_profile`.

These unrelated failures were not changed as part of this direction.

## Next safe steps

1. Run the local read-only shadow regularly and accumulate at least 21 distinct
   days before evaluating promotion.
2. Review `manual_review_sample.csv`, especially the boundary and uncertain
   cases, and record false-positive patterns.
3. Re-run allocation modelling after a longer history is available. Do not
   promote the current walk-forward candidate.
4. Recalibrate reconstruction by segment only if the accumulated synthetic and
   real shadow evidence shows stable gains without false uplift.
5. Any production proposal requires a separate review and explicit approval.

There is no production rollback step because this work made no production
changes.

## Commits

| Hash | Message |
| --- | --- |
| `7469f9d` | `feat: add offline stockout direction pipeline` |
| `5f029e8` | `docs: record offline stockout direction results` |

## Worktree note

At handoff creation, `.claude/settings.local.json` and
`docs/ops/CURRENT_STATE.md` contained pre-existing unrelated local changes.
They were deliberately preserved and excluded from this work.
