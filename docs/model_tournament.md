# Model Tournament

`scripts/run_model_tournament.py` is the read-only entry point for comparing
forecast models on one frozen SKU-day panel. It never connects to ClickHouse,
activates runs, changes snapshots, or publishes forecasts.

## Current MVP

The runner provides:

- one canonical `date + bakery + product` evaluation universe;
- causal deterministic baselines;
- adapters for forecast columns already stored in the input panel;
- fresh LightGBM P60 quantile and weighted-weekday residual candidates;
- strict duplicate, coverage, finite-value and non-negative-value guards;
- MAE, MSE, RMSE, WMAPE, zero-safe MAPE, sMAPE and bias;
- underforecast, overforecast, demand-satisfaction and restored-loss diagnostics;
- breakdowns by date, week, month, weekday, city, category, bakery and product;
- row-level wins/losses against an explicitly recorded reference model;
- an input SHA-256, target ID and scope ID in `metadata.json`.

The accuracy universe is formed before any future price/cost filtering.
Economics must be calculated on a separately reported eligible subset and must
never silently change the accuracy leaderboard.

## Example

```powershell
.venv\Scripts\python.exe scripts/run_model_tournament.py `
  --input reports/comparable_produced_scope_p50_loss075_20260911/predictions.parquet `
  --date-from 2026-08-01 `
  --date-to 2026-08-31 `
  --target demand `
  --target-id historical_reconstructed_demand_artifact_20260911 `
  --scope-id comparable_produced_scope_p50_loss075_20260911 `
  --sales-column observed_sales_qty `
  --models lag7,mean7,mean14,same_weekday_mean2,weighted_weekday_formula_v1 `
  --prediction-column direct_plan_artifact=direct_plan `
  --prediction-column p50_plan_artifact=p50_plan `
  --reference-model weighted_weekday_formula_v1 `
  --min-coverage 0.70 `
  --output-dir reports/model_tournament_example
```

`--min-coverage 1.0` is the default. Lower values are allowed only for an
explicit diagnostic run. All models are then evaluated on the exact common-key
intersection, and pre-intersection coverage remains visible in `coverage.csv`.

`--lead-days 1` (default) permits target-derived facts only through the day
before each target date. For a forecast issued once for several future days,
set `--history-cutoff` to the last genuinely available fact date. This fixed
origin applies to every evaluation row; otherwise the runner simulates a new
lead-one forecast each day. `lag7` means the exact calendar date `D-7`, not
the seventh previously observed row. `mean7` and `mean14` require all 7 or 14
calendar days ending at the applicable cutoff; missing dates reduce coverage.

`weighted_weekday_formula_v1` uses at most the previous eight same-weekday
calendar dates for each target, takes up to the five most recent available
values, and applies the production weights. It is **only the weighting formula
on the supplied target panel**. It does not recreate production demand
reconstruction, assortment selection, data availability, or publication.

To evaluate freshly fitted candidates, add
`--trainable-model lgbm_quantile_p60` and/or
`--trainable-model lgbm_weighted_residual_v1` with
`--train-end 2026-07-31`, for example. The current trainable adapter supports
daily lead-one evaluation only. Its target-derived features are calculated
from prior calendar dates; training labels end before `--date-from`. Hyperparameters
are fixed in code and are **not** selected on the evaluation period. This is a
diagnostic candidate, not a production-ready model or a complete rolling-origin
backtest.

## Outputs

- `detail.parquet`: canonical row-level target and forecasts;
- `leaderboard.csv`: global metrics recomputed from row-level errors;
- `coverage.csv`: model coverage before the common-scope intersection;
- `wins_losses.csv`: pairwise result versus the reference model;
- `breakdowns/*.csv`: independently recomputed metrics by business dimension;
- `metadata.json`: complete run contract and input fingerprint.

## Interpretation Rules

- Positive bias means overforecast; negative bias means underforecast.
- MAPE excludes zero-actual rows. Zero demand is reported through separate
  false-positive row and quantity metrics.
- Recognized restored loss is a diagnostic quantity, not observed true demand.
- Do not rank sales-target and reconstructed-demand runs in one leaderboard.
- Do not present allocation-only variants as end-to-end forecast models.
- Frozen model artifacts without verified training cutoffs must be labelled
  `historical_frozen_unverified` until leakage can be excluded.
- A lower error for an unverified artifact is not evidence that it would have
  been available at the historical forecast origin.
- Lead-1 snapshots overwritten for reporting are not independent historical
  forecasts.

See [implementation audit](model_tournament_implementation_audit.md) for
specific historical defects and what the new runner does or does not correct.

## Scope audit and snapshot-free research panel

`scripts/audit_tournament_scope.py` checks whether every selected SKU-day had
positive release in the exact prior 56 calendar days and, when given archived
snapshots, counts selected snapshot records created after 08:00 Moscow time.
It performs direct date-range spot-checks on the rolling calculation. Use a
fresh `--output-dir`; the script refuses to overwrite an existing report.

`scripts/build_tournament_causal_scope_panel.py` creates a separate research
panel from the source backtest's prior-activity universe, intersected with
exact prior-56-day production, without archived snapshots. It records hashes
of both inputs and refuses to overwrite an existing output directory. This
checks every source forecast key, including dates with no target-day flow.
It does **not** certify that upstream facts arrived on time, that the source's
prior-activity universe covers the production assortment, or that a daily
lead-one experiment matches the production 14-day forecast contract. See the
audit for the June–August diagnostic and its limitations.

## Next Adapters

The next implementation layers are:

1. independently reconstructable legacy LightGBM v6 sales adapter;
2. routed exp66 P50 demand adapter;
3. Direct raw-preference, common-parent allocation and full-architecture
   adapters as three separate comparison classes;
4. CatBoost residual over weighted weekday;
5. hurdle model for intermittent and tail SKUs;
6. economics simulation on a separately measured common economic scope.

For fixed-origin research, `scripts/run_fixed_origin_tournament.py` evaluates
legacy-objective adaptations (exp01/40/41/43), a corrected P60, new CatBoost,
new hurdle and weekday-residual candidates against the sales-based
weighted-weekday formula over non-overlapping 14-day windows. Every trained
candidate uses the same fixed-origin feature columns and labelled examples.
The output metadata records source lineage and parity limitations. The runner
does not use the daily lead-one runner's target-date scope. See the
implementation audit for results and remaining production-parity gaps.

## Research-only new models (2026-10-02)

The fixed-origin runner now also supports `new_lgbm_hurdle_median`, a
zero-inflated median built from a positive-sale classifier and conditional
positive-sale quantiles. It uses the same tabular training rows and evaluation
keys as the other candidates. Separate local-only runners evaluate a global
univariate N-HiTS (`scripts/run_nhits_fixed_origin.py`) and zero-shot
Chronos-2 Small/Base (`scripts/run_chronos2_fixed_origin.py`). Those neural models
need optional research environments; neither is imported by the production
forecast path. Chronos-2 Small pins the public checkpoint revision in its
runner. The April–June and July–August results and protocol differences are
documented in the [implementation audit](model_tournament_implementation_audit.md).
The Chronos runner defaults to Small; `--model-variant base` selects the
120M-parameter checkpoint, with both revisions pinned in source.

The Chronos runner requires `chronos-forecasting==2.3.2`; the N-HiTS runner
requires `neuralforecast==3.1.2`. Both require PyTorch and use CPU in the
recorded runs. `--max-pairs` is a smoke-test option and must never be used to
claim a full-scope comparable WMAPE.

Chronos-2 Small LoRA fine-tuning is a separate, research-only path in
`scripts/finetune_chronos2_fixed_origin.py` (additionally requires
`peft==0.19.1`). `scripts/rescore_chronos2_lora_checkpoint.py` verifies and
scores a saved adapter against the same frozen panel. Both paths explicitly
put the model in evaluation mode for scoring. Earlier post-fit reports made
while dropout was active are marked invalid; use only the `*_eval_20261002`
reports listed in the implementation audit. Fine-tuning improved the
retrospective sales WMAPE modestly but is not approved for production.

The first frozen forward shadow is documented in the implementation audit.
Its local input and forecast reports are under
`reports/chronos2_lora_forward_shadow_input_20261002/` and
`reports/chronos2_lora_forward_shadow_20261002/`. Only 3–15 October 2026
predictions are prospective; 2 October is excluded because the forecast was
generated after the business day started. The read-only scoring command is:

```powershell
.\.venv\Scripts\python.exe scripts/score_chronos2_lora_forward_shadow.py `
  --shadow-report reports/chronos2_lora_forward_shadow_20261002 `
  --output-dir reports/chronos2_lora_forward_shadow_score_20261017
```

It will refuse to score before 17 October, or if daily fact coverage is low.

For the latest already completed 14 days, run
`scripts/backtest_chronos2_lora_recent.py` with the frozen shadow input,
verified LoRA report, and `--origin 2026-09-17`; the resulting 18 September–
1 October comparison is in the implementation audit. This backtest is
retrospective and does not replace the forward shadow.

## Aggregate demand-target contract

For a future LoRA demand target, keep the raw sales and both adjustment
directions separately at bakery/SKU/day grain:
`adjusted_demand = observed_sales + restoration_uplift - outlier_reduction`.
Both components and the resulting daily target must be nonnegative. A local
target **may be below** that day's sales when a documented, capped outlier is
reduced. On the complete declared training panel, however, the sum of
adjusted demand must be **strictly greater** than the sum of sales. The
validator in `src/model_tournament/demand_target_contract.py` checks row
arithmetic and, only when explicitly called on the full panel, that aggregate
gate. Do not apply the aggregate gate to a batch, one bakery, or one case.

The existing stockout-only research dataset uses upward imputation alone and
therefore happens to stay above sales on every row; that is **not** a rule for
the new general target. The active production `normalized_demand_v1` is a
different smoothed signal; it can locally dip below sales, but its aggregate
effect and suitability as a latent-demand label still need separate audit.

## Causal reconstructed-demand research candidate (2026-10-05)

**Target-alignment correction (2026-10-05):** the sales-WMAPE screens below
measure operational sales-prediction risk, not accuracy against latent demand.
Their failure must not be interpreted as rejection of a demand-training target.
The cross-study audit and the same-scope proxy-demand re-score are in
`docs/research_target_alignment_audit_20261005.md`.

`scripts/build_reconstructed_demand_panel.py` builds a local-only daily panel
from the frozen historical ClickHouse extract. It selects bakery/SKU scope
from positive releases at the fixed training and evaluation origins, densifies
only after each pair's first release, and retains original sales beside the
candidate demand target. The builder uses strictly earlier same-weekday sales
for a reference. Upward restoration additionally requires near sell-through,
early SKU last sale relative to the bakery and the pair's historical closing
hour, and is capped at 50% of that day's sales and 15 units. Downward
adjustment requires an isolated high sale, enough same-product peer bakeries,
no peer-wide surge, and is capped at 20% of sales and 20 units. The separate
uplift/reduction columns allow case-level audit; these are *heuristic* labels,
not observed lost demand. Production's inventory carryover is not modeled in
this research extract, so stockout flags are not proof of censorship.

The full frozen January–August panel in
`reports/reconstructed_demand_full_20261005/` has 6,193 selected pairs,
1,162,855 SKU-days, gross restoration 112,742.694, gross reduction 6,217.46,
and net demand 1.1268% above 9,454,075.02 observed sales. The training
cutoff is 2026-07-05, before the four non-overlapping July–August test
origins. Per-series training-only diagnostics are saved for raw sales, after
restoration, and after reduction. Their R² is a descriptive prequential
prior-weekday fit, not a 14-day holdout score; smoothing is never promoted on
descriptive metrics alone.

As an initial causal falsification check, the same weighted-weekday formula
on exactly 251,972 future SKU-days scored 29.2137% WMAPE with raw sales
history and 29.3599% with reconstructed history, both against unchanged
observed sales (`reports/reconstructed_demand_baseline_eval_20261005/`).
Reconstruction reduced underforecast volume and underforecast days (23 to
20 of 56), but raised overforecast and worsened **sales** WMAPE. Thus it did
not improve this operational sales metric. The separate LoRA experiment in
`scripts/finetune_chronos2_reconstructed_demand.py` needs both the held-out
sales-risk score and a separately validated demand-label evaluation; fitting
the smoother training target alone is not proof of demand accuracy.

The 1,000-step pinned Chronos-2 Small LoRA did improve over zero-shot on the
*same reconstructed input* (25.9109% versus 26.2792% WMAPE), but not over
the verified sales-trained LoRA on exactly the same 251,972 SKU-days
(25.7950%). The demand-trained LoRA reduced aggregate negative bias from
-4.8250% to -2.7112% and underforecast days from 42 to 31 of 56, at the
cost of 0.1160 WMAPE point. On 192,635 heuristic non-stockout SKU-days, its
WMAPE was 27.7831% versus 27.6349% for the sales-trained LoRA. On 107,016
SKU-days from pairs with a reconstructed event in the 56-day input history,
the demand-trained LoRA was marginally better (23.3848% versus 23.4038%);
on otherwise unadjusted pairs, it was worse (35.0747% versus 34.4690%).
These are post-hoc segments, not a licensed routing rule. Full comparison is
in `reports/chronos2_demand_lora_comparison_v2_20261005/`.

Conclusion: the reconstruction candidate satisfies the aggregate-demand
contract and has a measurable bias/underforecast benefit **relative to sales**,
but it has **not** passed the sales-WMAPE improvement gate. Its true-demand
accuracy remains unresolved. Any revised restoration rule or
selective routing must be selected on earlier validation folds and then
tested on a new untouched period; repeatedly tuning against this July–August
holdout would invalidate the comparison.

An additional retrospective 2026-09-01..2026-09-14 fixed-origin check was
performed without retraining either adapter. To avoid turning an incomplete
baker-day into zero sales, the one bakery missing from the frozen outcome
source on 2026-09-13 was excluded on *all* 14 dates. On the remaining 54
bakeries, 4,454 pairs and 62,356 common SKU-days, demand-trained LoRA scored
26.6761% WMAPE versus 26.8791% for sales-trained LoRA, and was better on
10 of 14 days. Its bias was -6.8078% versus -8.2226%. Zero-shot with demand
context also beat zero-shot with sales context (26.9721% versus 27.1370%).
The reproducible report is `reports/chronos2_demand_lora_september_20261005/`.
This is encouraging temporal evidence, but still a retrospective, single
14-day window: historical fact-arrival times and production target/scope
parity have not been verified, and latent demand remains unobserved.
The adapter was also reloaded independently and reproduced 1,400 saved
SKU-day predictions within 0.000012 units; see
`reports/chronos2_demand_lora_full_1000_20261005/checkpoint_verification.json`.

## Inventory-guarded demand audit and early-fold decision (2026-10-05)

The January–August causal panel has daily sales, releases, transfers and
write-offs, but no independent measured opening/closing inventory. A
day-level flow residual can be computed, but it is only a proxy: a positive
prior-day residual may indicate carryover, while current sales exceeding
recorded net supply may reflect old stock or flow-data mismatch. The
research-only `src/model_tournament/inventory_evidence.py` withholds uplift
on either ambiguous case rather than asserting a confirmed stockout. Its
previous-day evidence requires an exact adjacent calendar day, so gaps are
never carried forward. The existing local `mart_zero_sales_60d` pilot export
does have `stock_balance`, but that field equals the same-day arithmetic
residual in all 42,695 exported rows; it is not independent physical stock.
On 38,021 SKU-days matched to the full causal panel, 7,428 sales quantities
disagree by more than one unit. This source should not be treated as
stockout ground truth until the discrepancy is reconciled.

The guard retains 19,220 of the original 24,233 upward-adjusted days and
reduces gross uplift from 112,742.694 to 84,821.174. The guarded full and
half-uplift target variants still satisfy the aggregate-demand-above-sales
contract (+0.8314% and +0.3828%, respectively). Evaluation was restricted to
four non-overlapping pre-July origins, 2026-03-30 through 2026-06-21, with
outcomes ending 2026-07-05. On unchanged observed sales, the weighted-weekday
screen returned:

| Historical target | WMAPE | Clean-subset WMAPE |
| --- | ---: | ---: |
| Raw sales | 29.3135% | 31.7271% |
| Original full reconstruction | 29.4516% | 31.8122% |
| Inventory-guarded full | 29.4095% | 31.7677% |
| Inventory-guarded half | 29.3251% | 31.7096% |

The predeclared **sales-risk** screen required overall WMAPE no worse than raw
sales, clean-subset WMAPE no worse than raw +0.1 percentage point, and no
increase in underforecast volume. **No candidate passed that sales-risk
screen**; this is not a demand-accuracy verdict. No candidate was promoted to
LoRA retraining or a new forward shadow under that policy. The full per-series CV, ACF/PACF,
weekly-seasonality, dominant-cycle, mean/variance stability, and prequential
R² diagnostics are in
`reports/reconstruction_early_folds_20261005/series_diagnostics.parquet`;
small descriptive improvements do not override the forecast gate. Later
July–September windows were already inspected in earlier research and must
not be reused to tune this target or portrayed as untouched confirmation.

A separate early-window pseudo-stockout audit (2026-06-01..2026-07-05)
artificially hid sales after 12:00, 15:00 or 18:00 on source-compatible days
whose recorded sales continued until at least 21:00. The existing restoration
rule identified 70%, 66% and 49% of these eligible synthetic cases, but
recovered only 24%, 42% and 68% of their known hidden units, respectively.
See `reports/reconstruction_synthetic_early_20261005/`. This confirms that
the present target is a conservative regularization, not a reliable estimate
of true lost demand. Synthetic truncation does not prove performance on
natural stockouts, and simply raising its caps could worsen forecast error
or waste; it needs a separately validated decision objective.
