# Model implementation audit (2026-10-01)

This is a code-level audit of the paths named below, not an endorsement of
their historical scores. It is deliberately incomplete: other historical
experiments still require their own source, feature-availability, target,
scope, and training-cutoff review before entering a decision-grade tournament.

| Implementation | Observed defect or risk | Current handling |
| --- | --- | --- |
| `src/experiments_v2/42_asymmetric_loss/run.py` and `src/experiments_v2/52_asymmetric_loss_demand/run.py` | The custom absolute-loss gradient is piecewise constant while the Hessian is set to an artificial constant. This is not the Hessian of that loss. Both experiments are marked broken in `src/experiments_v2/EXPERIMENTS.md` (100% WMAPE in their recorded runs). | Do not replay those scores or custom objectives. `lgbm_quantile_p60` is freshly trained with LightGBM's native quantile objective and `alpha=0.6`, on a blocked holdout. It is a new candidate, not a repaired historical score. |
| `src/models/tune_model.py` | Optuna's objective evaluates each trial on `X_test/y_test`, and `train_best_model` later reports metrics on that same three-day period. The reported tuned test is therefore a selection set, not an untouched test. | The tournament does not import or use those tuned scores. New candidates use fixed hyperparameters and a training end strictly before the evaluation start. A future tuning adapter needs separate train, validation, and untouched test periods. |
| `src/preprocessing.py` | `groupby(...).shift(7)` means seven *observed rows*, not seven calendar days, when a bakery-product has missing dates. Row-count rolling windows have the same calendar ambiguity. | Tournament features are looked up by exact dates and use an explicit fact cutoff. Missing required dates stay missing and lower model coverage rather than silently changing the lag meaning. The original preprocessing pipeline is not changed. |
| `src/experiments_v2/43_quantile/run.py` and `src/experiments_v2/53_quantile_demand/run.py` | Both correctly request LightGBM's native quantile objective. Their scripts train on a single tail split and consume precomputed CSV features; that code does not establish whether every feature and demand label was available at each forecast origin. The reported P50 metrics also use predictions rounded to two decimals. | Do not call these experiments broken. Keep their historical numbers separate from the fresh P60 candidate, and withhold decision-grade claims until their data construction, as-of timing, and repeated-origin performance are verified. |
| Frozen forecast columns in a historical panel | The saved column alone cannot prove when the underlying model was trained or when features, target reconstruction, and scope became available. | Such columns are marked `historical_frozen_unverified` in leaderboard, coverage, and metadata. They can be compared descriptively but are not eligible as evidence of deployable lift until provenance is reconstructed. |
| Weighted-weekday recreation | Reproducing its five-value weighting from an input target is not the same as recreating the production pipeline's calculated-demand target and assortment. A multi-day backtest that consumes future days within the horizon leaks facts. | The runner calls this `weighted_weekday_formula_v1`, has an optional fixed `--history-cutoff`, and records the cutoff rule. Full production parity remains unverified. |
| Mixed target or SKU scope comparisons | Sales and reconstructed demand have different meanings. Restricting to rows with forecasts, production, or prices can create a different evaluation population. | The runner requires explicit `target_id` and `scope_id`, evaluates every model on the same canonical-key intersection, and reports coverage before that intersection. Economics must be a separate eligible subset. The truth and scope's point-in-time provenance still require audit. |

The audit also found that the initial tournament prototype silently clipped
negative saved forecasts and dropped invalid metric rows. Both behaviors have
been removed: invalid target/sales/prediction values now fail visibly.

## Initial August diagnostic (superseded), not a deployment decision

The corrected local run at
`reports/model_tournament_trainable_sales_corrected_20261001/` uses observed
sales as target, training labels through 2026-07-31, daily lead-one forecasts
for 2026-08-01 through 2026-08-31, and the common 82,894 SKU-day intersection
of a historical produced-pair panel. WMAPE is 22.66% for fresh P60 quantile,
23.40% for fresh weighted-weekday residual, 24.45% for exact-calendar mean14,
and 24.62% for weighted-weekday formula. Mean14 covers only 79.13% of the
unintersected evaluation rows, so the common leaderboard is not the entire
panel. A later scope audit found that this panel uses selected-latest snapshot
assortment records, including records created after the morning forecast
cutoff. These August numbers are retained for traceability but superseded by
the snapshot-free research runs below. They must not be compared directly to
production or to a reconstructed-demand leaderboard.

## Scope and snapshot timing audit

`scripts/build_comparable_produced_scope_backtest.py` calculates its claimed
56-day production eligibility using `shift(1).rolling(56)` on observed rows.
That is not a 56-*calendar-day* condition when a pair has gaps. The independent
calendar audit at `reports/model_tournament_scope_audit_timing_20261001/`
found 910 of 735,163 selected rows without positive production in D-56..D-1,
including 435 August rows. It also checked 25 keyed rows with direct date-range
sums. An initial version of the auditor incorrectly matched grouped rolling
results by position; the direct spot-check caught it, and the final audit
joins rolling results by canonical SKU-day keys.

The selected-latest snapshot file has a separate timing problem. Among rows
actually selected by the comparable panel, the stored snapshot record was
created after 08:00 Moscow time on its forecast date for **85,296/85,296 June**
rows, **45,087/94,050 July** rows, and **22,810/80,621 August** rows. A late
record does not prove what scope was available by 08:00. An early record still
does not independently prove feature or source-fact availability. The
selected-latest snapshots therefore remain unsuitable as point-in-time scope
evidence for a decision-grade comparison.

## Snapshot-free monthly diagnostic

`scripts/build_tournament_causal_scope_panel.py` instead uses the source
backtest's prior-activity universe and intersects it with exact positive
production in D-56..D-1, without snapshots. Its frozen panel and SHA-256
provenance are in `reports/model_tournament_causal_scope_panel_v2_20261001/`.
The first version of this builder was itself wrong: it calculated scope only
for dates with a flow-panel row on the target day, thereby excluding many
zero-activity forecast days **using the outcome**. Its 753,804-row panel and
all reports named `model_tournament_snapshot_free_*` are superseded. The
corrected builder requests historical eligibility for every source forecast
key, including dates absent from the flow table, and produces 1,025,190
SKU-days. This is guarded by a zero-activity-day regression test.

Fresh candidates were fit once per month, using labels only through the prior
month and daily lead-one features. Every model in a month was scored on the
same complete research-panel SKU-day keys, with 100% candidate coverage.
`mean14` is excluded from this primary comparison because it cannot forecast
roughly 17–20% of those rows without filling absent calendar history.

| Sales WMAPE | June (128,575 rows) | July (136,310 rows) | August (144,207 rows) |
| --- | ---: | ---: | ---: |
| LGBM weighted-weekday residual | 26.50% | 26.06% | 26.91% |
| LGBM P60 quantile | 26.58% | 25.68% | 25.66% |
| Weighted-weekday formula | 28.75% | 28.50% | 29.42% |

The residual candidate is ahead by only 0.08 percentage points in June; P60
is ahead by 0.37 points in July and 1.25 points in August. The ranking changed
after zero-activity days were restored, showing why scope validation must
precede model selection. These remain research diagnostics: upstream flow
arrival times and the source's prior-activity universe are not independently
proven as-of, and daily lead-one scoring is not the production 14-day forecast
contract. No production recommendation follows from this table.

## Fixed-origin 14-day research backtest

`scripts/run_fixed_origin_tournament.py` freezes the SKU scope on each
forecast origin using only positive production in the preceding 56 calendar
days. It creates all 14 future SKU-days for every eligible pair, including
days with no sales. Its sales-derived features are computed only through that
origin. The future sales join supplies truth **after** the features and scope
are fixed. Tests perturb future truth and confirm that the forecasts' inputs
do not change. The script rejects overlapping evaluation windows and any
training label date that reaches the evaluation period.

The run at `reports/model_tournament_fixed_origin_h14_20261001/` trains on six
historical origins (February–June), with the last training target on 5 July,
then scores four non-overlapping 14-day releases issued 6 and 20 July, 3 and
17 August. The common evaluation has 251,972 SKU-day forecast instances.

| Sales WMAPE | Overall | 6 Jul | 20 Jul | 3 Aug | 17 Aug |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh LGBM P60 | 27.45% | 26.17% | 27.58% | 28.05% | 27.93% |
| Fresh LGBM weighted-weekday residual | 27.92% | 25.75% | 26.77% | 28.52% | 30.28% |
| Weighted-weekday formula on sales | 32.28% | 31.09% | 32.29% | 32.75% | 32.90% |

P60 is better overall and on both August origins; residual is better on both
July origins. P60's bias is +3.59% overall, while residual's is -3.21%, so
accuracy alone is not an economic decision. At lead 1, P60 WMAPE is 25.25%;
at lead 14 it is 28.09%, demonstrating horizon degradation that the earlier
daily lead-one exercise could not measure.

This remains a **research** sales-target contract, not production parity:
the production model uses reconstructed demand and a different assortment
source; historical source-fact arrival times and revisions are not verified;
and these July–August dates have already informed candidate development, so
they are not a pristine model-selection holdout. Do not activate a model from
this result.

## Expanded fixed-origin tournament: legacy and new architectures

The first 14-day run above was a contract smoke test, **not** a comprehensive
model tournament. Its P60-first conclusion is superseded by the expanded run
at `reports/model_tournament_fixed_origin_expanded_20261001/`. All nine rows
share the same 354,186 training and 251,972 evaluation forecast instances,
the same fixed-origin features, and the same four July–August releases.

| Candidate | Type and lineage | Sales WMAPE | Forecast bias |
| --- | --- | ---: | ---: |
| LGBM P50 | Adapted exp43 native quantile | **26.81%** | -4.33% |
| LGBM Tweedie | Adapted exp40, power 1.5 | 26.99% | -1.25% |
| CatBoost MAE | New project candidate | 27.13% | -4.53% |
| LGBM log-target | Adapted exp41 `log1p/expm1` | 27.25% | -7.92% |
| LGBM hurdle mean | New two-stage nonzero/positive-sales candidate | 27.40% | +0.03% |
| LGBM P60 | Native-quantile correction of broken exp42 idea | 27.45% | +3.59% |
| LGBM L2 | Adapted exp01 objective | 27.55% | +0.12% |
| LGBM weekday residual | New project candidate | 27.92% | -3.21% |
| Weighted weekday | Sales-only formula baseline | 32.28% | +4.47% |

These are **objective/architecture adaptations**, not replays of the original
exp01/40/41/43 models: the original files have no saved model artifacts here,
and their feature sets, short test split, and historical scopes differ. The
expanded metadata records the source and parity level for every candidate.
P50 wins aggregate WMAPE but underforecasts by 4.33%; Tweedie is closer to
balanced, while the hurdle's near-zero total bias does not imply calibration
or better SKU-level decisions. Per-origin rankings also vary. There is no
production recommendation yet.

Exp42/52 custom asymmetric objectives remain excluded as broken; P60 is their
corrected *idea*, not the old implementation. Exp44 mixture-of-experts and
exp45 in-sample residual correction require separate routing/out-of-fold
audits before a faithful adapter. Exp66 routed demand clusters and exp67/68
SKU/Prophet specialists require their own as-of feature construction, target,
and SKU coverage contracts; simply appending their reported metrics would
mix incompatible test populations.

Before any production recommendation, rerun candidates at several historical
forecast origins with frozen as-of inputs, a documented target, a dated
assortment, a common SKU-day population, untouched test windows, uncertainty
or interval calibration, and separately measured economic effects. No
production activation or publication is part of this audit.

## New-model tranche (2026-10-02): hurdle median, N-HiTS, Chronos-2

Three further research candidates were evaluated on the same frozen
`causal_panel.parquet` SHA-256 and the exact same SKU-day keys and observed
sales as the adapted exp43 P50 comparator. For both seasonal panels, key
equality and zero maximum actual-value difference were checked. All runs are
local read-only reports and forecast 14 days from a fixed origin.

| Candidate | Spring Apr 28–Jun 22 (235,326 rows) | July 7–Aug 31 (251,972 rows) |
| --- | ---: | ---: |
| Adapted exp43 LGBM P50 | 27.44% | 26.81% |
| New LGBM zero-inflated hurdle median | 27.46% | 26.65% |
| New global univariate N-HiTS | 27.46% | 26.26% |
| Chronos-2 Small, zero-shot P50 | **26.79%** | **26.10%** |
| Chronos-2 Base, zero-shot P50 | not run | 26.13% |

The spring origins are 27 April, 11 May, 25 May, and 8 June. P50, hurdle,
and N-HiTS spring training ends on 16 March. The July–August origins and
training cutoff are the expanded tournament's 6/20 July and 3/17 August,
with the last local training label on 5 July. Chronos-2 uses no local
training; for each origin it receives only the previous 56 calendar days of
observed sales. Its checkpoint is pinned to
`autogluon/chronos-2-small@ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a`.
The 120M-parameter Chronos-2 Base was also run on every July–August key,
using `amazon/chronos-2@29ec3766d36d6f73f0696f85560a422f50e8498c`;
it was 0.04 percentage points worse than the 28M-parameter Small model.

The hurdle median combines a positive-sale classifier with positive-sale
conditional quantiles. It has only a small July–August advantage over P50
and loses slightly in spring. N-HiTS is trained globally on dense daily
observed-sales series, using 100 CPU training steps; unlike the tabular
models it uses all causal rolling windows through the training cutoff, not
only supervised rows from the six or two training origins. It wins in summer
but is essentially tied with P50 in spring. Chronos-2 Small is best overall
in both periods and wins seven of eight origin-level comparisons against
P50, but on the 17 August origin it still underforecasts by 11.58% despite
its lower WMAPE. This makes total bias and business costs essential checks.

Report directories:

- `reports/model_tournament_hurdle_median_20261002/` and
  `reports/model_tournament_hurdle_median_spring_20261002/`;
- `reports/model_tournament_nhits_20261002/` and
  `reports/model_tournament_nhits_spring_holdout_20261002/`;
- `reports/model_tournament_chronos2_full_20261002/` and
  `reports/model_tournament_chronos2_spring_20261002/`;
- `reports/model_tournament_chronos2_base_full_20261002/`.

All figures above are **retrospective observed-sales WMAPE**. The historical
source-fact arrival times and revisions, production assortment and target
parity, regional rollout scope, and economic outcomes remain unverified.
April–August was already used for project research; these are robustness
diagnostics rather than a pristine prospective holdout. Chronos-2 should
enter a frozen forward shadow trial, not production activation, on this basis.

## Chronos-2 Small LoRA fine-tuning (2026-10-02)

The research-only `scripts/finetune_chronos2_fixed_origin.py` fine-tunes the
pinned Small checkpoint with the library's LoRA mode (learning rate `1e-5`,
batch size 16, context 56 days, horizon 14 days, seed 42). It trains on dense
daily observed-sales series, filling absent sparse fact rows with zero, and
never uses labels from an evaluation window. The spring run has 2,805 eligible
series and a 16 March label cutoff. Its 100/500/1,000-step sweep was used to
choose 1,000 steps. The summer run has 4,802 eligible series, a 5 July label
cutoff, and uses the spring-selected 1,000-step setting. Each seasonal
comparison uses the **exact same SKU-day keys and actuals** as its zero-shot
score: 235,326 spring and 251,972 summer rows (zero key or actual mismatch).

| Candidate | Spring observed-sales WMAPE | July–August observed-sales WMAPE |
| --- | ---: | ---: |
| Chronos-2 Small zero-shot P50 | 26.7916% | 26.0952% |
| Chronos-2 Small LoRA, 100 steps | 26.7337% | not run |
| Chronos-2 Small LoRA, 500 steps | 26.4927% | not run |
| Chronos-2 Small LoRA, 1,000 steps | **26.3621%** | **25.7950%** |

The 1,000-step adapter improves summer WMAPE by 0.3002 percentage points,
but worsens aggregate forecast bias from -4.19% to -4.82%; its 17 August
origin bias is -12.29%. A small average accuracy gain is therefore not an
operational adoption decision.

An implementation error was caught during verification: the library's
`fit()` returns a model still in training mode, and `predict_df()` does not
switch it to evaluation mode. Initial post-fit forecasts were stochastic and
disagreed with the saved adapter. All four initial LoRA score reports are
marked `score_valid: false` and superseded. The corrected runner explicitly
calls `model.eval()`, and `scripts/rescore_chronos2_lora_checkpoint.py` loads
the adapter onto the pinned base model via PEFT before deterministic scoring.
The authoritative corrected reports are:

- `reports/model_tournament_chronos2_lora_spring_100_eval_20261002/`;
- `reports/model_tournament_chronos2_lora_spring_500_eval_20261002/`;
- `reports/model_tournament_chronos2_lora_spring_1000_eval_20261002/`;
- `reports/model_tournament_chronos2_lora_summer_1000_eval_20261002/`.

These are retrospective, **observed-sales** results. Origin-day history may
not have existed at a production 08:00 run; historical fact-arrival timing,
production target and assortment parity, and economic impact remain
unverified. The project had already used April–August in model-family
research, so summer is not a pristine prospective holdout. Keep the adapter
research-only until an as-of-correct, frozen forward shadow trial establishes
value and guards against underforecast.

## Frozen forward shadow initiated 2 October 2026

The local-only forward shadow freezes current ClickHouse sales/release facts
through 1 October, restricted to the original **55 pilot bakeries** in the
research panel. The extraction ran 2 October 12:41 UTC, after the 2 October
business day had started. Thus 2 October is deliberately excluded; only
3–15 October target dates are preserved as prospective predictions. The
research scope has 4,568 bakery/SKU pairs, yielding 59,384 forecast SKU-days
for each Chronos contender. No future actuals were read and no forecast was
published or activated.

On those same 59,384 keys, zero-shot totals 612,493.05 and the selected
1,000-step LoRA adapter totals 596,386.09 observed-sales units (2.63% lower).
This is **not an accuracy result**. The currently active production run at
freeze was `draft_normalized_regions_20261002_h14`, using the normalized-demand
weighted-weekday model. It overlaps on only 43,966 SKU-days; its target and
scope differ, so total-volume comparison against it is not a model ranking.
The frozen extraction also cannot certify historical 08:00 fact availability
or an atomic multi-query database snapshot.

Files and reproducible entry points:

- `reports/chronos2_lora_forward_shadow_input_20261002/` contains frozen
  facts, the active-run slice, timestamps, and input hashes;
- `reports/chronos2_lora_forward_shadow_20261002/` contains the immutable
  prospective predictions and metadata;
- `scripts/freeze_chronos2_forward_shadow.py` and
  `scripts/run_chronos2_lora_forward_shadow.py` create those local files;
- `scripts/score_chronos2_lora_forward_shadow.py` is fail-closed until
  **17 October 2026**, at least one full day after the final target date.
  It requires at least 50 of the 55 bakeries to have sales facts on every
  target day, freezes actuals, scores LoRA against zero-shot on identical
  SKU-day keys, and separately scores all models on the incumbent overlap.

The scheduled evaluation must inspect WMAPE **and** aggregate/daily bias;
production adoption still requires target/scope parity and economic review.
An in-chat daily follow-up named `Chronos-2 LoRA: итог forward-shadow` is
active; it stays quiet before 17 October, retries only if future facts are
incomplete, reports the completed comparison, then disables itself.

## Most recent completed 14-day retrospective check (2026-09-18..2026-10-01)

The user's requested fast backtest uses a **single fixed origin, 17 September**,
with 56 calendar days of observed-sales context ending at that origin and
sales labels only on 18 September–1 October. It reuses the already frozen
2 October read-only fact extraction and the pre-origin LoRA adapter trained
through 5 July; no retraining or selection was done on September outcomes.
All three candidates have identical 64,848 SKU-day keys and actuals: 4,632
bakery/SKU pairs in 55 bakeries. Outcome fact coverage is 55 bakeries daily.

| Candidate | WMAPE | Aggregate bias |
| --- | ---: | ---: |
| Chronos-2 Small LoRA, 1,000 steps | **24.2303%** | **-3.9399%** |
| Chronos-2 Small zero-shot | 24.5308% | -1.8970% |
| Weighted-weekday observed-sales formula | 30.8746% | +2.8512% |

LoRA reduces WMAPE by 0.3005 percentage points versus zero-shot and wins
10 of 14 dates and 43 of 55 bakeries on absolute error. Yet its aggregate
underforecast is 2.04 percentage points worse, with daily bias -11.12% on
18 September and -10.23% on 25 September. These are competing operational
signals, not evidence for automatic adoption. The report is
`reports/chronos2_lora_recent_14d_20260918_20261001/`, created by
`scripts/backtest_chronos2_lora_recent.py`.

The model never sees future labels during inference, but the 2 October fact
snapshot may contain late-arriving or revised facts that were unavailable
at the historical 17 September origin. That 08:00 as-of parity is **not
verified**. The target is observed sales, not the active production model's
normalized-demand target; the population covers the 55 research bakeries,
not the full 248-bakery serving scope. This is one recent 14-day window, not
an independent multi-origin prospective validation. The frozen forward
shadow above remains necessary.
