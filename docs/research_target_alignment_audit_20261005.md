# Audit of training targets and evaluation labels (2026-10-05)

This is a read-only research audit. It does not change any model, production
forecast, active run, or historical score file. The immediate reason is that
some demand-trained candidates were ranked against *observed sales* and the
sales ranking was subsequently interpreted as a verdict about demand quality.

## Decision rule

`WMAPE(prediction, observed sales)` is a valid measure of sales-prediction
accuracy and an operational risk diagnostic. It is **not** an estimate of
`WMAPE(prediction, true demand)` when availability can censor sales. Conversely,
`WMAPE(prediction, reconstructed demand)` describes fit to a *proxy* until the
reconstruction is independently validated. Comparing models trained on
different labels is legitimate only when every model is scored on the **same
explicitly named** test label and common SKU-day scope. Training on one label
and scoring on another can answer a downstream question, but cannot establish
accuracy for the training label.

The user's demand contract is aggregate, not pointwise: reconstructed demand
may be below a particular day's sales after justified outlier reduction, but
must exceed sales on the declared complete panel in total.

## Findings by research family

| Research family | Training / historical signal | Actual evaluation label | Correct interpretation and status |
| --- | --- | --- | --- |
| Earlier sales-only LightGBM experiments, fixed-origin model tournament, sales Chronos-2/N-HiTS | Observed sales | Observed sales | Target-aligned **for sales**. Their leaderboard does not rank latent-demand quality or establish parity with the production calculated-demand target. |
| Legacy demand experiments `03`, `50–56`, `60` and successors using `daily_sales_8m_demand.csv` | `Спрос` proxy (and demand lags) | `Спрос` proxy | **Not** the sales-label mismatch. Their numbers rank models against that proxy, not true demand. The profile builder constructs pair profiles from all available hourly dates before the later train/test split; as-of validity and independence of the test proxy are not established. A tail split with precomputed demand lags is not automatically a 14-day fixed-origin backtest. |
| Bakery target-cleaning experiment `78` | Raw or capped/cleaned bakery target, depending on variant | `actual_sales` (`bakery_sales_observed` when present) | Its 9.869–10.095% WMAPE table is a **sales** comparison of training policies, not evidence that the capped series is closer to true bakery demand. |
| `reconstructed_demand_baseline_eval`, `reconstruction_early_folds`, weekly bridge, unexplained dips, hierarchical bridge, volatility guard | Restored/regularized historical proxy versus raw history | Unchanged observed sales | These are sales-risk and retrospective predictability screens. A variant failing their sales WMAPE gate has **not** thereby failed a demand-forecasting test. CV, ACF, PACF and prior-weekday R² are descriptive, not independent demand accuracy. |
| Full reconstructed-demand LoRA, guarded-half LoRA, July–August and September comparisons | Reconstructed or guarded demand proxy | Unchanged observed sales | Their published WMAPE, bias and underforecast are all **relative to sales**. The sales results remain valid; any conclusion that the demand-trained model is worse at predicting demand is unsupported. This includes the guarded 2×2 comparison. |
| `model_tournament_mvp_corrected_20261001` | Recomputed baselines from the historical reconstructed-demand artifact; other candidates are frozen forecasts | Historical reconstructed-demand artifact | This leaderboard is at least **target-labelled** as proxy demand. It does not validate that the proxy is true demand; frozen model provenance and point-in-time availability remain explicitly unverified. Do not mix it with sales-target leaderboards. |
| P50/direct strategy and unified/economic simulations | Various forecast/plan signals | Simulated service, loss and profit using reconstructed demand (with observed checkout for some factual baselines) | These are **scenario results conditional on the demand proxy and inventory/economic assumptions**, not observed counterfactual profits or independent demand labels. They are not directly invalidated by the sales-WMAPE mismatch, but need sensitivity/label validation before promotion. |
| Early synthetic stockout audit | Restoration rule | Known sales artificially hidden after a chosen hour | Provides partial, independent *synthetic* recovery evidence. It does not validate loss on naturally censored days. |
| Active normalized-demand production forecast | Distinct production normalization | Operational sales/serving metrics | Production state is unchanged. This audit does not establish whether the active signal predicts true demand; a separate, versioned operational review is required before any deployment decision. |

In legacy experiment `03`, the archived scores are 29.45% WMAPE for the
sales-trained model and 28.19% for the demand-trained model **against the same
`Спрос` proxy**. This avoids the sales-label mismatch. However, model A uses
41 features and model B uses 52, including additional demand lags, so this
comparison also does not isolate *training target alone*. The proxy has a
reported +14.92% aggregate uplift over sales and 45.1% censored rows, making
its provenance particularly important.

## Concrete ranking reversal on the same held-out SKU-days

For the 251,972 July–August SKU-days, the frozen raw-sales LoRA and full
reconstructed-demand LoRA were re-scored without changing predictions. Keys and
observed sales matched exactly; every row had a proxy-demand label in the
frozen panel. The held-out proxy total is 2,272,613.628 versus 2,240,854.27
observed sales (+1.417%). 7,853 of the 251,972 test labels differ from sales.

| Model | WMAPE to observed sales | WMAPE to reconstructed-demand proxy | Bias to proxy |
| --- | ---: | ---: | ---: |
| Raw-sales LoRA | **25.795%** | 24.982% | -6.155% |
| Full reconstructed-demand LoRA | 25.911% | **24.800%** | -4.071% |

Thus the ranking **reverses when the label changes**. This directly confirms
that the previous sales-only ranking cannot decide the demand question. It does
**not** validate the demand LoRA as a true-demand winner: the held-out proxy is
generated by the same reconstruction approach used for training, so its use as
a final success criterion would be partly circular.

The reproducible read-only audit in `scripts/audit_research_eval_targets.py`
also re-scored **38 models/variants in nine same-scope reports** against both
labels, with exact sales-truth checks. Full results and source hashes are in
`reports/research_target_alignment_audit_20261005/dual_target_scores.csv` and
`metadata.json`. Two report-internal choices visibly reverse:

| Report / candidate | Sales WMAPE | Same-method demand-proxy WMAPE |
| --- | ---: | ---: |
| Weighted weekday, raw history | **29.214%** | 27.916% |
| Weighted weekday, reconstructed history | 29.360% | **27.821%** |
| Early-fold raw history | **29.313%** | 28.040% |
| Early-fold inventory-guarded half | 29.325% | **27.980%** |

These reversals show why a sales gate cannot settle a demand objective. They
remain proxy-concordance results, **not** independent latent-demand validation.

## Corrections to previous conclusions

1. Preserve all observed-sales metrics and reports, but relabel them as
   `sales_evaluation` or `operational_sales_risk`; do not erase them.
2. Withdraw the claim that failing the sales WMAPE screen disqualifies a
   reconstruction as a demand-training target. It may still be rejected for
   operational sales risk, but demand accuracy remains **unresolved**.
3. Treat older demand-proxy leaderboards as proxy-concordance evidence only.
   The legacy all-date profile construction must be frozen per origin before
   their numbers are used for point-in-time model choice.
4. Do not compare WMAPE numbers from different target IDs or SKU scopes in a
   single ranking. Do not use post-hoc smoother labels as independent proof
   that the same smoother is right.

## Required re-evaluation contract

For each preserved prediction artifact, freeze `model_id`, `train_target_id`,
`context_target_id`, `evaluation_target_id`, `scope_id`, origin/horizon,
fact-arrival cutoff, and label provenance. Score *all* candidates on the same
common SKU-days for each evaluation label; keep sales and demand-proxy columns
as separate leaderboards.

For a credible demand evaluation, make an **independent test adjudication**:
on high-confidence unconstrained days, sales are a usable demand observation;
on likely constrained days, use source-reconciled intraday sales, production,
transfers and write-offs to form confidence tiers or lower/upper bounds. A
two-hour zero-sales gap alone must not label a low-frequency SKU as absent.
Where demand remains unobserved, report bounds/coverage and abstain from a
single exact WMAPE. Preserve synthetic censoring as a separate known-tail
test. Construct labels after outcomes for offline evaluation if necessary,
but never expose future outcomes or adjudication to a forecast's as-of input.

Only after that independent contract is frozen should the demand-trained and
sales-trained models, other historical candidates, and the incumbent be
re-ranked on their exact common eligible intersection. Economic scenarios
remain a separate, proxy-sensitive decision layer.

## Scope and evidence

This first-pass inventory inspected 92 top-level report `metadata.json` files,
48 experiment `metrics.json` files, and the decision-bearing code paths
below. It is **not** a certification of every older report directory: many
historical artifacts lack explicit target metadata. Such artifacts remain
`target_unverified`, not implicitly valid or invalid, until their source code,
input hashes, keys and score computation are traced. No production comparison
or prospective natural-stockout ground truth was established in this audit.

- `scripts/finetune_chronos2_reconstructed_demand.py`: training target is
  `reconstructed_demand_qty`, primary evaluation target is original sales.
- `scripts/finetune_chronos2_volatility_guard.py` and
  `scripts/evaluate_chronos2_volatility_guard_factorial.py`: guarded training
  and context, all four cells scored to sales.
- `scripts/evaluate_reconstruction_early_folds.py`,
  `scripts/evaluate_reconstructed_demand_baseline.py`, and the weekly-bridge/
  volatility scripts: candidate histories, unchanged sales truth.
- `src/experiments_v2/03_demand_target/run.py` and
  `src/experiments_v2/53_quantile_demand/run.py`: both use `Спрос` for the
  demand score; `03` also reports sales separately.
- `src/experiments_v2/03_demand_target/build_demand_profiles.py`: profiles
  are built from all full days before the downstream split.
- `src/experiments_v2/78_bakery_target_cleaning/run.py`: varied training
  targets, `actual_sales` evaluation.
- `reports/model_tournament_mvp_corrected_20261001/metadata.json`: explicit
  historical reconstructed-demand target ID.
- `scripts/run_unified_architecture_tournament.py` and
  `scripts/evaluate_historical_economic_gate.py`: reconstructed demand enters
  simulation, with some observed-checkout baselines.
