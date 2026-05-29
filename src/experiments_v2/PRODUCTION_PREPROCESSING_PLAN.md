# Production preprocessing plan

## Chosen modeling split

Current target architecture after reviewing the Yandex Lavka demand forecasting
talk and running exp79/exp80:

`LGBM bakery-level top-down with enriched event features -> SKU/hour split`

We keep the bakery-driven top-down backbone already implemented in this repo:

1. forecast the bakery-level daily target with one LGBM top-down model;
2. split bakery-day to bakery-hour using bakery hour profiles;
3. split bakery-hour to SKU-hour using SKU hour-share profiles;
4. evaluate the resulting plan with planning metrics, not only MAE.

Residual correction / ensemble architecture is not the selected production
direction right now. Exp79 showed that a global correction layer over the
current LGBM top-down baseline worsened MAE, WMAPE, and large-error share.
That line stays in research backlog only for future gated correction work.

The selected improvement path is to strengthen the single top-down LGBM with
calendar and event context. Exp80 showed that enriched event features improved
7/14-day quality materially and slightly improved 30-day MAE/WMAPE.

Production feature set now includes explicit event/payday features:

- holiday name and event-window type;
- pre/post event flags;
- event distance bins;
- event x city and event x weekday interactions;
- payday distance and payday-window features.

## Planning metrics

MAE remains a diagnostic metric, but it is no longer the primary decision rule.
Experiment comparisons should report:

- bias by aggregate planning level, especially `city x category` and
  `city x product`;
- WMAPE by the same aggregate levels;
- share of material errors where both conditions hold:
  - absolute error is greater than `50` units;
  - relative absolute error is greater than `20%`;
- direction of material errors:
  - underforecast risk as a proxy for availability / stockout pressure;
  - overforecast risk as a proxy for waste / overproduction pressure.

Reusable implementation:

- `src/experiments_v2/planning_metrics.py`

The production path should use a two-layer target strategy:

1. **Base model**
   - learns regular bakery-level demand;
   - should be robust to rare spikes, holidays, and special events;
   - may downweight contextual anomalies instead of learning them as normal demand.

2. **Correction layer**
   - learns systematic deviations from the base forecast;
   - can use holidays, events, and other operational context;
   - should be evaluated separately from the base model.

This is preferred over a single all-features model because it keeps the core
forecast stable and makes event uplift/downlift auditable.

## Outlier handling

Product moves are not reflected in the check lines used to build sales, so they
must not be subtracted from sales or used as sales-outlier context in the
production preprocessing layer.

The preprocessing layer should therefore:

- preserve observed sales;
- flag sales outliers from the observed sales series;
- split high outliers into:
  - `contextual_high_outlier_flag`
  - `unexplained_high_outlier_flag`
- create a separate base-training target where residual unexplained outliers
  are capped against the robust expected baseline;
- use `base_model_sample_weight` for baseline training rather than deleting rows;
- expose `correction_candidate_flag` for the correction model.

The intended target sequence is:

`observed_sales -> base_capped_sales`

`observed_sales` remains the factual audit column. `base_capped_sales` is only
for the regular base model and should not replace the fact table.

## Event handling at base layer

The base layer does **not** use explicit event/holiday flags to treat or remove
holiday rows. The reasoning:

- the current rolling quantile cap is expected to absorb rare event spikes
  automatically — a Christmas peak that is far above q95 in its bucket will be
  clipped to q95 and never enters the base target;
- adding an explicit event mask would couple the base model to a hand-curated
  calendar and create a second source of truth alongside the cap.

This is an explicit trade-off and depends on the assumption that any single
event repeats only once or twice in current history. With `~16` months of
training data this is true. When history reaches `2+` annual cycles, repeated
events will start drifting `q95` upward and stop being treated as outliers —
at that point base preprocessing must be revisited:

- either lower `upper_quantile`,
- or add an explicit event mask that downweights repeating holidays before
  quantile estimation.

To monitor this empirically, the cleaning audit reports a
`holiday_hit_rate`:

`holiday_hit_rate = |dates with sales_high_outlier_flag == 1 ∩ known holidays| / |known holidays|`

Reading guide:

- `hit_rate > 30%` — most known holidays are still flagged as outliers, so the
  cap is still doing event handling on its own;
- `hit_rate < 10%` — known holidays are already absorbed into the `q95`
  baseline and are no longer treated as outliers; this is the signal to add
  explicit event handling.

## Capping recipe

The chosen production target is weekday-aware trailing rolling quantile
capping:

- estimate `q05` and `q95` by `bakery_id x dow` from the trailing window of
  `26` same-weekday observations (~6 months) with `min_periods=8`;
- shift by one row so today's value never defines its own cap;
- fall back to expanding quantile when the rolling window is thin;
- the cap moves with trend and slow seasonality, while the long window keeps
  the quantile estimate stable;
- preserve contextual high outliers by default so they can be handled by the
  correction layer.

This variant was selected over the alternatives based on exp78 results across
7/14/30-day backtests:

- `rolling_quantile_capped_target_lgbm` wins on 7-day (MAE 142.65 vs 150.68
  baseline, -5.3%);
- comparable to global quantile on 14/30-day, with materially better bias
  (-0.47 vs -0.89 on 30d);
- locally tighter cap means outlier rows ~19% of dataset versus ~11% for the
  static global quantile.

Two alternatives are kept available as benchmarks but not default:

- `add_quantile_capped_base_target` (static `q05/q95` over the entire history
  per `bakery_id x dow`) — best aggregate MAE on 30d but ignores trend;
- `add_rolling_median_capped_base_target` (rolling median with multiplier
  caps) — consistently the weakest in exp78.

Initial implementation lives in:

- `src/experiments_v2/sales_cleaning.py`

## Integration order

1. Add robust bakery-day outlier flags and base training weights.
2. Add weekday-aware quantile-capped base target.
3. Build cleaning audit reports under `reports/sales_cleaning_audit/`.
4. Rebuild bakery-level backtests with and without sample weights.
5. Apply the same philosophy to share profiles:
   - do not use anomalous days blindly in share means;
   - downweight unstable/contextual days;
   - maintain fallback profiles.
6. Only after the base layer is stable, train correction models using event
   signals.

## Share layer direction

The SKU-hour share should balance recency and stability:

- recent profile: higher responsiveness;
- long profile: stability;
- fallback chain gated on raw sample size (`n_days`), not on a composite
  reliability score:
  - `bakery x sku x dow x hour` when `n_days >= 8`;
  - else `bakery x sku x hour`;
  - else `city/category/sku x dow x hour`;
  - else global.

Sample-quality columns (`n_days`, `zero_share_rate`, `cv_share`,
`anomaly_share`) and their aggregate `reliability_score` are produced by the
profile builder, but they are **observational metrics** for audit dashboards
only. A composite score cannot be validated without a production A/B, so it
should not drive routing or uplift behaviour — those decisions stay on simple,
isolated thresholds we can move one at a time.

The uplift rule:

- if an observed/current SKU share is below its historical mean share, lift it
  toward the mean (`max(observed, mean)`);
- renormalize within `date x bakery_id x hour`;
- track the uplift amount as an auditable column.

No reliability gate on the uplift in the current iteration. If production
backtests show the uplift damaging accuracy in specific regimes, add a simple
guard on `n_days` or `zero_share_rate` then — not a composite score.
