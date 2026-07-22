# Demand-adjusted SKU profile experiment — 2026-07-22

## Goal

Test the proposed separation between two stockout mechanisms:

1. allocation failures are handled by allocation logic; and
2. missing bakery demand is handled by reconstructing demand before profiles
   and training targets are built.

This is an offline experiment. It does not write to production.

## Input selection

The earlier prototype used only 397 cases where the historical model was
already proven to underforecast. That is inappropriate for demand-history
preprocessing because the old forecast must not determine whether observed
sales were censored.

The expanded input contains all 1,296 clear-stockout SKU-days from 2026-06-01
through 2026-07-19. The run-time-independent bakery-volume classifier produced:

- 705 robust allocation cases;
- 83 robust demand-loss cases;
- 508 uncertain cases.

Following the agreed inverse rule, preprocessing candidates are all 591 cases
that are not robustly classified as allocation.

## Reconstruction

- Cases with positive reconstruction: 581/591.
- Total reconstructed demand: 3,775.16 units.
- Median per case: 4.94 units.
- Maximum per case: 20 units.
- At least three reference days: 98.3% of cases.

For the temporal profile experiment, training ends on 2026-07-05 and holdout
is 2026-07-06 through 2026-07-19. The train portion contains 430 adjusted
SKU-days, 232 bakery/SKU pairs, and 2,758.38 reconstructed units.

## Profile construction

Imputed post-stockout SKU-hour rows are inserted before calculating bakery-hour
totals and SKU shares. Thus both the SKU numerator and bakery denominator
describe reconstructed demand rather than observed sales.

The A/B reproduces production serving rules:

- exact tier-1 requires `n_days >= 8`;
- thin contexts use the bakery/hour fallback;
- tier-1 shares are renormalized after gating.

## Routing discontinuity

Demand reconstruction added 90 tier-1 SKU rows across 66 existing exact
bakery/day-of-week/hour contexts. These additions were useful on holdout.

It also created one entirely new exact context. Switching that whole context
from fallback to exact sharply worsened its clean-SKU WAPE. The guarded variant
therefore permits reconstructed SKU rows inside contexts that were already
exact in the observed-sales profile, but does not let reconstruction alone
switch a whole context from fallback to exact.

## Holdout result

Guarded demand-adjusted profile versus observed-sales profile:

| Scope | Baseline WAPE | Adjusted WAPE | Absolute delta |
| --- | ---: | ---: | ---: |
| All holdout, SKU-hour | 1.2593 | 1.2566 | -0.0027 |
| Clean bakery-days, SKU-hour | 1.2254 | 1.2240 | -0.0014 |
| Clean SKU-days, SKU-hour | 1.2582 | 1.2543 | -0.0039 |
| All holdout, SKU-day | 0.8279 | 0.8244 | -0.0035 |
| Clean SKU-days, SKU-day | 0.8301 | 0.8259 | -0.0042 |
| New tier-1 member contexts, SKU-day | 1.2377 | 1.1648 | -0.0729 |

On adjusted bakery/SKU pairs, underforecast quantity fell by 305 units but
overforecast quantity rose by 341 units. Their aggregate SKU-day WAPE worsened
slightly by 0.0007, so the method is not uniformly beneficial at pair level.

## Decision

The direction is promising and does not harm normal holdout aggregates, but it
is not ready for production. The useful effect comes primarily from restoring
history for thin SKU-hour cells, not from changing mature profile shares.

The next candidate must preserve the observed-sales routing boundary:

- reconstruction may change profile values;
- reconstruction may add a SKU to an already mature exact context;
- reconstruction must not by itself promote an entire fallback context to
  exact.

Before any deployment proposal, repeat the experiment on multiple rolling
cutoffs and evaluate the reconstructed bakery-day target separately from SKU
allocation.

## Rolling-cutoff validation

The guarded profile was repeated on three 14-day windows with training cutoffs
2026-06-21, 2026-06-28, and 2026-07-05.

- Clean SKU-day WAPE improved in 3/3 windows, by 0.00212 on average.
- Clean SKU-hour WAPE improved in 3/3 windows, by 0.00248 on average.
- Contexts with a newly restored tier-1 SKU improved in 3/3 windows, by
  0.11909 SKU-day WAPE on average.
- Directly adjusted bakery/SKU pairs worsened in 3/3 windows, by 0.00361
  SKU-day WAPE on average. Underforecast fell, but overforecast rose more.

The aggregate benefit is therefore repeatable, but pair-level applicability is
not. The useful signal is restored membership; applying it to every reconstructed
pair is too broad.

## Bakery-day target A/B

The current global bakery-day LightGBM was retrained for the same three cutoffs.
The adjusted variant used reconstructed demand as the training label and
recomputed every target lag and rolling feature. Holdouts remained observed and
the 11 pilot bakeries were evaluated inside the full-network training context.

The adjusted target improved WAPE in only 1/3 windows. More importantly, on
holdout demand-loss days the baseline already exceeded the reconstructed-demand
proxy by 3,300, 968, and 479 units respectively. Raising the target globally
therefore duplicates volume already supplied by the bakery model. This variant
must not be promoted.

Observed-sales WAPE on censored demand-loss days is retained only as a safety
diagnostic; it is not treated as the optimization target. Even against the
reconstructed-demand proxy, however, the adjusted target won only the final
window.

## Share shrinkage

Guarded adjusted shares were blended with baseline shares at 25%, 50%, and 75%.
All weights preserved the 3/3 aggregate clean-SKU win and the restored-membership
benefit, but none improved directly adjusted pairs in any window. Shrinkage only
slightly reduced excess overforecast. The failure is pair selection/direction,
not merely correction magnitude.

The next candidate is a pair-level walk-forward gate: apply reconstructed
history only when earlier, non-overlapping evidence shows that the pair benefits;
new or unsupported pairs remain on the observed-sales profile. This must be
evaluated with context-level renormalization so bakery-hour totals remain
coherent.

## Pair gate result

The pair-level gate was evaluated on three folds. Each eligibility window ended
before its target window began, and hybrid predictions were renormalized back to
the original bakery-hour total.

The gate improved aggregate clean SKU-day WAPE in 3/3 folds, but the selected
pairs repeated their earlier improvement only 40-46% of the time. Stricter gain
thresholds retained very few pairs and did not restore temporal stability. The
aggregate gain came mainly from renormalization spillover onto neighbouring
SKUs, not from persistent bakery/SKU behaviour. A static pair gate is therefore
rejected.

## Membership-only promotion

Detailed decomposition showed that restored tier-1 membership has a different
effect from mature share correction:

- the promoted SKU itself was overforecast when given its full reconstructed
  share;
- its presence corrected the denominator and removed a substantially larger
  overforecast from the other SKUs in that bakery/dow/hour context;
- contexts whose membership did not change can remain exactly on baseline.

The final experiment keeps all mature exact shares and the entire fallback on
the observed-sales profile. A reconstructed row that crosses `n_days >= 8` is
promoted with a controlled seed share, followed by normal tier-1
renormalization.

A 5% seed is the conservative rolling winner:

- clean SKU-day WAPE improved in 3/3 folds, mean delta -0.000307;
- adjusted-pair clean SKU-day WAPE improved in 3/3 folds, mean delta -0.000128;
- new-membership contexts improved in 3/3 folds, mean delta -0.031466;
- all-holdout SKU-day WAPE improved in 3/3 folds, mean delta -0.000255;
- bakery-hour totals remain unchanged.

Larger seeds provide more aggregate gain but reintroduce pair-level
overforecast. The 5% candidate is still offline-only and requires prospective
shadow validation before any production proposal.

## Local shadow registration

The 5% candidate is now registered in the local stockout-direction shadow
manifest as `demand_adjusted_membership_seed_0.05`. It is deliberately listed
as a candidate, not an enabled component.

Historical rolling evidence ends on 2026-07-19 and therefore does not count as
prospective evidence. Candidate observations are stored separately under
`reports/stockout_direction_shadow/membership_seed_history/` only when
`evaluated_through` advances beyond 2026-07-19. Repeated runs for the same
evaluation date are idempotent and cannot inflate the 21-day promotion gate.

The shadow runner supports `--skip-refresh` to rebuild the manifest from
existing local artifacts without rerunning ClickHouse analyses or overwriting
their reports. The first registration run correctly reports zero prospective
membership-seed days. Production remains unchanged.

## Artifacts

- Reconstruction: `scripts/build_demand_adjusted_stockout_history.py`
- Profile A/B: `scripts/experiment_demand_adjusted_profiles.py`
- Tests: `tests/test_build_demand_adjusted_stockout_history.py` and
  `tests/test_experiment_demand_adjusted_profiles.py`
- Local detailed reports:
  `reports/demand_adjusted_profile_experiment_all_non_allocation/`
- Rolling summary: `reports/demand_adjusted_profile_rolling/aggregate/`
- Bakery-target A/B: `reports/demand_adjusted_bakery_target_experiment/`
- Shrinkage A/B: `reports/demand_adjusted_profile_shrinkage/`
- Pair gate: `reports/demand_adjusted_pair_gate/evaluation/`
- Membership seed: `reports/demand_adjusted_membership_seed/`
