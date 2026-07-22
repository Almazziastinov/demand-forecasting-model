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

## Artifacts

- Reconstruction: `scripts/build_demand_adjusted_stockout_history.py`
- Profile A/B: `scripts/experiment_demand_adjusted_profiles.py`
- Tests: `tests/test_build_demand_adjusted_stockout_history.py` and
  `tests/test_experiment_demand_adjusted_profiles.py`
- Local detailed reports:
  `reports/demand_adjusted_profile_experiment_all_non_allocation/`

