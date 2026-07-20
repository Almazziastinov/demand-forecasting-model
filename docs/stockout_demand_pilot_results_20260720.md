# Pilot stockout-demand experiment results

Date: 2026-07-20

## Scope and safety

- Offline research only; production services, ClickHouse tables, forecast runs,
  profiles, and timers were not changed.
- Source period: 2026-04-30 through 2026-07-19 (81 days).
- Pilot bakeries: `16, 20, 21, 22, 28, 80, 89, 107, 221, 222, 257`.
- Daily source: 42,695 rows from `Svezhar.mart_zero_sales_60d`.
- Hourly source: 807,263 check lines from `Svezhar.stg_check_lines`.

## Method corrections before the run

The pilot pipeline was corrected before evaluation:

1. `bakery_hour_sales` now includes every product sold by the bakery. The SKU
   frame remains limited to bakeable categories, but the bakery-share
   denominator is no longer normalized inside that subset.
2. Profile reconstruction is performed separately inside each training window;
   no holdout observations are used to build its reference or normal-day
   benchmark.
3. The pseudo-stockout low-volume guardrail uses normal daily volume from the
   training period instead of the hidden true total of the holdout day.
4. A case is eligible for reconstruction only when at least three normal days
   exist for the same bakery, SKU, and weekday.

## Input and label quality

- Inventory balance consistent within one unit: 42,695 / 42,695 (100%).
- Hourly and daily sales agree within one unit: 34,235 / 42,695 (80.19%).
- Inventory stockouts: 27,246.
- Reliable inventory stockouts after hourly/daily agreement: 23,248.
- Strong temporal stockouts with at least three normal reference days: 2,015.

The high-confidence reconstruction adjusted 7,424 hours across 51 products and
all 11 pilot bakeries. It added 7,387.67 demand units. The previous permissive
`normal_days > 0` variant adjusted 5,156 cases and added 20,318.56 units, but
3,141 of those cases had fewer than three reference days and the largest case
used only one reference day. That variant is rejected.

## Pseudo-stockout backtest

The test hides the final two, three, or four selling hours on known
non-stockout days. With the three-day reference requirement:

- recovery ranges from 68.96% to 95.44% for the capped policy;
- bias remains negative in every tested segment, from -4.56% to -31.04%;
- WAPE ranges from 47.03% to 66.98%;
- results are conservative but remain noisy at individual SKU-day level.

The strongest segment is higher-volume products with a two-hour hidden tail.
The weakest segment is low-volume products with a four-hour hidden tail.

## Profile comparison

Holdout: 2026-07-06 through 2026-07-19, restricted to balance-consistent,
hourly/daily-consistent, non-stockout rows.

| History | Adjusted train hours | Imputed train units | Baseline weighted share MAE | Demand weighted share MAE |
| ---: | ---: | ---: | ---: | ---: |
| 28 days | 86 | 73.46 | 0.06894697 | 0.06897067 |
| 42 days | 860 | 875.23 | 0.06893876 | 0.06894776 |
| 56 days | 2,685 | 2,774.64 | 0.06888782 | 0.06889732 |

Demand-adjusted profiles are slightly worse in all three normal-day holdout
comparisons. The difference is small, but there is no evidence that replacing
the current profile improves ordinary realized-sales allocation.

## Decision

Do not deploy this demand-adjusted profile and do not replace the active
`stockout_20260716` production correction from this experiment.

The reconstruction remains useful as an offline estimate of censored tail
demand. The next useful experiment is a segmented forecast backtest that keeps
the correction limited to high-confidence cases and reports results by bakery,
SKU volume band, and hidden-tail length. A production change requires a clear
improvement over both the sales-profile baseline and the currently deployed
stockout correction.
