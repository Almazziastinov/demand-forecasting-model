# Rolling comparison with actual operational state (2026-08-26)

Production releases, transfers, sales and previous-day calculated closing
stock were joined to the same 20 rolling forecast dates. The common
inventory-controlled scope contains 282,842 SKU-days, 188 bakeries and 528
products.

Actual available-to-sell is defined as production + positive previous-day
calculated closing + received - sent. Demand is observed sales plus the
rolling calibrated post-last-sale loss.

| Variant | Volume | Surplus | Underbake | Imbalance |
|---|---:|---:|---:|---:|
| Actual state | 2,886,153 | 466,915 | 1,155,055 | 1,621,970 |
| Current | 2,735,395 | 454,798 | 1,293,696 | 1,748,494 |
| Predictive, same volume | 2,741,692 | 394,322 | 1,226,923 | 1,621,245 |
| P50 + Predictive | 3,043,379 | 535,389 | 1,066,303 | **1,601,692** |
| P50 + Predictive + simple floor | 3,576,546 | 810,525 | **808,272** | 1,618,798 |

P50 + Predictive beats actual-state underbake on all four folds and reduces
aggregate underbake by 88,752, while adding 68,475 surplus. It also has the
lowest aggregate equal-cost imbalance, although it loses to actual imbalance
on the last two folds. The simple floor reduces underbake by 346,783 versus
actual, but adds 343,611 surplus; it beats actual imbalance on two folds and
loses on two.

Actual-state underbake contains a 118,223-unit availability reconciliation
gap across 40,712 rows where registered sales exceed computed availability.
The calibrated lost-demand portion is 1,036,832 units. The gap is a data-
quality/opening-stock limitation, not confirmed operational underbake, and
must remain explicit in decision tables.

Therefore P50 + Predictive is the strongest conservative candidate. The
simple floor is the underbake-first candidate, but it approximately exchanges
one additional surplus unit for one unit of underbake reduction relative to
actual state.

Artifacts:

- `scripts/evaluate_rolling_with_actual_state.py`
- `reports/rolling_actual_state_comparison_20260826/`

Production was unchanged.
