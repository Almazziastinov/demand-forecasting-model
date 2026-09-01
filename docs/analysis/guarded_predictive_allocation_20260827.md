# Guarded predictive allocation (2026-08-27)

Research-only candidate on 282,842 controlled SKU-days / 3,554 bakery-days.

## Candidate mechanics

1. Exact-zero predictive rows receive their causal-trend daily share.
2. Other SKU shares are proportionally reduced so every bakery-day Predictive
   total remains unchanged (maximum numerical delta 9.1e-13).
3. The existing causal same-weekday floor is applied to the filled allocation.
4. Total floor increment is capped at +25% of P50 per bakery-day.
5. Any SKU above 20% of bakery-day volume is water-filled into other SKUs while
   preserving the guarded bakery-day total exactly.

No production state was changed.

## Shape and accuracy

The fill removes all 1,323 exact zeros with positive reconstructed demand,
versus 1,318 remaining under the original floor. It assigns 490 P50 units
against 4,688 reconstructed-demand units on those rows; subset WAPE improves
from 93.28% current to 89.82%. Thus numeric coverage is fixed, but most of the
thin-SKU underforecast remains.

P50 WAPE changes only 44.811% -> 44.835%. The final guarded floor has WAPE
45.210%, RMSE 11.784, R2 0.7388, and bias -0.254%, versus original floor
45.290% / 11.801 / 0.7380 / +0.063%. Exact-zero positive-demand rows fall
1,318 -> 0. Concentration is essentially unchanged because the original floor
already has only seven >=20% bakery-days. SKU 1071 remains top on 2,837 days
and bakery 29 / 2026-08-23 remains 354.5 units / 18.66%, versus reconstructed
demand 161 / 8.25%.

The volume guard changes the unbounded floor uplift distribution from p90
26.38%, p99 37.51%, max 57.88% to an exact maximum 25%.

## Kazan FIFO economics

| Variant | Served | Lost | Strategy expiry | Terminal carry | Gross profit | Delta vs actual |
|---|---:|---:|---:|---:|---:|---:|
| Original floor | 1,878,536 | 516,532 | 50,738 | 90,466 | 111.641m | +9.454m |
| Filled unrestricted floor | 1,878,550 | 516,517 | 51,133 | 90,494 | 111.612m | +9.424m |
| Filled + volume guard | 1,875,519 | 519,549 | 50,241 | 89,309 | 111.545m | +9.357m |
| Final guarded candidate | 1,875,638 | 519,429 | 50,093 | 89,365 | 111.552m | +9.365m |

Coverage filling costs about 29k gross profit versus original floor. The 25%
volume guard reduces profit by a further approximately 67k; the 20% share cap
recovers about 7.5k. These differences are small relative to total profit, but
the candidate does not resolve 1071 dominance or the magnitude of thin-SKU
underforecast. It should not replace the original candidate yet.

Outputs: `reports/guarded_predictive_allocation_20260827/`.
