# Full SKU allocation screen — 2026-07-20

## Scope

The screen covers every SKU available in the accepted pilot stockout dataset,
not only the nine previously labelled as regime-shift candidates. The period is
2026-06-01 through 2026-07-19, across 11 pilot bakeries: 7,715 classified
SKU-days and 53 SKUs.

Pair-level conclusions use confirmed non-stockout days only. A bakery/SKU pair
is eligible after at least five such days. Allocation is evaluated after
replacing the forecast bakery total with the actual bakery total, so the metric
isolates the SKU share from bakery-volume error.

## Result

- 448 bakery/SKU pairs have enough clean evidence.
- 32 pairs across 17 SKUs have allocation bias below -10%.
- 17 pairs are `missing_allocation`: observed sales exist but allocated share
  is zero. Their confirmed deficit is 1,064 units.
- 15 pairs are `persistent_local_underallocation`: allocation is non-zero but
  too low. Their confirmed deficit is 337.1 units.
- Three additional SKUs show only a stockout-regime gap at the aggregate SKU
  level: `Сочень`, `Губадия мини`, and `Хот-дог Датский куриный`.

The material non-zero local cases are led by:

| SKU | Problem bakeries | Confirmed pair deficit |
| --- | ---: | ---: |
| Клубника и банан НОВЫЙ | 6 | 128.8 |
| Маковка | 2 | 106.9 |
| Киш курица | 1 | 39.3 |
| Пирог с Манго | 3 | 34.6 |
| Губадия | 1 | 15.8 |

`Капуста и курица` has missing allocation in eight bakeries and therefore
looks systemic rather than like an isolated bakery/SKU anomaly.

## Post-fix check

Missing allocation did not disappear after the 2026-07-15 assortment/cap
ordering fix. On 2026-07-15 through 2026-07-19, confirmed non-stockout rows
with positive sales and zero allocation include:

- bakery 257: nine SKU-days, 139 units sold;
- `ЖарПицца Пикантная` at bakery 257: three days, 78 units;
- `Капуста и курица`: four SKU-days across bakeries 16, 21, 22, and 107,
  16 units in total.

The post-fix window is only five calendar days, so it is sufficient to prove
that zero-allocation coverage gaps remain, but not to estimate their stable
rate.

## New-product eligibility check

Zero allocation is not automatically a defect: the production logic is
allowed to exclude a newly introduced SKU until it has enough history to enter
the assortment/profile allocation. The 126 positive-sales SKU-days behind the
1,064-unit `missing_allocation` total were therefore checked against sales
history available strictly before each forecast date.

None of these rows is a cold-start case:

- every row had more than 21 prior selling days;
- the minimum prior-history count for an affected bakery/SKU pair was 26 days;
- the largest cases had 30 selling days within the preceding 30 calendar days;
- `ЖарПицца Пикантная` at bakery 257 had 34 prior selling days at its first
  zero-allocation event and up to 76 by the last one;
- `Капуста и курица` had at least 28 prior selling days in every affected
  bakery and usually 30 selling days in the preceding 30-day window.

The profile consumer requires at least three observations for its fallback
hour profile and eight same-weekday observations for the exact tier. These
cases have ample product-level history; the next trace must establish whether
that history is absent from the profile keys, removed by assortment/product-id
mapping, or lost while the forecast grid is constructed.

## Interpretation and next actions

The candidates require separate mechanisms:

1. `missing_allocation` must be traced through assortment membership, profile
   coverage, product-id mapping, and forecast-grid construction. The observed
   cases have already passed the new-product history test. An uplift cannot
   repair a missing row or a zero base share.
2. `persistent_local_underallocation` should be corrected at bakery/SKU level,
   with shrinkage and minimum evidence rather than a global SKU multiplier.
3. `stockout_regime_shift` should be tested with leakage-free prior-stockout
   signals and a capped additive uplift; it must not redistribute volume away
   from other SKUs.
4. The top-selling SKUs remain a control group for ensuring that targeted
   corrections do not distort the main sales volume.

Generated detail files are under `reports/full_sku_allocation_screen/`.
