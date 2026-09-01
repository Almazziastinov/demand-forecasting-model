# Relaxed stockout detector and demand quantiles (2026-08-26)

The stockout detector was rebuilt as requested:

- available = production + positive prior-day closing stock + received - sent;
- stockout when available is positive but no greater than sales and the last
  sale is before 19:00;
- raw lost demand = same-day average sales rate from 07:00 through the last
  sale multiplied by the remaining time through 23:00;
- the existing conservative cap remains: min(10 units, 50% of daily sales).

Only bakery-product pairs with observed production or movement activity are
eligible. This prevents absent inventory facts for resale-only products from
being interpreted as zero availability.

Across the complete label history, the rule identifies 28,391 SKU-days and
142,059 capped lost units. On the common 2026-07-02..2026-07-19 comparison
scope (18 dates, 11 bakeries), recognized lost demand rises from the previous
2,339 to 30,564 units.

| Model | Forecast | Surplus | Underbake | Imbalance | Bias vs sales | Lost covered |
|---|---:|---:|---:|---:|---:|---:|
| Sales-target LightGBM | 217,294 | 435 | 34,132 | 34,567 | -1.42% | 15.04% |
| Direct demand | 230,338 | 1,746 | 22,400 | 24,146 | +4.50% | 37.72% |
| Selective uplift | 217,294 | 435 | 34,132 | 34,567 | -1.42% | 15.04% |
| P50 | 230,616 | 2,156 | 22,531 | 24,687 | +4.62% | 38.34% |
| P55 | 233,233 | 2,892 | 20,650 | 23,542 | +5.81% | 42.16% |
| P60 | 236,339 | 3,712 | 18,364 | 22,077 | +7.22% | 48.22% |
| P67 | 240,030 | 4,875 | 15,836 | 20,711 | +8.89% | 54.63% |
| P75 | 243,878 | 6,330 | 13,443 | 19,773 | +10.64% | 60.69% |

Strict demand is 250,991 units (220,427 sales plus 30,564 reconstructed lost
demand). Even P75 remains below strict demand in aggregate. Selective uplift
collapses to the sales baseline because the relaxed label is too prevalent
for its rare-event classifier contract.

The result is a diagnostic, not rollout evidence. The detector is much less
conservative and needs pseudo-stockout validation before its absolute lost
volume is accepted. Production was not changed.
