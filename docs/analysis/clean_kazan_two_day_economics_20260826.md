# Clean Kazan two-day economics (2026-08-26)

Research-only correction of the two-day FIFO economic backtest.

## Scope and mechanics

- 114 Kazan bakeries identified from the local bakery/city report.
- 20 forecast dates split into four uninterrupted blocks: 2026-07-22..08-02,
  08-11..08-13, 08-17..08-18, and 08-21..08-23.
- 324 products matched to the Kazan price/cost workbook.
- Forecast variants are target available stock, not fresh production.
- Candidate production is `max(forecast + sent - carry - received, 0)`.
- Yesterday's stock sells first at a 30% discount; only unsold yesterday stock
  expires. Fresh remainder becomes tomorrow's carry.
- Opening factual stock is tagged separately from strategy-created carry.
- Stock remaining at each block end is reported as terminal carry, not expiry.
- Price and cost are SKU-specific; category is used only for aggregation.

## Aggregate result

| Variant | Production | Served | Lost | Initial-stock expiry | Strategy expiry | Terminal carry | Gross profit | Delta vs actual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Actual state | 1,719,575 | 1,654,881 | 740,187 | 2,812 | 33,168 | 63,556 | 102.187m | - |
| Current | 1,617,159 | 1,562,885 | 832,182 | 2,812 | 24,892 | 60,187 | 93.632m | -8.555m |
| P50+Predictive | 1,767,930 | 1,704,960 | 690,107 | 2,812 | 29,111 | 64,668 | 101.601m | -0.586m |
| P50+Predictive+floor | 1,988,933 | 1,878,536 | 516,532 | 2,812 | 50,738 | 90,466 | 111.641m | +9.454m |

Floor beats actual gross profit in every block by approximately 5.904m,
1.494m, 0.811m, and 1.245m. P50 wins only in the longest block and loses in
the other three. Terminal carry remains an unresolved asset/liability and its
production cost is already charged while no post-block revenue is credited,
so reported candidate profit is conservative but not a full accounting result.

## Heterogeneity

Floor is positive in the largest baked categories but remains negative for
custom products (-0.641m) and cold drinks (-0.205m). At SKU level it is highly
heterogeneous: Kystyby P (+0.645m), Chicken Triangle (+0.634m), and Sausage
Pizza (+0.508m) lead gains, while Closed Pizza (-0.439m), Gubadia order
(-0.207m), and Potato/onion fried pie (-0.202m) lead losses. This supports a
causally trained SKU/category economic gate rather than universal floor.

Outputs: `reports/clean_kazan_two_day_economics_20260826/`.
Production state was not changed.
