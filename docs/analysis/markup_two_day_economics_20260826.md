# Two-day economics with actual markups and 30% day-two discount (2026-08-26)

The Kazan markup workbook provides current unit price and new unit cost by
product. Exact normalized product-name matching to ClickHouse mapped 328
products in the rolling controlled scope, covering 96.76% of demand. Day-one
sales use full price; carried day-two sales use 70% of current price. Produced
units incur the workbook's new unit cost; no extra disposal fee is added
because the production cost of expired units is already sunk.

| Strategy | Served units | Lost units | Expired units | Revenue | Production cost | Gross profit | GP delta vs actual |
|---|---:|---:|---:|---:|---:|---:|---:|
| Actual-state simulation | 2,406,729 | 1,051,611 | 62,117 | 231,586,700 | 85,034,450 | 146,552,300 | - |
| Current | 2,366,284 | 1,092,056 | 148,575 | 215,889,700 | 87,378,530 | 128,511,200 | -18,041,070 (-12.31%) |
| P50 + Predictive | 2,593,220 | 865,120 | 177,917 | 234,416,400 | 96,777,750 | 137,638,700 | -8,913,574 (-6.08%) |
| P50 + Predictive + floor | 2,871,844 | 586,496 | 318,822 | 263,231,200 | 122,039,400 | 141,191,900 | -5,360,418 (-3.66%) |

The earlier relative-cost sensitivity was optimistic because it ignored SKU
price/cost mix and discounted carried sales. With actual workbook economics,
neither candidate beats actual gross profit. Floor earns 3.553m more gross
profit than P50 and 12.681m more than current, but remains 5.360m below actual.
Its additional 31.645m revenue versus actual costs 37.005m more production.

P50 shifts 754,887 units to discounted day-two sales versus 385,820 actual;
floor shifts 1,086,473. This discount materially erodes the value of higher
service. Full automation therefore cannot optimize only demand coverage or
underbake. The objective must use product-level price, cost, carry age and
expected discounted sell-through, with a profit guardrail that can reject
unprofitable uplift.

Limitations: 3.24% of demand lacks a valid price/cost mapping; the workbook is
Kazan-specific while the rolling scope contains the network; transfers and
opening-stock reconciliation remain approximate; capacity, labor and batch
costs are excluded. The result is decision-useful directionally but is not yet
an accounting-grade network P&L.

Artifacts:

- `scripts/build_markup_price_mapping.py`
- `scripts/evaluate_markup_two_day_economics.py`
- `reports/markup_price_mapping_20260826/`
- `reports/markup_two_day_economics_20260826/`

Production was unchanged.
# Correction: forecast is target stock, not production (2026-08-26)

The first version overstated candidate production and expiry because it treated
the forecast quantity as fresh production every day. The corrected FIFO
simulation treats it as the target available stock and computes candidate
production as `max(forecast + sent - yesterday_carry - received, 0)`.
Yesterday's units are sold first at a 30% discount; only yesterday's unsold
units expire. Fresh unsold units carry into the following consecutive day.

On the 328 mapped products (96.76% of demand), corrected gross profit is
146.552m for actual state, 133.611m for current, 145.213m for P50+Predictive,
and 159.542m for P50+Predictive+simple floor. Thus P50 is 1.340m below actual,
while floor is 12.990m above actual in this simulation. Corrected expiry is
62,117 / 47,369 / 50,580 / 81,708 units respectively. The floor result is not
yet an automation recommendation: category/SKU decomposition shows harmful
positions hidden by the aggregate gain, and the Kazan workbook is still being
applied to a wider network scope.

Category and SKU outputs are in `by_category.csv` and `by_product.csv`.
Prices and costs remain SKU-specific; category is an aggregation dimension,
not a substituted average margin.
