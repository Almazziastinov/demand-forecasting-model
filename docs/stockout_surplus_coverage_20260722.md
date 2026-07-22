# Stockout deficit versus closing surplus

Date: 2026-07-22
Status: offline research only; production unchanged

## Question

Can closing surplus on other SKUs explain reconstructed demand lost to clear
stockouts? If it can, the bakery produced enough total units and the main fault
is SKU allocation. If it cannot, the uncovered part requires a bakery-volume
uplift. Partial coverage is a mixed case.

For each bakery-day:

- `D` is reconstructed stockout demand;
- `E` is usable closing surplus on non-stockout SKUs;
- allocation component is `min(D, E)`;
- bakery-volume gap is `max(D - E, 0)`;
- residual excess is `max(E - D, 0)`.

The baseline definition of `E` is deliberately conservative. It uses only
balance-consistent rows whose hourly and daily sales agree, excludes the
stockout recipient SKU, excludes two-day products, and leaves one unit of
closing reserve per donor SKU. Product metadata was read from ClickHouse; no
database writes were performed.

## Result

The sample contains 1,296 clear stockout SKU-days aggregated into 461 positive
deficit bakery-days from 2026-06-01 through 2026-07-19. Reconstructed deficit is
8,305.8 units.

Under the baseline one-unit reserve:

- 5,754.9 units, or 69.3%, can be paired with usable surplus elsewhere in the
  same bakery-day;
- 2,550.8 units, or 30.7%, remain a bakery-volume gap;
- 281 days have surplus above 110% of deficit;
- 26 days are approximately balanced at 90-110%;
- 133 days are mixed at 10-90%;
- 21 days have at most 10% coverage and support a volume-shortage explanation.

A temporal sensitivity requires donor products to have a last sale hour no
earlier than the latest recipient stockout hour. It still explains 5,274.4
units, or 63.5%. Its day split is 242 allocation-plus-excess, 39 balanced, 142
mixed, and 38 volume-shortage days.

Reserve sensitivity is material but does not reverse the conclusion:

| Reserve per donor SKU | Allocation component | Bakery-volume gap |
| ---: | ---: | ---: |
| 0 units | 82.0% | 18.0% |
| 1 unit | 69.3% | 30.7% |
| 2 units | 57.2% | 42.8% |

Closing surplus is not unique to stockout days. In the same date window, its
median is 24 units on the 462 reconstructed-stockout bakery-days and 21 units
on 77 non-stockout bakery-days; means are 29.0 and 26.7. Therefore surplus is a
useful capacity/allocation constraint, but not proof that a particular donor
SKU directly displaced a particular recipient SKU.

The largest baseline donor pools are concentrated in recurring products,
including chicken and beef triangles, sausage pastry, cabbage bekken, and
chicken pizza. This makes the phenomenon systematic enough to model rather than
treating it as isolated outliers.

## Decision

Use this balance decomposition as a bakery-day regime label:

1. For covered demand, train or evaluate a dynamic SKU-share allocator while
   keeping the available bakery total fixed.
2. For the uncovered `D - E` part, correct historical demand before profiles
   and bakery-volume targets are built.
3. For mixed days, apply both actions in those measured proportions.
4. Track `E - D` separately as overproduction; do not force it into the
   stockout reconstruction.

The label is suitable for offline training/evaluation now. It is not yet a
production rule because end-of-day stock does not identify exact causal
SKU-to-SKU transfers, and the small normal-day control group shows that some
surplus is structural.

## Artifacts

- Analysis: `scripts/analyze_stockout_surplus_coverage.py`
- All-case reconstruction mode:
  `scripts/build_demand_adjusted_stockout_history.py`
- Tests: `tests/test_stockout_surplus_coverage.py`
- Baseline report: `reports/stockout_surplus_coverage/`
- Reserve sensitivities: `reports/stockout_surplus_coverage_reserve0/` and
  `reports/stockout_surplus_coverage_reserve2/`
