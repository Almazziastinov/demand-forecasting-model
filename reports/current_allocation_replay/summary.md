# Current Allocation Replay — 2026-07-20

## Scope

Diagnostic replay of the active 2026-07-20 allocation shares against the 397
historical confirmed stockout-underforecast cases. Current bakery/SKU/day-of-
week shares were multiplied by the historical bakery forecast total. This
isolates the current allocation rule while preserving the historical top-level
bakery volume.

This is not a leakage-free model backtest: the current profile includes data
through 2026-07-19. It is a mechanism diagnostic that asks whether the current
allocator would still assign less than the observed, stockout-censored sales.

## Aggregate result

| Metric | Old allocation | Current allocation replay |
| --- | ---: | ---: |
| Confirmed shortfall, all 397 cases | 1,509.3 | 1,005.9 |
| Shortfall where bakery volume was sufficient, 310 cases | 1,190.3 | 783.2 |
| Pure-allocation shortfall at actual bakery volume, all cases | 1,510.4 | 1,007.9 |
| Pure-allocation shortfall at actual bakery volume, sufficient-volume cases | 1,281.4 | 874.2 |

- Coverage: 397/397 cases.
- Fully fixed: 90 cases.
- Improved but still short: 192 cases.
- Unchanged: 3 cases.
- Worsened: 112 cases.

The refreshed allocator removes about one third of the confirmed deficit, but
307 cases still remain below observed sales. The near-identical result when
using actual bakery volume confirms that the remaining issue is allocation,
not only bakery-level volume.

## Largest residual concentrations

Residual shortfall by bakery starts with:

| Bakery | Remaining cases | Remaining shortfall |
| ---: | ---: | ---: |
| 221 | 63 | 215.5 |
| 16 | 26 | 116.7 |
| 107 | 37 | 115.8 |
| 22 | 27 | 111.7 |
| 222 | 34 | 110.3 |

Most repeated residual SKU include `Клубника и банан НОВЫЙ`, `Маковка`,
`Губадия`, `Губадия мини`, `Пицца с колбасой`, `Сметанник маковый`,
`Вак-бэлиш`, and `Трехслойник НОВЫЙ`.

## Top-selling SKU check

The five highest-volume SKU across the 11 pilot bakeries in the analysis
window were:

1. `Треугольник курица безд` — 60,053.4 units.
2. `Кыстыбый П` — 43,335.0 units.
3. `Пакет спасибо` — 26,143.4 units.
4. `Треугольник говядина безд` — 24,834.4 units.
5. `Сосиска в тесте` — 20,797.0 units.

The issue is not limited to new or low-volume products. For `Треугольник
курица безд`, replay shortfall remains 114.4 of the original 133.6 units. For
`Кыстыбый П`, it worsens from 40.9 to 51.3 units. `Пакет спасибо` has no
confirmed cases in this set.

## Interpretation

The stale refresh explained a material share of the failures, especially
missing and near-zero allocations, but not the systematic remainder. The next
experiment should estimate a local bakery/SKU calibration factor from recent
non-stockout days, shrink it toward city/SKU and network/SKU priors, and apply
separate eligibility/evidence rules for new products. Bakery total should stay
fixed in this allocation-only experiment.
