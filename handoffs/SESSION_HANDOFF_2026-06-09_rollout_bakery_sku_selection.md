# Session Handoff - 2026-06-09 - Rollout Bakery and SKU Selection

## Scope

We started selecting Kazan bakeries for the first rollout of the forecasting /
baking-plan system.

The user already had a manually inspected seed list:

```text
Баки Урманче 6 Казань
Татарстан 16 Казань
Баумана 29/11 Казань
Зорге 101 Казань
Гудованцева 27 Казань
```

Important framing:

- do not present the selection as purely metric-optimal;
- bakery-day metrics are useful as a sanity check, but not sufficient;
- uplift and operational effects can make clean metric interpretation unreliable;
- the real rollout risk is SKU allocation quality, not only total bakery-day
  quantity.

## Data Used

Main bakery-day source:

```text
reports/bakery_day_model_bias_by_bakery.csv
```

Main SKU holdout source:

```text
reports/prod_holdout_sku_backtest/by_bakery_sku.csv
```

SKU variant source for recent/blend correction:

```text
reports/prod_holdout_sku_backtest_variants/blend_recent_50_by_bakery_sku.csv
reports/prod_holdout_sku_backtest_variants/core_recent_70_by_bakery_sku.csv
reports/prod_holdout_sku_backtest_variants/dead_0d_by_bakery_sku.csv
reports/prod_holdout_sku_backtest_variants/active_3d_by_bakery_sku.csv
```

The bakery-day report reads correctly with pandas as UTF-8. Some terminal `rg`
output looked like mojibake, but pandas output was correct Russian.

## Seed Bakery-Day Metrics

The original user-selected points:

```text
id   bakery                         actual/day   WMAPE   bias
142  Баки Урманче 6 Казань              1787      8.1%  -6.0%
102  Татарстан 16 Казань                1714     11.3%  -2.7%
30   Баумана 29/11 Казань               2057     12.9%  +1.5%
23   Зорге 101 Казань                   1677      8.0%  +1.2%
28   Гудованцева 27 Казань              1598      7.0%  +0.4%
```

Initial similar Kazan points by bakery-day scale and sanity metrics:

```text
id   bakery                         actual/day   WMAPE   bias
105  Фучика 96 Казань                   1718      8.5%  -0.3%
29   Айдарова 8А корп 1 Казань          1759      8.8%  -0.6%
89   Парина 6 Казань                    1910      7.8%  -0.7%
16   Кулагина 4 Казань                  2077      8.6%  -2.0%
79   Ильича 19/43 Казань                2199      7.7%  -0.5%
107  Четаева 46А Казань                 1673      9.8%  +1.6%
14   Ямашева 19А Казань                 1631      9.7%  +0.4%
62   Шамиля Усманова 16А Казань         1569     10.4%  -0.6%
25   Фучика 30 Казань                   1473      9.0%  +1.3%
21   Парковая 7 Казань                  1461      8.8%  -1.4%
60   Мусина 68 Казань                   1405      7.7%  +2.3%
5    Губкина 17 Казань                  1444      9.2%  -0.4%
```

## Important User Correction

The user explicitly rejected these points for rollout:

```text
Баумана 29/11 Казань
Фучика 96 Казань
Мусина 68 Казань
```

Reason:

```text
The SKU share distribution is fundamentally incorrect there.
```

This should override bakery-day metrics. These stores can be useful as hard
diagnostic/control examples, but not as safe first rollout candidates.

## SKU-Level Finding

Bakery-day total quality does not imply SKU allocation quality.

Baseline SKU WMAPE on candidate stores was high:

```text
roughly 62% to 90% on SKU-level baseline
```

Recent/blend correction helps significantly:

```text
best SKU variants roughly 38% to 46% WMAPE on the cleaner candidate stores
```

But this is still not accurate enough for fully automatic production without
human review. The first rollout should be positioned as decision support.

## Candidate Stores After Exclusions

Recommended cleaner rollout shortlist:

```text
16   Кулагина 4 Казань
28   Гудованцева 27 Казань
14   Ямашева 19А Казань
89   Парина 6 Казань
25   Фучика 30 Казань
23   Зорге 101 Казань
62   Шамиля Усманова 16А Казань
29   Айдарова 8А корп 1 Казань
```

Use with more caution / manual review:

```text
142  Баки Урманче 6 Казань
102  Татарстан 16 Казань
107  Четаева 46А Казань
5    Губкина 17 Казань
21   Парковая 7 Казань
```

Excluded from rollout for now:

```text
30   Баумана 29/11 Казань
105  Фучика 96 Казань
60   Мусина 68 Казань
```

## Runner SKU Layer

The user emphasized that frequently sold / high-volume SKU need separate
review, because the business impact is concentrated there.

We used a practical runner definition:

```text
fact_qty >= 500 over holdout
recent_days_sold >= 20
exclude service / non-production categories such as Прочие товары
```

On the cleaner store set, the important runner SKU behaved as follows under
`blend_recent_50`:

```text
product                           fact_qty   forecast_qty   bias    WMAPE
Треугольник курица безд             67964        63334       -7%     16%
Кыстыбый П                          37571        36848       -2%     18%
Треугольник говядина безд           28714        29193       +2%     19%
Беккен капуста                      21798        23321       +7%     22%
Сосиска в тесте                     21343        22104       +4%     20%
Сосиска под шубой                   19751        20277       +3%     20%
Элеш с курицей                      19247        18461       -4%     20%
Пицца с колбасой                    16970        18396       +8%     20%
Вак-бэлиш                           13768        14918       +8%     24%
Жар пицца с курицей                 13071        15327      +17%     27%
Треугольник острый                  12596        14274      +13%     27%
Сочень                              11113        12949      +17%     37%
Сметанник                            8786        12046      +37%     47%
```

Interpretation:

- core runners such as chicken triangle, beef triangle, kystybyi, sausage bake,
  sausage under coat, and elesh are relatively usable;
- similar-looking savory / sweet products can still be overallocated;
- runner SKU review must be separated from long-tail SKU review.

## Eclair Problem

User noted, and data confirms, that eclairs are overforecast almost everywhere.

Across the cleaner candidate stores under `blend_recent_50`:

```text
all eclair-like SKU:
fact_qty      5313
forecast_qty 16972
bias_qty     +11659
```

Examples for `Эклер классический`:

```text
Кулагина 4 Казань              171 fact -> 1021 forecast
Зорге 101 Казань               175 fact ->  928 forecast
Парина 6 Казань                205 fact ->  952 forecast
Баки Урманче 6 Казань          198 fact ->  943 forecast
Фучика 30 Казань                70 fact ->  763 forecast
Шамиля Усманова 16А Казань      89 fact ->  768 forecast
Татарстан 16 Казань            136 fact ->  785 forecast
```

This is not a single-store issue. It needs a SKU-specific rule before rollout:

```text
Cap or manually review all Эклер* SKU.
```

## Practical Rollout Rule

Do not evaluate candidate stores only by total bakery-day metrics.

Use two layers:

1. total bakery-day sanity;
2. top runner SKU sanity.

The first rollout should include a manual review checklist:

```text
- exclude stores with broken SKU share distribution;
- check top 20 runner SKU by actual quantity;
- cap/review all Эклер* SKU;
- ignore or de-emphasize service SKU such as package products;
- separately inspect forecast-only SKU with recent_days_sold = 0;
- treat first wave as decision support, not automatic production control.
```

## Open Next Step

If continued, the next useful artifact would be a small report or notebook that
outputs for a candidate bakery list:

```text
- bakery-day total metrics;
- top runner SKU table;
- eclair-specific rows;
- forecast-only SKU rows;
- category-level bias;
- pass / manual-review / exclude recommendation.
```

This would make future rollout selection repeatable instead of relying on
manual spreadsheet inspection.
