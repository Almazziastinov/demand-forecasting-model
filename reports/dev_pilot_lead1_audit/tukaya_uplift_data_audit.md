# Габдуллы Тукая 62А: audit причины нулевого bakery uplift

Дата аудита: 2026-06-16

## Вывод

`Габдуллы Тукая 62А` не получает нормативный bakery-level uplift из текущего production механизма.
Проблема не в LightGBM forecast как таковом: модель обучается на bakery target, где для этой пекарни uplift почти равен нулю.

## Факты

Активный dev lead-1 run:

- run_id: `dev_lead1_history_20260601_20260614_hour_guard_v4`
- период: `2026-06-01..2026-06-14`
- bakery forecast total: `10976.97`
- actual total: `10985.22`
- bias: `-0.08%`

`forecast_final` выше `forecast_base` только на `+0.39%`.

## Training target / uplift source

В `data/processed/bakery_daily_sales_uplifted.csv` для `bakery_id=222`:

- весь период: mean uplift rate `0.06%`, median `0%`, max `6.8%`
- последние 60 дней перед июнем: total uplift rate `0.0012%`
- май 2026: uplift rate `0%`

Сравнение пилотных пекарен за последние 61 день:

| bakery_id | bakery | total uplift rate |
|---:|---|---:|
| 20 | Мира 45 Дербышки Казань | 27.64% |
| 21 | Парковая 7 Казань | 26.88% |
| 22 | Сибирский Тракт 25 Казань | 0.13% |
| 28 | Гудованцева 27 Казань | 27.29% |
| 80 | Калинина 63 Казань | 30.53% |
| 89 | Парина 6 Казань | 27.12% |
| 107 | Четаева 46А Казань | 28.98% |
| 221 | Салиха Батыева 15 Казань | 29.14% |
| 222 | Габдуллы Тукая 62А Казань | 0.001% |
| 257 | Ярмарочная 12 Чебоксары | 25.62% |

## Механическая причина

Текущий bakery uplift строится из `sku_hour_share_profile_daily_smoothed.csv` / `sku_hour_uplift_multiplier_embedded`:

```text
bakery_hour_sales_uplifted = bakery_hour_sales * max(raw_share_sum / norm_share_sum, 1)
```

Для Тукая uplift flags на отдельных SKU есть, включая ходовки, но сумма adjusted SKU shares внутри часа не становится выше normalized суммы.
После `clip(lower=1)` bakery multiplier становится `1.0`, поэтому bakery total не растет.

Иными словами, текущий uplift умеет поправлять SKU-mix, но не видит bakery-level потерянный спрос, если проблема выглядит как абсолютный недовыпуск ходовок, а не как превышение суммы SKU share.

## Root cause: chunk-local normalization artifact

Более точная проверка показала, что для Тукая проблема не в отсутствии SKU uplift flags.
`sku_share_in_hour_adj` по Тукая в среднем повышается примерно так же, как у пекарен, где uplift работает.

Отличие в колонке `sku_share_in_hour_adj_norm`:

| bakery_id | bakery | avg adj sum | avg norm sum | avg ratio |
|---:|---|---:|---:|---:|
| 20 | Мира 45 | 1.228 | 1.000 | 1.228 |
| 28 | Гудованцева 27 | 1.253 | 1.000 | 1.253 |
| 107 | Четаева 46А | 1.268 | 1.000 | 1.268 |
| 222 | Габдуллы Тукая 62А | 1.272 | 1.993 | 0.640 |
| 22 | Сибирский Тракт 25 | 1.284 | 1.981 | 0.652 |

У нормальных пекарен `adj_norm` суммируется к `1.0` на каждый `date x bakery x hour`.
У Тукая и Сибирского Тракта она почти всегда суммируется к `2.0`.

Причина в `src/experiments_v2/smooth_sku_hour_share_profile.py`: `smooth_applied_chunk()` нормализует `sku_share_in_hour_adj_norm` внутри pandas chunk:

```python
group_cols = ["date", "bakery_id", "hour"]
denom = work.groupby(group_cols)["sku_share_in_hour_adj"].transform("sum")
work["sku_share_in_hour_adj_norm"] = work["sku_share_in_hour_adj"] / denom
```

Но входной applied-файл не гарантирует, что все строки одного `date x bakery x hour` попадут в один chunk.
Если группа разрезана границей chunk-а, каждая часть отдельно нормализуется к `1.0`.
В итоговом CSV такая группа имеет `norm_sum ~= 2.0`.

Затем `build_uplift_multiplier_frame()` считает:

```text
sku_uplift_multiplier = raw_share_sum / norm_share_sum
```

Для Тукая получается примерно:

```text
1.27 / 1.99 = 0.64 -> clip(lower=1) -> 1.0
```

Поэтому bakery uplift пропадает.
Для Гудованцева/Четаева:

```text
1.25 / 1.00 = 1.25
```

Поэтому uplift работает.

## Дополнительный сигнал

`src/experiments_v2/76_normative_v1_v2/normative_daily_candidates.csv` для Тукая на свежем срезе показывает bakery normative примерно `+2.8%` к observed, но этот сигнал не участвует в текущем production bakery target.

По отдельным SKU нормативный слой видит часть runner gaps:

- `Кыстыбый П`: normative/fact `1.22`
- `Жар пицца с курицей`: `1.13`
- `Пицца с колбасой`: `1.12`
- `Сосиска под шубой`: `1.15`
- `Беккен капуста`: `1.12`
- `Сэндвич курица`: `1.14`

При этом `Треугольник курица безд` в текущем normative candidate дает `0.97` к факту, поэтому один только старый normative_v1/v2 тоже не полностью закрывает бизнес-наблюдение по ходовкам.

## Что чинить

Первый фикс должен быть техническим: пересобрать `sku_hour_share_profile_daily_smoothed.csv` так, чтобы нормализация `date x bakery x hour` не зависела от chunk boundaries.

Варианты:

1. Перед smoothing читать/писать данные отсортированными по `date, bakery_id, hour`, а chunk reader должен переносить незавершенную группу в следующий chunk.
2. Делать normalization вторым проходом: сначала записать `sku_share_in_hour_adj`, затем посчитать denominators по `date x bakery x hour`, затем нормализовать join-ом.
3. Для ClickHouse path считать `adj_norm` и uplift multipliers прямо SQL-агрегацией, без pandas chunk-local normalization.

После этого надо пересобрать uplift multipliers и bakery daily uplifted target.
Ожидаемо Тукая и Сибирский Тракт должны получить uplift ближе к остальным пекарням, потому что `adj_sum` у них уже высокий.

Дополнительно, если после технического фикса uplift останется недостаточным, нужен отдельный bakery-level normative guard:

1. Посчитать runner demand floor по последним 30-60 дням на SKU-day уровне.
2. Суммировать положительные runner gaps в bakery-day uplift floor.
3. Ограничить floor мягким cap, например `+10..20%` для первого dev варианта.
4. После подъема bakery total перераспределить массу runner guard-ом внутри SKU.

Этот guard надо тестировать сначала только на проблемных пекарнях: `222`, `22`, возможно `21`.
