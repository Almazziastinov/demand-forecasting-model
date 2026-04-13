# Experiments V2 Registry

Короткий содержательный реестр по `experiments_v2`.

Этот файл опирается не только на наличие файлов в каталогах, но и на зафиксированные выводы из:

- `src/experiments_v2/EXPERIMENTS.md`
- `reports/report.md`

Полный narrative и детали отдельных гипотез остаются в `EXPERIMENTS.md`. Здесь зафиксировано, что считать главным по смыслу.

## 1. Current Baseline Candidates

Это эксперименты и направления, которые выглядят наиболее сильными кандидатами на текущий исследовательский baseline.

| ID / Direction | Status | Почему в фокусе |
|---|---|---|
| `43_quantile` | leading candidate | Лучший MAE среди описанных sales-target экспериментов, плюс интервальные прогнозы P25/P50/P75 |
| `45_log_residual` | strong candidate | Один из лучших результатов при умеренной сложности, практичный компромисс |
| `40_tweedie` | strong candidate | Небольшой, но устойчивый прирост на high-demand проблеме |
| `60_baseline_v3` | active baseline | Опорный baseline для более новых v3/v2-экспериментов |
| `66_cluster_features` | strong candidate | Лучший из текущих demand/v3-производных экспериментов: routed cluster_ts дал MAE 2.8469 |
| `03_demand_target` | strategic candidate | Не лучший по sales-MAE, но критически важен как переход к прогнозу спроса вместо продаж |
| `61_censoring_behavioral` + `62_assortment_availability` + `63_combined_61_62` | strategic candidate | Продолжение ключевой линии про восстановление реального спроса из цензурированных продаж |

## 2. Proven Positive Results

Это уже не просто гипотезы, а результаты, которые в документах выглядят подтвержденными и полезными.

| ID | Result | Summary |
|---|---|---|
| `01_baseline_8m` | proven baseline | Сильная историческая точка отсчета: MAE около 2.29 на 8 месяцах |
| `02_optuna_8m` | marginal positive | Гиперпараметры уже близки к оптимуму, большого hidden win здесь не видно |
| `09_price_features` | clear positive | Самое заметное улучшение среди feature-engineering экспериментов на demand target |
| `23_cluster_models` | conditional positive | Есть прирост, но полезность сегментно-зависима: low/medium demand выигрывает, high-demand может страдать |
| `66_cluster_features` | positive | Кластеризация как фича почти нейтральна, но routed model по `cluster_ts` улучшает exp63 на -0.0071 MAE |
| `40_tweedie` | positive | Улучшает baseline без сильного усложнения |
| `43_quantile` | best positive | Лучший MAE и дополнительная бизнес-ценность через интервалы |
| `45_log_residual` | positive | Рабочий двухэтапный подход, конкурентный по качеству |
| `46_variance_weighted` | marginal positive | Эффект есть, но он небольшой |
| `03_demand_target` | business-positive | Переход к спросу улучшает не только метрики по спросу, но и бизнес-результат в P&L логике |

## 3. Neutral Or Low-Value Results

Эксперименты, которые дали исследовательский вывод, но не выглядят сильным направлением для продолжения в текущем виде.

| ID | Result | Why |
|---|---|---|
| `05_profiles` | neutral | Явные профили почти не добавляют информации сверх категориальных идентификаторов |
| `08_location_features` | neutral | Статические location-фичи почти бесполезны при наличии bakery ID |
| `41_log_target` | low-value | Почти baseline, без убедительного выигрыша |

## 4. Negative Or Rejected

Эксперименты, которые на текущем этапе стоит считать отрицательными или временно закрытыми.

| ID | Status | Reason |
|---|---|---|
| `10_stacking` | rejected for now | Сложность выше пользы, ансамбль не дал выигрыша |
| `42_asymmetric_loss` | failed | Custom objective реализован неудачно, результат сломан |
| `44_mixture_of_experts` | rejected for now | Разделение на expert-модели ухудшило качество |

## 5. Strategic Research Directions

Это главные направления, которые по документам выглядят стратегически важнее локальных улучшений MAE.

| Direction | Priority | Why |
|---|---|---|
| Demand вместо sales | highest | Главная смысловая ось проекта: прогнозировать, сколько нужно произвести, а не сколько успели продать |
| Censored demand recovery | highest | В `report.md` зафиксировано, что большая доля данных цензурирована ранним stockout |
| High-demand behavior | high | Систематический недопрогноз на товарах с высоким спросом явно выделен как центральная проблема |
| Adaptive demand profiles | high | Улучшение механизма восстановления спроса влияет на сам target, а не на одну дополнительную фичу |
| Segment-aware modeling | medium-high | Кластеры уже дали второй подтверждающий сигнал: routing по TS cluster улучшил exp63 |

## 6. Deferred Directions

Направления, которые пока не выглядят первоочередными относительно demand/censoring/high-demand линии.

| Direction | Status | Comment |
|---|---|---|
| Per-bakery models | deferred | Высокий риск переобучения и рост сложности |
| Per-product models | deferred | Слишком мало наблюдений на часть SKU |
| Hierarchical reconciliation | deferred | Высокая инженерная сложность для текущего этапа |
| Deep learning family | deferred | Имеет смысл только после стабилизации сильного табличного baseline |
| Demo web alignment | deferred | Не относится к исследовательскому приоритету проекта |

## 7. Practical Reading Order

Если нужно быстро войти в контекст проекта, читать в таком порядке:

1. `reports/report.md`
2. `src/experiments_v2/STATUS.md`
3. `src/experiments_v2/REGISTRY.md`
4. `src/experiments_v2/EXPERIMENTS.md`
5. каталоги ключевых экспериментов: `43`, `45`, `40`, `03`, `60`, `61-63`, `66`

## 8. Update Rule

При появлении нового значимого эксперимента:

1. Сначала обновлять этот файл по смыслу результата
2. Потом дополнять narrative в `EXPERIMENTS.md`
3. При необходимости обновлять `INDEX.md`, если изменился состав артефактов
