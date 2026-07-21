# Stockout Historical Shadow Results — 2026-07-21

## Boundary

The replay is offline and read only. ClickHouse was queried only by the
existing shadow pipeline; the historical analyzer itself consumes local
artifacts and has no ClickHouse client. No production table, profile, run,
service, timer, environment variable, or deployment was changed.

## Coverage and leakage contract

The requested development window was 8–12 weeks. Confirmed forecast-miss
cases are currently available only for `2026-06-01..2026-07-19`: 49 calendar
days, or 7 complete weeks. The replay uses all available days.

For every evaluated case:

- the bakery and SKU counterfactual uses prior same-weekday observations only;
- demand reconstruction uses dates strictly before the case date;
- the current production profile is excluded from the historical result;
- historical days do not count toward the 21-day prospective shadow gate.

## Aggregate result

| Metric | Result |
| --- | ---: |
| Confirmed model-underforecast cases | 397 |
| Robust allocation cases | 231 |
| Robust demand-loss cases | 26 |
| Uncertain cases | 140 |
| Demand-loss cases adjusted | 25 |
| Reconstructed demand | 140.313 units |
| Baseline confirmed shortfall | 1,509.285 units |
| Demand-shadow shortfall | 1,453.585 units |
| Shortfall reduction | 55.700 units / 3.69% |
| Cases fixed | 23 |
| Cases improved | 25 |
| Cases worsened | 0 |

Demand loss was detected in all seven weeks, so it is not a one-day or
one-week outlier. Its magnitude is not stable: weekly shortfall reduction
ranges from 0.3% to 11.5%. The weeks starting 2026-06-08 and 2026-06-29
produce 41.9 of the total 55.7 reduced units, approximately 75% of the gain.
The mechanism is therefore persistent but narrow and episodic.

## Recurrence

Using a conservative recurrence definition of at least two cases in at least
two distinct weeks:

- 4 bakeries have recurrent demand-loss cases: 107, 80, 222, and 221;
- 4 SKU have recurrent demand-loss cases: Маковка (11474), Губадия (100),
  Клубника и банан НОВЫЙ (11301), and Жар Киш курица (4424);
- only 2 exact bakery×SKU pairs are recurrent demand-loss pairs, both at
  bakery 107: Губадия (100) and Маковка (11474);
- 52 bakery×SKU pairs have recurrent allocation cases.

The distinction is important: demand loss recurs as a general mechanism, but
usually moves between products or bakeries. It should remain a generic
censoring preprocessor with conservative guards, not a hardcoded SKU uplift.

## Top-volume SKU and other problematic SKU

Sales ranks use all observed SKU sales in the evaluation window, not only the
397 failure rows. Four recurrent problematic allocation pairs are in the
top five SKU by sales within their bakery:

| Bakery | SKU | Bakery sales rank | Confirmed shortfall |
| --- | --- | ---: | ---: |
| 257 — Ярмарочная 12 | ЖарПицца Пикантная (10485) | 3 | 69.0 |
| 222 — Габдуллы Тукая 62А | Кыстыбый П (10340) | 2 | 37.9 |
| 22 — Сибирский Тракт 25 | Элеш с курицей (10667) | 5 | 31.1 |
| 221 — Салиха Батыева 15 | Треугольник говядина безд (1076) | 5 | 4.0 |

The output also retains all non-top-5 problematic pairs. Among the strongest
recurrent allocation examples are Жар Киш грибы курица at bakery 221,
Киш грибы курица and Пицца с колбасой at bakery 22, and Бейгл курица at
bakery 257. This produces a concrete target set for the next allocation-model
redesign without restricting analysis to top sellers.

## Decision

The historical replay strengthens the decision to keep conservative
demand-loss preprocessing in local shadow:

- it appears in every available week;
- it improves 25 cases and creates no worsened confirmed-miss case;
- it is too small and concentrated to justify bakery-model retraining or a
  production rollout.

This dataset contains only confirmed misses, so it cannot measure false uplift
on ordinary non-stockout days. True censored demand is also unobserved. The
synthetic reconstruction backtest and at least 21 real prospective shadow days
remain mandatory before any production proposal.

For allocation, the 52 recurrent pairs are sufficient to define the next
experiment population. The rejected model should not be retuned against the
same noisy daily-share target. The next model should estimate a smoothed
non-stockout share regime and predict a guarded residual correction while
preserving bakery total.

## Reproduction

Historical analysis from existing local shadow artifacts:

```powershell
.venv\Scripts\python.exe scripts\analyze_stockout_historical_shadow.py
```

Full read-only refresh and historical analysis:

```powershell
.venv\Scripts\python.exe scripts\run_stockout_direction_shadow.py --env-file .env
```

Primary outputs are under `reports/stockout_historical_shadow/`. The main
shadow manifest now embeds the historical summary under
`historical_walk_forward`.
