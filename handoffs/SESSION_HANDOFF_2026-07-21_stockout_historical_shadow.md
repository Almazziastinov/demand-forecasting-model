# Session Handoff — 2026-07-21 — Stockout Historical Shadow

## Boundary

Continues `SESSION_HANDOFF_2026-07-20_stockout_direction_offline_pipeline.md`.
The user approved the next stage: a historical walk-forward shadow before
redesigning allocation. Production remained explicitly out of scope.

All ClickHouse access was read only. No production state, service, timer,
profile, active run, environment variable, or deployment was changed.
`docs/ops/` remains the operational source of truth.

## Implemented

- Added `scripts/analyze_stockout_historical_shadow.py`.
- Added the historical step to `scripts/run_stockout_direction_shadow.py`.
- Added recurrence settings to `config/stockout_direction_shadow.json`.
- Added tests for date cutoff, demand-loss-only adjustment, empty calendar
  days, recurrence, and top-5 sales ranking.
- Generated daily, weekly, bakery, SKU, and bakery×SKU stability artifacts.
- Embedded the historical result into the main shadow manifest.

The analyzer consumes local shadow artifacts and has no ClickHouse client.
The upstream classifier uses prior same-weekday values only; reconstruction
uses dates strictly before each evaluated case.

## Result

Available confirmed-case history is `2026-06-01..2026-07-19`: 49 days / 7
weeks, shorter than the requested 8–12 weeks.

| Metric | Result |
| --- | ---: |
| Cases | 397 |
| Robust allocation / demand loss / uncertain | 231 / 26 / 140 |
| Adjusted cases | 25 |
| Reconstructed demand | 140.313 |
| Shortfall before / after | 1,509.285 / 1,453.585 |
| Reduction | 55.700 / 3.69% |
| Fixed / improved / worsened | 23 / 25 / 0 |

Demand loss appears in every available week, but roughly 75% of the shortfall
reduction comes from two weeks. It is persistent but narrow and episodic.

Recurrence using at least two cases in at least two weeks:

- 4 bakeries with recurrent demand loss: 107, 80, 222, 221;
- 4 recurrent demand-loss SKU: 11474, 100, 11301, 4424;
- 2 recurrent exact demand-loss pairs, both at bakery 107: products 100 and
  11474;
- 52 recurrent allocation pairs;
- 4 recurrent problematic allocation pairs are top-5 by observed sales in
  their bakery.

The latest full shadow refresh also changed the current-profile diagnostic
slightly because it reads the current active profile. Its shortfall is now
1,009.762 rather than 1,005.857. It remains look-ahead diagnostic evidence and
is not eligible for shadow promotion.

## Decision

- Keep conservative demand-loss preprocessing in local shadow.
- Do not retrain the bakery model yet: the reconstructed volume is too small.
- Do not promote or retune the rejected allocation model against noisy daily
  share labels.
- Use the 52 recurrent allocation pairs as the evaluation population for a
  new target: smoothed non-stockout share regime plus guarded residual,
  preserving bakery total.
- Historical days do not count toward the required 21 prospective shadow
  days.

The confirmed-miss dataset cannot measure false uplift on normal days, and
true censored demand is unobserved. Synthetic validation, manual review, and
prospective shadow remain required before any production proposal.

## Verification

- Full read-only shadow runner completed successfully in about 40 seconds.
- Targeted stockout tests: `10 passed`.
- Ruff: clean for changed Python files.
- Production writes: none.

## Artifacts

- `docs/stockout_historical_shadow_results_20260721.md`
- `reports/stockout_historical_shadow/summary.json`
- `reports/stockout_historical_shadow/weekly_stability.csv`
- `reports/stockout_historical_shadow/bakery_stability.csv`
- `reports/stockout_historical_shadow/sku_stability.csv`
- `reports/stockout_historical_shadow/bakery_sku_stability.csv`
- `reports/stockout_direction_shadow/manifest.json`

## Reproduction

```powershell
.venv\Scripts\python.exe scripts\run_stockout_direction_shadow.py --env-file .env
```

## Commits

| Hash | Message |
| --- | --- |
| `5e4dd0b` | `feat: add historical stockout shadow replay` |
| `56b6f0e` | `docs: record historical stockout shadow results` |

## Next stage

Build the new allocation experiment around a smoothed, regime-aware
non-stockout share target. Evaluate all recurrent pairs, with separate output
for top-5 SKU and other problematic SKU, and keep the bakery total exactly
constant.
