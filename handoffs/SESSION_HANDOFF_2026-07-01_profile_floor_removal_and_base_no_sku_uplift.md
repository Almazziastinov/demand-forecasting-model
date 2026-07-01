# Session Handoff - 2026-07-01 - SKU-Hour Floor Removal, Base-Model Prod Switch, Lead-1 Rebuild

## Scope

- Investigated whether hourly demand dips in the SKU-hour share profile
  reflect shelf-absence (stockout) censoring or genuine low demand.
  Conclusion: no reliable signal found; removed the `max(share, mean)`
  floor-uplift entirely.
- Fixed an OOM crash in the profile builder (VM has only 16GB RAM) by
  vectorizing a per-group Python loop.
- Fixed two bugs in `weekly_profile_refresh.py` surfaced by the first
  successful end-to-end run.
- Reviewed a pending decision from `SESSION_HANDOFF_2026-06-30_base_raw_pilot_evaluation.md`
  and switched prod to a new `base_no_sku_uplift` scenario (base bakery-day
  model, no bakery-level uplift, no SKU-hour uplift multiplier).
- Rebuilt lead-1 backfill for 2026-06-01..2026-06-29 under the new scenario.
- Updated `docs/ops/CURRENT_STATE.md` and `docs/ops/DECISIONS.md`.

## Why The Floor Was Removed

Concept under test: hourly shares below the historical mean might mean the
product was off the shelf (stockout), not that demand was actually lower —
so lifting them to the mean ("floor-uplift") could correct for that.

Signals tried and their results (all inconclusive or negative):

- Rolling sell-through / `closing_stock=0` classification across 10 pilot
  bakeries: 20 CENSORED, 304 AMBIGUOUS, 121 PATTERN out of 445 — not a clean
  split, and `closing_stock=0`/`sell_through>1` both have rational
  non-censoring explanations (leftover management, bakery behavior).
- Dip-depth comparison (CENSORED vs PATTERN groups): 0.794 vs 0.743 — only a
  7% difference, not enough to justify a floor.
- Five intraday signals (last-sale-hour, SKU-zero-while-bakery-active,
  mid-day gap, tail-cutoff-vs-DOW-typical, closing-stock=0): only 1 out of
  969 profile cells flagged as censored across all signals combined.
- Two category-floor formula variants (`cat_share × product_frac` and a
  relative-shape version) both produced values *below* the raw profile due
  to incompatible denominators (raw mean is conditional on sales occurring;
  day-level share includes zero-sales hours) — neither is usable as a floor.

Conclusion: correcting for shelf-absence requires real inventory/shelf
availability data, which is not collected. Any floor based on inferred
patterns from sales data alone is an unjustified upward distortion.

## Code Changes — Profile Pipeline

`src/experiments_v2/smooth_sku_hour_share_profile.py`:

- `build_adjusted_applied_chunk()`: `sku_share_in_hour_adj` is now a
  straight passthrough of `sku_share_in_hour` (was
  `max(raw_share, mean_share)`). The chunked renormalize/rebuild logic is
  otherwise unchanged.

`src/experiments_v2/build_sku_hour_share_profile.py`:

- The per-group `for key, group in applied.groupby(...)` Python loop inside
  `build_sku_hour_share_profile()` was OOM-killing the VM when building the
  profile over the full ~10-month/61M-row history (16GB RAM ceiling hit
  during the loop; `profile_parts` list growing unbounded). Replaced with
  vectorized `groupby().agg()` calls (weighted means via
  `_w_eff`/`_sw_eff` helper columns, separate long-window and recent-window
  aggregations merged on `group_cols`). Verified output matches the old
  row-by-row implementation exactly on synthetic data (all 12 numeric
  columns `np.allclose`).
- `build_from_raw()`: the raw-chunk aggregation loop now flushes/merges
  partial results every 10 chunks (`FLUSH_EVERY`) instead of holding all ~60
  chunks in memory before one final `pd.concat` + groupby.

`scripts/weekly_profile_refresh.py` (two pre-existing bugs, both hit on the
first real end-to-end run of this script — it had never completed before):

- Step 5 passed `--mode load-uplift`; the target script
  (`pipelines/forecast_publish/sku_hour_profile_store.py`) only accepts
  `load-uplift-multipliers`. Fixed.
- Step 5's `--applied-path` pointed at the raw daily file
  (`sku_hour_share_profile_daily.csv`), but `build_uplift_multiplier_frame()`
  needs `sku_share_in_hour_adj`/`sku_share_in_hour_adj_norm`, which only
  exist in the *smoothed* daily file. Added `profile_applied_smoothed` and
  pointed step 5 at it.

Tests: `tests/test_smooth_sku_hour_share_profile.py` updated (the lift test
now asserts passthrough, not floor-lift). `tests/test_build_sku_hour_share_profile.py`
unchanged, all 13 tests in both files pass.

## Profile Refresh Execution On The VM

The VM has 16GB RAM; running the full pipeline needed two attempts:

1. First attempt OOM-killed at the same point every time (chunk 60 of ~61,
   right where the old per-group Python loop ran) — this was *before* the
   vectorization fix was uploaded.
2. After vectorizing, `build_share_profile` completed in 581s (previously
   would hang for 40+ min then OOM). Because `exec_command(get_pty=True)`
   over a long-running SSH channel is fragile (a channel drop kills the
   remote process even though the job is legitimate), long jobs on the VM
   should be started with `nohup ... > log 2>&1 &` from a *separate* shell
   invocation, not the same one that streams output — the streaming
   `exec_command` itself was the single point of failure in the first
   two attempts.

Full pipeline completed for `profile_version=weekly_20260701`:

- `sku_hour_share_profile_smoothed_embedded`: `3,291,510` rows loaded
  (`--truncate`).
- `sku_hour_uplift_multiplier_embedded`: `26,937` rows loaded (`--truncate`,
  after the two script fixes above).

Sanity check on the reloaded profile: `mean_sku_share_in_hour ==
median_sku_share_in_hour` on all 3,291,510 rows — this is **not** new, it's
a pre-existing cosmetic bug (`profile["median_sku_share_in_hour"] =
profile["mean_sku_share_in_hour"]` in the rebuild step, unrelated to the
floor removal). Confirmed via code (`apply_bakery_profiles.py`) that only
`mean_sku_share_in_hour_norm` is consumed downstream — `median_...` is dead
weight, left as-is.

## Prod Scenario Switch: base_no_sku_uplift

Context found in `SESSION_HANDOFF_2026-06-30_base_raw_pilot_evaluation.md`:
a 7-day pilot found `base_raw_uplift` (base bakery model + raw SKU-hour
uplift multiplier) strongly beat prod (`uplifted_norm`) — bias +6.6% vs
+11.9%, wMAPE 35.2% vs 72.2%. A 28-day extended backfill was queued for
confirmation before deploying.

What actually happened this session:

- The 21-day extended backfill (`dev_base_raw_*`, 2026-06-01..2026-06-21)
  had completed successfully (21/21) before this session started.
- Running the planned 28-day comparison
  (`analyze_variants_comparison.py --start 2026-06-01 --end 2026-06-28
  --variants base_raw`) produced **internally inconsistent numbers** — bias
  swinging to +216.7% (prod) / -73.3% (base_raw), base_raw row count ~13k vs
  an expected ~48k. This is unexplained and was **not** used to make the
  decision below — treat that specific comparison run's output as broken
  until someone root-causes it.
- Separately, this session's own investigation concluded the SKU-hour
  uplift multiplier (the "raw uplift" half of `base_raw_uplift`) has no
  evidentiary basis (see floor-removal section above) and should not be
  reactivated in prod.
- Net decision: switch the *bakery-day model* to base (per the 7-day pilot
  signal + manual review), but do **not** turn on the SKU-hour uplift
  multiplier. Neither existing scenario matched this combination.

Added a third scenario to
`pipelines/forecast_publish/run_production_inference.py::SCENARIOS`:

```python
"base_no_sku_uplift": {
    "description": "base bakery forecast + raw SKU-hour profile allocation, no SKU-hour uplift multiplier",
    "run_id_suffix": "base_bakery_no_sku_uplift",
    "dataset_attr": "base_dataset_path",
    "model_attr": "base_model_path",
    "meta_attr": "base_meta_path",
    "bias_attr": "base_bias_path",
    "forecast_name": "bakery_day_forecast_prod_base_no_sku_uplift.csv",
    "output_suffix": "prod_base_bakery_no_sku_uplift",
    "model_version": "bakery_day_lgbm_base",
    "profile_version": "clickhouse_no_sku_uplift",
    "use_raw_uplift_multiplier": False,
},
```

Deployed on the VM:

```bash
cd /opt/demand-forecasting-model && git pull
.venv/bin/python -m pipelines.forecast_publish.run_production_inference \
  --env-file .env --scenario base_no_sku_uplift --activate-run base_no_sku_uplift \
  --horizon-days 14 \
  --notes 'switch to base_no_sku_uplift: base bakery model, raw SKU-hour profile, no SKU uplift multiplier'
```

Result: `base_no_sku_uplift: prod_base_bakery_no_sku_uplift_20260701_h14 active`
(exit 0). Updated VM `.env`:

```text
FORECAST_SCENARIO=base_no_sku_uplift
FORECAST_ACTIVATE_RUN=base_no_sku_uplift
```

so the nightly `forecast-production.timer` keeps using this scenario.
`recent_correction_mode` was **not** changed — it's still
`runner_city_prior_soft_weekpart` / 30 days, applied on top of whichever
bakery-day model and SKU-hour profile are in use, independent of both
switches above.

Gotcha hit during deploy: the VM had uncommitted local edits (from the
SFTP-uploaded floor/vectorization fixes earlier in the session) that
blocked `git pull`. Confirmed via `git diff -w origin/master` that they
were identical to the already-pushed commit content (whitespace/line-ending
only), then `git checkout --` on those 3 files followed by `git pull`
(user-confirmed before running — this is exactly the kind of action the
auto-mode classifier correctly blocks without explicit confirmation).

## Lead-1 Backfill Rebuild (2026-06-01..2026-06-29)

The existing lead-1 snapshots for June were built under the old
`uplifted_norm` scenario before this session's changes (both the profile
floor removal and the scenario switch). Rebuilt them under the new
combination.

Found and fixed a naming bug while doing this: `build_prod_lead1_model_backfill.py`
picked `run_id`/`model_version` from only two branches keyed on
`use_raw_uplift_multiplier`, so a base-model + no-sku-uplift backfill would
have been mislabeled as `backfill_uplifted_bakery_norm_uplift_sku_...`.
Added a third branch keyed on `model_path` resolving to the base model:

```python
model_is_base = Path(args.model_path).resolve() == BASE_MODEL_PATH.resolve()
if use_raw:
    run_id = f"backfill_base_bakery_raw_uplift_sku_{date_part}_h1"
    ...
elif model_is_base:
    run_id = f"backfill_base_bakery_no_sku_uplift_{date_part}_h1"
    model_version = "bakery_day_lgbm_base_lead1_backfill"
else:
    run_id = f"backfill_uplifted_bakery_norm_uplift_sku_{date_part}_h1"
    ...
```

Command used (via `nohup` on the VM, not the same SSH channel that was
streaming output — see the fragility note above):

```bash
.venv/bin/python scripts/build_prod_lead1_model_backfill.py \
  --env-file .env \
  --date-from 2026-06-01 --date-to 2026-06-29 \
  --dataset-path data/processed/bakery_daily_sales.csv \
  --model-path models/bakery_day_model.joblib \
  --meta-path models/bakery_day_meta.joblib \
  --bias-path models/bakery_day_bias.json \
  --uplift-profile-version weekly_20260701 \
  --replace-existing
```

All 29 days completed successfully as `backfill_base_bakery_no_sku_uplift_YYYYMMDD_h1`.
Per-day elapsed time grew from ~150s (early June) to ~364s (June 29) —
no errors, just genuinely slower toward the end (larger history window per
day). Sample loaded rows: `2026-06-29`: bakery=203, sku_day=32291,
sku_hour=341976.

## Suspected (Unconfirmed) Frontend Issue During Backfill

While the backfill was running, the user reported `Internal Server Error`
on the embedded Bitrix24 app
(`https://franshizasvezhar.bitrix24.ru/devops/placement/115/`). Hypothesis:
`load_forecast_run()`'s delete+insert into `sku_forecast_day_snapshots` /
`sku_forecast_hour_snapshots` (the same tables the embedded app's
`_sku_day_source()`/`_sku_hour_source()` union in for `lead_days=1` rows)
raced with live reads from the app during the ~1-hour backfill window.
**Not confirmed** — the backfill finished before this was investigated
further. Re-check the app now that the backfill is done; if it's still
broken, the cause is something else (see the separate
`forecast-production.service` failure below, which is unrelated in timing
but worth ruling out too).

## Unrelated Pre-Existing Bug Found (Not Fixed This Session)

`forecast-production.service` (the nightly systemd timer, 03:30 UTC) has
failed 3 nights running (Jun 29, Jun 30, Jul 01) with:

```text
PermissionError: [Errno 13] Permission denied: '/opt/demand-forecasting-model/reports/production_dataset_refresh_summary.json'
```

This happened *before* today's session started (03:31 UTC vs session start
~06:43 UTC), so it's unrelated to any change here. It means the nightly
automatic dataset/weather refresh has not been running — only the manual
`run_production_inference` runs (today's scenario switch) have kept the
active run current. Needs a permissions fix on that reports file/directory
before relying on the nightly timer again.

## Docs Updated

- `docs/ops/CURRENT_STATE.md`: active scenario section rewritten for
  `base_no_sku_uplift`; new "SKU-Hour Share Profile Floor Removed" section;
  "Base-Raw Variant Evaluation" section marked resolved with the honest
  caveat about the broken 28-day comparison numbers.
- `docs/ops/DECISIONS.md`: two new entries — floor-uplift rejection
  rationale, and the base-model-switch rationale (explicitly noting
  `base_raw_uplift` should not be reactivated without revisiting the
  uplift-multiplier rejection).

## Commits (pushed to `origin/master`)

```text
625605d fix: remove floor-uplift smoothing, vectorize profile build, fix refresh CLI bugs
88a02ce feat: add base_no_sku_uplift production scenario
792253d docs: record SKU-hour floor removal and base_no_sku_uplift prod switch
d4788ab fix: correct run_id naming for base+no-sku-uplift lead-1 backfill
```

## Immediate Next Steps

1. Re-check the embedded app now that the lead-1 backfill has finished;
   confirm whether the `Internal Server Error` was actually caused by the
   concurrent-write hypothesis above or something else.
2. Fix the `forecast-production.service` `PermissionError` on
   `reports/production_dataset_refresh_summary.json` so the nightly timer
   stops failing.
3. Root-cause the broken 28-day `analyze_variants_comparison.py` numbers
   for `base_raw` before trusting that script's output for any future
   decision.
4. If bakery-day-level model quality needs re-evaluation later, compare
   against `base_no_sku_uplift`, not `base_raw_uplift`.
5. Optional/discussed but not started: a recency-based bias correction at
   the bakery-day (LightGBM) level, analogous to
   `runner_city_prior_soft_weekpart` at the SKU-hour level — currently the
   bakery-day bias table (`bakery_day_bias.json`) is static, built once from
   a holdout, not refreshed with recent actual-vs-forecast drift.

## Do Not Do

- Do not run production forecast generation from VibeCode/Blackhole.
- Do not enable Blackhole forecast timers.
- Do not reactivate `base_raw_uplift` without first revisiting the
  SKU-hour uplift-multiplier rejection recorded in `DECISIONS.md`.
- Do not trust the 28-day `analyze_variants_comparison.py --variants
  base_raw` output until the row-count/bias discrepancy is root-caused.
- Do not print `.env`, ClickHouse credentials, VibeCode API keys, or SSH
  private keys.
