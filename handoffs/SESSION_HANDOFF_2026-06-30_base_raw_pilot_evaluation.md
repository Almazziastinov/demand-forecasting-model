# Session Handoff - 2026-06-30 - Base-Raw Variant Pilot Evaluation

## Scope

Evaluated three experimental lead-1 forecast variants (prior14, bias_corr,
base_raw) for pilot bakeries `[20, 21, 22, 28, 80, 89, 107, 221, 222, 257]`
over the week 2026-06-22..2026-06-28. `base_raw_uplift` significantly
outperformed the current production scenario. An extended backfill was started
to validate over 28 days before prod deploy.

## Variants Tested

| name | description |
| --- | --- |
| `prior14` | uplifted model, recent_correction_days=14 (vs 30 in prod) |
| `bias_corr` | uplifted model, days=30, + rolling SKU bias correction for top-5 SKU |
| `base_raw` | **base (norm) bakery model + raw uplift multiplier on SKU allocation** |

**Note:** `prior14` and `bias_corr` showed data completeness issues (~1800 rows
vs ~10000 expected). Root cause not confirmed — these two variants should not be
used for production decisions until the issue is investigated.

## Initial Results (7 days, 10 pilot bakeries)

```
period: 2026-06-22 .. 2026-06-28
bakeries: [20, 21, 22, 28, 80, 89, 107, 221, 222, 257]
```

| metric | prod (uplifted_norm) | base_raw |
| --- | ---: | ---: |
| bias% | +11.9% | +6.6% |
| wMAPE% | 72.2% | 35.2% |

`base_raw` is strongly better on both metrics. The gap in wMAPE (35% vs 72%)
is large enough to warrant caution — verify with the extended period.

## Extended Backfill (in progress at handoff time)

Running locally as of 2026-06-30:

```
PID:      30544
log:      %TEMP%\backfill_base_raw_extended.log
dates:    2026-06-01 .. 2026-06-21  (21 more days)
variants: base_raw only
script:   scripts/build_dev_lead1_variants.py
```

After it finishes, run the comparison:

```bash
.venv/Scripts/python.exe analyze_variants_comparison.py \
  --start 2026-06-01 --end 2026-06-28 --variants base_raw
```

Check the log tail to confirm it completed:

```powershell
Get-Content "$env:TEMP\backfill_base_raw_extended.log" -Tail 30
```

Expected final line: `COMPLETE / Failed: 0`

## Excel Report

A per-bakery Excel report was built covering 2026-06-22..2026-06-28:

```
outputs/pilot_bakeries_forecast_vs_fact.xlsx
```

Script to rebuild (with updated date range after extended backfill):

```bash
.venv/Scripts/python.exe build_pilot_excel_report.py
```

The script accepts hardcoded dates — update `START_DATE`/`END_DATE` at the top
if you want the full 28-day period in the Excel.

## Infrastructure Fixes Made This Session

### ClickHouse connection-drop on profile streaming

`src/experiments_v2/apply_bakery_profiles_clickhouse.py`:
- `load_profile_lookup_frames(bakery_ids=...)` — added optional bakery filter to
  all three sub-queries (tier1_sums, fallback, thin_triples).
- `stream_profile_chunks(bakery_ids=...)` — same filter on the streaming query.

`scripts/build_dev_lead1_variants.py`:
- After generating the bakery-day forecast, immediately filters `bak_df` down to
  `PILOT_BAKERY_IDS` before passing it to `allocate_from_clickhouse`. This is the
  key fix: the bakery forecast was running on all 216 active bakeries and passing
  all 216 IDs to the profile streaming, which caused 150+ chunks and connection
  timeout. Filtering to 10 bakeries drops to ~10 chunks.

### Dev app visibility

Backfill writes to prod tables (no suffix) because `load_forecast_run` is called
with `.env` (not `.env.dev`). Data was manually copied to `_dev` tables:

```sql
INSERT INTO forecast_runs_embedded_dev
SELECT * FROM forecast_runs_embedded WHERE run_id like 'dev_base_raw_%' or ...;
-- repeated for bakery_forecast_day_embedded_dev, sku_forecast_day_embedded_dev,
-- sku_forecast_hour_embedded_dev
```

## Production Deployment Path

Scenario `base_raw_uplift` already exists in
`pipelines/forecast_publish/run_production_inference.py`. No code changes needed.

**Only deploy if 28-day analysis confirms improvement.**

On the VM (`root@201.51.7.24`, path `/opt/demand-forecasting-model`):

```bash
# Pull latest code first
git pull

# Run base_raw_uplift inference, activate immediately
.venv/bin/python -m pipelines.forecast_publish.run_production_inference \
  --env-file .env \
  --scenario base_raw_uplift \
  --activate-run base_raw_uplift \
  --refresh-datasets \
  --history-start-date 2025-12-01 \
  --notes 'switch to base_raw_uplift after 28-day pilot validation 2026-06-30'

# After activation, rebuild assortment/bakeable tables
.venv/bin/python scripts/build_city_assortment_from_forecast.py --env-file .env
.venv/bin/python scripts/build_bakeable_products_table.py
.venv/bin/python scripts/load_city_assortment_to_clickhouse.py --env-path .env --replace-current
.venv/bin/python scripts/load_bakeable_products_to_clickhouse.py --env-path .env --replace-current

# Verify
.venv/bin/python -m scripts.verify_prod_deploy --env-file .env
```

Expected: `VERIFY OK`

**Note:** this switches ALL bakeries to `base_raw_uplift`, not just the pilots.
There is currently no per-bakery override in the embedded app. If you want a
pilot-only switch, that requires app-level changes (not ready).

## Decision Checklist for Tomorrow

- [ ] Wait for backfill to finish (`Get-Content "$env:TEMP\backfill_base_raw_extended.log" -Tail 30`)
- [ ] Run `analyze_variants_comparison.py --start 2026-06-01 --end 2026-06-28 --variants base_raw`
- [ ] If 28-day bias%/wMAPE% confirm improvement → run prod deploy command above on VM
- [ ] If improvement is marginal or reversed → keep current prod, investigate further

## Files Added/Modified

```
scripts/build_dev_lead1_variants.py          -- backfill script (new)
scripts/build_prod_lead1_model_backfill.py   -- prod lead-1 backfill (new, from prior session)
analyze_variants_comparison.py               -- comparison report (new, now parametrized)
build_pilot_excel_report.py                  -- per-bakery Excel report (new)
outputs/pilot_bakeries_forecast_vs_fact.xlsx -- Excel report 22-28 June (new)
src/experiments_v2/apply_bakery_profiles_clickhouse.py  -- bakery filter fix
docs/ops/CURRENT_STATE.md                    -- updated
```

## Do Not Do

- Do not print `.env`, ClickHouse credentials, VibeCode API keys, or SSH keys.
- Do not run production forecast from VibeCode/Blackhole.
- Do not enable Blackhole forecast timers.
- Do not activate `backfill_*_h1` or `dev_*` runs as the main production run.
- Do not deploy `base_raw_uplift` to prod before checking the 28-day analysis.
