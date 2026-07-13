# Session Handoff - 2026-07-13 - Rolling Bakery-Day Bias Correction

## Scope

User reported systematic underforecast on two pilot bakeries (Парковая 7 =
bakery 21, Парина 6 = bakery 89) for 2026-07-06..11, visible in the embedded
app's weekly forecast-vs-fact cards. Root-caused, fixed, validated on dev,
deployed to prod, and started a prod lead-1 history backfill with the fix.

## Root Cause

`models/bakery_day_bias.json` is a **one-time snapshot** of
`mean(actual - forecast)` per bakery, computed once from the June holdout at
training time and applied unconditionally to every forecast forever
(`forecast_final = forecast_base + bias.json[bakery_id]` in
`apply_bias_correction()`, `src/experiments_v2/bakery_day_forecast.py`).

After the 2026-07-06 bakery-day model retrain (`bakery_sales_lag365` added,
see `SESSION_HANDOFF_2026-07-06_bakery_day_lag365_retrain.md`), the model's
own behaviour shifted, but the static correction did not — it kept applying
June-derived constants to a different July model. Confirmed directly from
live ClickHouse (`bakery_forecast_day_snapshots.forecast_base` vs
`.forecast_final` on the already-active prod run), not from a backtest
reconstruction:

- Bakery 89 (Парина 6): static correction was a constant **-125.6/day**,
  computed in June when the old model overforecast this bakery by ~8%. The
  retrained July model no longer had that bias, so the stale correction was
  purely harmful.
- Bakery 21 (Парковая 7): static correction was near-zero (+3.3), while the
  retrained model itself persistently underforecasts this bakery by
  5-10% — the static file never had the right number for this bakery either.

Also ruled out: `bakery_sales_lag365` positional-vs-calendar bugs, missing
`fct_check_lines` history, YoY demand decline (all checked directly against
ClickHouse and found not to explain the pattern).

## Fix

New module `pipelines/forecast_publish/rolling_bakery_bias.py`:

- `compute_rolling_bias()` — `mean(actual - forecast_base)` over a trailing
  window (default 7 days) of live lead-1 performance, per bakery.
- `build_effective_bias_table()` — rolling bias where a bakery has `>= 3`
  days of recent history, else falls back to the static snapshot, else `0`.
- `build_rolling_bias_table()` — end-to-end: queries
  `bakery_forecast_day_snapshots` (lead_days=1, table-suffix aware) +
  `mart_sales_60d`, computes, blends.

Wired into `run_production_inference.py` as the **default** behaviour
(`_resolve_effective_bias_path()` builds a fresh effective bias CSV per run
and passes it to the existing `run_forecast_mode`/`apply_bias_correction`
unchanged). Opt out with `--no-rolling-bias-correction` to restore the old
static-only behaviour. Same `bias_clip_pct=0.15` safety cap as before. New
flags: `--rolling-bias-days` (default 7), `--rolling-bias-min-days`
(default 3).

`scripts/build_prod_lead1_model_backfill.py` also extended with
`--use-rolling-bias` (recomputes bias per backfilled date from data strictly
before it — true walk-forward, no lookahead) and `--no-bias-correction`
(raw model, no correction at all) for comparison backfills.

## A Considered-And-Rejected Alternative: Trend Extrapolation

User asked whether a trend-extrapolated correction (linear fit + damped
one-step-ahead projection over a trailing 14-day window) would do better
than a flat rolling mean, on the theory that a flat mean always lags a
persisting trend. Tested empirically on the same real data:

| variant | wMAPE |
| --- | ---: |
| static | 8.11% |
| rolling mean (7d) | 5.84% |
| rolling trend (14d, damped) | 6.09% |

Trend was not better — this transition looks like a **level shift** (from
the retrain), not a sustained trend, and the model already has its own
`bakery_sales_trend` feature, so extrapolating trend in the residual risks
double-counting. Not implemented; noted in `DECISIONS.md` for revisit if a
longer, more clearly trending period shows up later.

## Dev Validation

11-day walk-forward lead-1 backfill on `.env.dev` (`_dev`-suffixed tables),
all 10 pilot bakeries `[20,21,22,28,80,89,107,221,222,257]`,
`2026-07-01..11`, using `scripts/build_prod_lead1_model_backfill.py
--use-rolling-bias --replace-existing`.

**First pass used stale/default weather** (local
`bakery_weather_features.csv` only went through 2026-06-22;
`WEATHER_DEFAULTS["temp_mean"]=10.0` was silently used for every day) —
caught by the user noticing the UI showed a flat 10°C for the whole week.
Refetched real Open-Meteo weather
(`src/experiments_v2/build_bakery_weather_features.fetch_weather_features`)
and reran. Real July was 19-24°C with several rainy days — materially
changed the numbers:

| variant | wMAPE (stale weather) | wMAPE (real weather) |
| --- | ---: | ---: |
| no correction (raw `forecast_base`) | 7.03% | 5.71% |
| rolling (this fix) | 5.62% | 5.79% |
| static (prod at the time) | 8.11% (weather-independent, live prod data) | same |

With real weather, "no correction" and "rolling" are close in aggregate —
the raw retrained model is already decent once it sees real weather. Static
is worse than both for 8/10 pilot bakeries in every variant tested. Rolling
clearly wins for bakeries with a **persistent** (non-weather, non-noise)
bias — bakery 21: wMAPE 6.68% (no correction) vs 4.58% (rolling); bakery 89
went from dramatically-helped-looking under bad weather data to roughly a
wash under real weather (raw model already near-perfect for it).

Decision: keep the rolling correction as the shipped default — not because
it's a huge aggregate win, but because (a) it's clearly better than the
alternative it replaces (static) in every test, (b) it helps the bakeries
that actually have a persistent bias without needing anyone to notice and
manually intervene, and (c) unlike a static snapshot it can't go stale the
same way again.

## Also Discovered (Not This Session's Fix, Flagged For Follow-Up)

**ReplacingMergeTree sort key gap.** `bakery_forecast_day_snapshots`,
`sku_forecast_day_snapshots`, `sku_forecast_hour_snapshots` (prod *and*
`_dev`) are all `ReplacingMergeTree` with `ORDER BY (forecast_date,
lead_days, bakery_id[, product_id[, hour]])` — **`source_run_id` is not
part of the key.** Background merges silently collapse multiple runs
sharing a `(date, bakery[, product[, hour]])` key down to one, regardless of
run_id. Confirmed directly: ran two deliberately-parallel dev backfills
(`*_rollingbias_*` and `*_nocorrection_*`, same 11 dates) and watched 9 of
11 days of the first variant disappear from ClickHouse within about an hour
as merges caught up, while the 2 most-recently-written days still had both.
This likely also explains earlier-observed "run_id mixing" in ad-hoc
historical lead-1 queries throughout this project. Needs its own decision
(adding `source_run_id` to the sort key means a full table rebuild) —
**not resolved**, only documented in `DECISIONS.md`.

**VM working-tree drift + docs/ops permission issue.** Discovered while
trying to `git pull` on the deploy VM (`root@201.51.7.24:/opt/demand-forecasting-model`):

- `docs/ops/*.md` are `root:root` owned — the `forecast` user (who does
  `git pull` in `deploy/vm/deploy.sh`) can't unlink/replace them, so a
  normal `git pull --ff-only` fails outright.
- Separately, the working tree had substantial uncommitted drift around
  `apps/forecast_embedded/` and `apps/baking_plan/` — turned out to be a
  **different, concurrent agent session** (user-confirmed) actively
  deploying baking-plan changes to this same VM in parallel (`origin/master`
  gained 3 new commits — `e3f39e6`, `4a4fa74`, `6e27bd9` — mid-session).
  Confirmed settled (no running processes, no recently-modified files, VM
  timestamps ~1h37m stale) before proceeding.
- Worked around by **not** attempting `git pull`: SFTP'd only the 3 changed
  files directly to their paths, `chown forecast:forecast`, verified they
  import cleanly, then `systemctl start forecast-production.service`
  directly (bypassing `deploy/vm/deploy.sh`'s git-pull step). VM's `git log`
  therefore does **not** reflect commit `0dcb638` yet — file contents are
  correct and live, but this needs someone to eventually resolve the
  ownership + drift and do a real `git pull` to bring history back in sync.
- Separately hit the same class of problem doing the prod lead-1 backfill
  (below): several `data/processed/*lead1*` scratch/output files from an
  **old 2026-07-01/07-07 backfill run** were `root:root` owned (mode 644,
  not world-writable), blocking the `forecast` user from overwriting them.
  Fixed with a scoped `chown forecast:forecast` + `chmod 664` on exactly the
  `*lead1*202607*` files in `data/processed/` and `reports/` (28 files) —
  did not delete anything, these are disposable intermediate/output files
  regenerated fresh by the script.

## Prod Deploy

1. Committed `0dcb638` (`fix: replace static bakery-day bias correction
   with rolling trailing-window correction`) — pushed to `origin/master`.
   Staged only the 5 files actually touched (`git add` by exact path, not
   `-A`) — the repo working tree has substantial unrelated uncommitted work
   from other sessions (`apps/baking_plan/*`, various `analyze_*.py`,
   `_tmp_*.py` scratch scripts) that was explicitly left untouched.
2. VM `git pull` blocked (see above) — deployed via targeted SFTP of the 3
   changed files (`rolling_bakery_bias.py`, `run_production_inference.py`,
   `build_prod_lead1_model_backfill.py`), `chown forecast:forecast`,
   verified clean import under the VM's venv.
3. `systemctl start forecast-production.service` (manual trigger, ~9 min,
   memory-heavy) instead of waiting for tomorrow's 03:30 UTC timer, since
   the whole point was to fix today's already-live-wrong forecast.
4. `scripts.verify_prod_deploy --env-file .env` → `VERIFY OK`.

New active run: `prod_base_bakery_no_sku_uplift_20260713_h14`
(`generated_at 2026-07-13 18:33:59+03:00`, horizon `2026-07-13..2026-07-26`).

Confirmed the new correction is live by reading `forecast_final -
forecast_base` directly from `bakery_forecast_day_snapshots` for this run:

- Bakery 21: constant `+114.3`/day adjustment across the whole 14-day
  horizon (a single run computes the rolling bias once, as of the run's
  `as_of_date`; it refreshes again on tomorrow's run) — vs the old
  near-zero static value.
- Bakery 89: `-5.2`/day — vs the old `-125.6`.

## Docs

`docs/ops/CURRENT_STATE.md` and `docs/ops/DECISIONS.md` were updated with
full detail (deploy method, dev validation numbers, the ReplacingMergeTree
and VM-drift findings) and committed **locally only** as `a1b99f6`
(`docs: record rolling bakery-day bias correction deploy (2026-07-13)`) —
**not pushed**. The push of the first attempt at this commit was denied by
the Claude Code auto-mode classifier ("Excess Sensitive Detail" — VM paths,
systemd unit names, the SFTP-bypass workaround, internal service names).
User explicitly chose to keep it local-only rather than trim the detail and
push. **Next session: decide whether to push as-is (the existing docs/ops
files already carry comparable detail) or trim first.**

## Prod Lead-1 Backfill (Started, May Still Be Running)

User asked to also rebuild lead-1 history on prod (real tables, not `_dev`)
with the new rolling correction, for dates `2026-07-01..12` (matching the
dev validation window). Existing lead-1 snapshots for that range already
existed (built with the old raw-uplift/static-corrected models) — this
overwrites them with `--replace-existing`.

Command (run via SSH as `forecast`, in the background):

```bash
cd /opt/demand-forecasting-model && sudo -u forecast .venv/bin/python \
  scripts/build_prod_lead1_model_backfill.py \
  --env-file .env \
  --date-from 2026-07-01 --date-to 2026-07-12 \
  --dataset-path data/processed/bakery_daily_sales.csv \
  --model-path models/bakery_day_model.joblib \
  --meta-path models/bakery_day_meta.joblib \
  --bias-path models/bakery_day_bias.json \
  --uplift-profile-version weekly_20260701 \
  --use-rolling-bias \
  --replace-existing \
  --summary-path reports/prod_lead1_rolling_bias_backfill_summary.json
```

Failed twice on stale `root:root`-owned scratch files from an old backfill
run before the ownership fixes described above; currently running clean.
**As of writing this handoff, only `2026-07-01` had completed** (run id
`backfill_base_bakery_no_sku_uplift_rollingbias_20260701_h1`, 211 bakeries,
no errors) — the remaining `2026-07-02..12` were still in progress in a
background process. At ~6 min/day this needs roughly another hour from
`07-02` to finish.

**Next session: check whether this backfill completed.** Verify with:

```sql
select forecast_date, source_run_id, count() as n
from bakery_forecast_day_snapshots
where lead_days = 1
  and source_run_id like 'backfill_base_bakery_no_sku_uplift_rollingbias_%'
  and forecast_date between '2026-07-01' and '2026-07-12'
group by forecast_date, source_run_id
order by forecast_date
```

Expect 12 rows (one run_id per date), `n=211` each. If some dates are
missing, resume with the same command using only the missing date range
(the script's `_existing_lead1_dates()` skip-check is not
`_dev`/table-suffix-aware, so with `--replace-existing` omitted it would
incorrectly check the prod-unsuffixed table regardless of `--table-suffix`
— not a problem for this prod run, but worth knowing if reused against dev
again). If it's still running, either wait or just re-run for the
remaining dates — it's idempotent per date with `--replace-existing`.

This backfill is **cosmetic/historical only** — it does not affect the live
active forecast (already fixed and verified above), only how far back the
"fact vs forecast" history in the embedded app reflects the new correction.

## Code Changed (Committed, Pushed)

- `pipelines/forecast_publish/rolling_bakery_bias.py` (new)
- `pipelines/forecast_publish/run_production_inference.py`
- `scripts/build_prod_lead1_model_backfill.py`
- `tests/test_rolling_bakery_bias.py` (new, 5 tests)
- `.claude/launch.json` (added a `forecast_embedded_dev` preview config,
  port 3001, points at `.env.dev` — used to visually verify the fix in the
  embedded UI against real dev data during this session)

## Verification Already Done

```bash
pytest tests/test_rolling_bakery_bias.py tests/test_bakery_day_forecast.py -q   # 13 passed
ruff check pipelines/forecast_publish/rolling_bakery_bias.py pipelines/forecast_publish/run_production_inference.py scripts/build_prod_lead1_model_backfill.py tests/test_rolling_bakery_bias.py --select=E,F,W   # all checks passed
```

Full local suite has 4 pre-existing unrelated failures
(`test_apply_bakery_profiles_clickhouse_recent.py`,
`test_daily_profile_blending.py`) — confirmed via `git diff` that this
session touched none of those files; not investigated further, out of
scope.

Production: `scripts.verify_prod_deploy --env-file .env` → `VERIFY OK`.
Live `forecast_final - forecast_base` spot-checked for bakeries 21/89 on
the new active run (see above).

## Immediate Next Steps

1. Check whether the prod lead-1 backfill (`2026-07-01..12`) finished; if
   not, resume it (see query above).
2. Decide whether to push the local-only docs commit `a1b99f6` as-is, or
   trim internal detail first.
3. Someone should resolve the VM's `docs/ops/*.md` root ownership and the
   concurrent baking-plan working-tree drift, then do a real `git pull` on
   the VM so its `git log` reflects `0dcb638` (and whatever landed after).
4. `ReplacingMergeTree` sort-key gap on the three snapshot tables (prod and
   dev) needs a decision — not urgent, but be aware historical multi-run
   lead-1 comparisons in ClickHouse can silently lose older runs to merges.

## Do Not Do

- Do not run production forecast generation on Blackhole.
- Do not enable Blackhole forecast timers.
- Do not activate `backfill_*_h1` runs as the main run.
- Do not `git reset --hard` / `git checkout -- .` on the VM's
  `/opt/demand-forecasting-model` working tree without first understanding
  whose uncommitted changes are there — this session found live,
  in-progress work from a different concurrent agent session.
- Do not print `.env`, ClickHouse credentials, VibeCode keys, VM SSH
  password, or the contents of `.codex/prod_vm.env`.
