# Session Handoff — 2026-07-13 — SKU-Hour Fallback Profile Fix and Assortment-Threshold Audit

## Scope

- Answered a question about where the baking-plan SKU list comes from
  (bakery assortment, city+bakery scope) and reverted the дефрост label
  text back to `"(ночная дефр)"` per user preference (cosmetic).
- User asked to check 6 specific SKUs at bakery 16 (Кулагина 4, Казань)
  showing `Итого = 0` in the baking plan — investigated whether they were
  junk/stale assortment entries.
- Deep-dove two unrelated real problems found during that check:
  1. 6 SKUs (4 pizza items + rolls) in `bakeable_products` with `scope='city'`
     despite real citywide sales share of only 5-30%, far below the
     documented 80% threshold.
  2. "Пирог с Манго" (product 11465) forecast collapsed to `~0.04`/day
     despite steady real sales of `~7`/day.
- Root-caused both by reproducing the actual production functions against
  real ClickHouse data (not guessing), including SSH access to the
  production VM for log/config verification.
- Fixed problem 2 in code, tested, committed, pushed, and deployed to the
  VM via targeted SFTP. Problem 1 turned out to need no code fix — see
  below.

## SKU list source + cosmetic defrost label

Confirmed the baking-plan SKU list comes from
`apps/baking_plan/assortment.py:get_bakeable_products` — union of
`scope='city'` rows for the bakery's city and `scope='bakery'` rows for
that specific bakery, filtered to the 5 bakeable categories, further
filtered to SKUs with a `baking_sku_meta` row. This is what led into the
"why are these 6 SKUs even here" question below.

Reverted `apps/baking_plan/rendering.py:DEFROST_SUFFIX` from
`"(доп. партия на завтра)"` back to `"(ночная дефр)"` per explicit user
request — the post-processing window-placement pass still usually lands
дефрост in the last window in practice, so the "ночная" framing reads
correctly in the common case even though it's no longer a hard guarantee.

## Problem 1: bakeable_products city-scope doesn't reflect the 80% threshold

Investigated by querying `bakeable_products`, `bakery_forecast_day_embedded`
(the threshold's denominator source), and `mart_sales_60d` directly.
Found: all 276 `scope='city'` rows for Казань come from `source =
'forecast_category_filter'` or `'partner_baking_markup'` — old sources
that don't check sales share at all. **Zero** rows exist anywhere with
`source = 'sales_window'` (the script that actually enforces the 80%
threshold, `scripts/build_city_assortment_from_sales.py`, wired into
`production_dataset_refresh.py`).

SSH'd into the production VM (`root@201.51.7.24`, read-only, explicit
user confirmation obtained first) to find out why. Discovered:

- `pipelines/forecast_publish/production_dataset_refresh.py` on the VM is
  **uncommitted, locally modified** relative to its git HEAD (`2c38e80`,
  from July 6) — the assortment-threshold code exists in the file content
  but isn't part of any commit.
- The file's mtime is `2026-07-13 11:46 UTC` — **after** this morning's
  03:30 UTC nightly run. `journalctl -u forecast-production.service` has
  **zero** mentions of "assortment" across its entire history.
- Conclusion: this code has never executed, not even once — it's not
  broken, it just hasn't had a chance to run yet. It belongs to some
  other, concurrent session's in-flight work (not part of this session),
  so it was deliberately left untouched rather than "fixed."
- First real execution: the 2026-07-14 03:30 UTC nightly timer.

**No code change made for this problem.** Flagged in `CURRENT_STATE.md`
and `DECISIONS.md` for whoever checks tomorrow morning.

## Problem 2: SKU-hour fallback profile collapse (fixed)

Root-caused via direct reproduction of the real production functions
against real ClickHouse data — see `docs/ops/DECISIONS.md` (2026-07-13
entry) for the full investigative trail. Summary:

- Reconstructed the pre-correction base daily forecast for product 11465
  at bakery 16: `~6.4`/day — matches real sales, so the raw profile
  *should* have been fine.
- Ran the actual `_build_recent_correction_targets` (recent-sales
  correction, mode `runner_city_prior_soft_weekpart`) on real data: also
  gives `6.39` — the correction step was not the bug, contradicting the
  first hypothesis (a renormalization "squeeze" from other boosted SKUs).
- Found the real cause: this SKU never reaches the tier-1 profile gate
  (`n_days>=8`) in any hour, so it's entirely dependent on the tier-2,
  dow-blind fallback (`load_profile_lookup_frames` in
  `apply_bakery_profiles_clickhouse.py`). That fallback averaged
  `mean_sku_share_in_hour_norm` with **no minimum sample-size filter** —
  a single `n_days=1` row at hour 22 (one Friday sale reading as "100% of
  that near-dead hour") produced an unsmoothed share of `0.5`, pulling
  the fallback's hour-22 share to `0.135` vs `~0.002-0.004` everywhere
  else. Since the whole bakery only sells ~1 unit total at hour 22, this
  SKU's (correctly-scaled) daily total got crushed into a tiny fraction
  of that near-empty pool instead of spreading across its real active
  hours.
- Confirmed systemic: bakery 16 alone has 16 profile rows with `n_days<=2`
  and share > 0.1 (9 at hour 22, 6 at hour 5), affecting at least 8-9
  distinct SKUs, not just the one reported.

### Fix

Added `MIN_FALLBACK_N_DAYS = 3` gate, excluding `n_days` 1-2 rows from the
fallback average, in:
- `src/experiments_v2/apply_bakery_profiles.py` — CSV path
  (`build_sku_hour_profile_fallback`).
- `src/experiments_v2/apply_bakery_profiles_clickhouse.py` — the
  production ClickHouse path (`load_profile_lookup_frames`), actually
  used by `run_production_inference.py`.

`n_days == 0` is deliberately still trusted as before (a legacy profile
missing the `n_days` column entirely defaults to 0 upstream — that means
"unknown," not "observed zero days," and should keep getting a fallback
estimate rather than being silently dropped — this distinction is what an
existing test, `test_legacy_profile_without_n_days_column_falls_through_gate`,
required).

Added 2 regression tests in `tests/test_apply_bakery_profiles_fallback.py`:
one is a direct reproduction of the real bug shape (single `n_days=1`
outlier vs well-supported rows), the other checks rows right at the new
gate still pass through.

Verified 4 pre-existing, unrelated test failures are untouched by this
change (confirmed via `git stash` — identical failures with or without
the fix): 3 pie-category-cap tests in
`tests/test_apply_bakery_profiles_clickhouse_recent.py` expecting numbers
the current code doesn't produce, and 1 collection error in
`tests/test_build_bakeable_products_table.py` (imports a function that no
longer exists in `scripts/build_bakeable_products_table.py`). Spawned a
background task (`task_b87fbf4a`) to investigate/fix those separately —
out of scope here.

## Deploy Status

| Artifact | Status |
|---|---|
| Code (`e3f39e6`) | ✅ committed, pushed to `origin/master` |
| VM: `apply_bakery_profiles.py` / `apply_bakery_profiles_clickhouse.py` | ✅ SFTP'd to `/opt/demand-forecasting-model/src/experiments_v2/`, backups at `*.bak_20260713_152709` |
| `py_compile` + live import on VM | ✅ both pass, `MIN_FALLBACK_N_DAYS=3` confirmed present |
| **Exercised in a real production run** | ❌ **not yet** — see below |

**Why not yet exercised:** a concurrent session manually restarted
`forecast-production.service` at `2026-07-13 18:33:59+03:00` (for an
unrelated fix, the rolling bakery-day bias correction — see
`docs/ops/CURRENT_STATE.md`), regenerating today's active run just
minutes after this fix's files were replaced on disk. Checked directly
afterward: product 11465 still shows `0.043775`/day, unchanged. Most
likely explanation: that process had already imported the old module
code into its Python interpreter before the SFTP replacement completed —
source files aren't re-read mid-process. **User decision: wait for the
2026-07-14 03:30 UTC nightly timer** rather than force another manual
run today.

## Pending Issues

- **Verify tomorrow morning (2026-07-14, after 03:30 UTC):**
  1. Product 11465 (bakery 16) forecast should now be `~6-7`/day, not
     `~0.04`/day — check `sku_forecast_hour_embedded` against
     `mart_sales_60d` directly, don't just trust that the timer ran.
  2. Whether `build_city_assortment_from_sales.py`'s `sales_window` source
     finally produces `scope='city'` rows for Казань, and whether the 6
     originally-reported SKUs (product_id 5105, 10670, 10628, 5106, 11213)
     drop out of city scope as the 80% threshold intends. This is
     unrelated, uncommitted work from another session — review, don't
     just assume it's correct.
- **VM housekeeping (not done this session, flagged only):** the VM's
  `/opt/demand-forecasting-model` git working tree has uncommitted drift
  (assortment-threshold code, among other things) and `docs/ops/*.md` are
  root-owned, blocking a clean `git pull` as the `forecast` user — see the
  "Known issue (2026-07-13)" note at the top of `CURRENT_STATE.md`. Not
  this session's mess to clean up, but it's actively blocking normal
  git-based deploys for everyone touching this VM.
- Background task `task_b87fbf4a`: fix the 4 pre-existing, unrelated test
  failures found in passing (pie-category-cap test expectations, a
  renamed-function import error).
- Minor: SFTP'ing the two fixed files converted their line endings to
  CRLF on the VM (source was LF, matching git). Doesn't affect execution,
  but `git diff` on the VM will show these two files as "fully changed"
  until someone normalizes them back to LF.

## Commits

| Hash | Message |
|---|---|
| `e3f39e6` | fix: exclude single-observation outlier rows from SKU-hour fallback profile |
