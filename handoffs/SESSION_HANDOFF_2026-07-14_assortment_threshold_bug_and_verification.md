# Session Handoff — 2026-07-14 — Assortment-Threshold Bug Fix and Live Verification

## Scope

Direct continuation of `handoffs/SESSION_HANDOFF_2026-07-13_sku_hour_fallback_profile_fix.md`.
That session ended with two fixes deployed but unexercised — waiting for
the 2026-07-14 03:30 UTC nightly timer. This session:

1. Checked the nightly timer's actual result.
2. Found and fixed a real bug that was the true cause of the
   assortment-threshold problem (not just "hasn't run yet" as previously
   documented).
3. Manually triggered a full production run to verify both fixes live,
   rather than waiting for the next timer.
4. Confirmed both fixes work, with one important caveat on the
   assortment fix's blast radius.

## Checking the overnight state

VM time was `2026-07-14 06:59 UTC`; the nightly timer had already fired
at `03:30 UTC` and completed successfully (`prod_base_bakery_no_sku_uplift_20260714_h14`
active). This was the assortment-threshold code's first-ever real
execution (per the 2026-07-13 finding that it had never run before) — and
it failed on the first try:

```
Assortment refresh FAILED: unsupported operand type(s) for -: 'str' and 'datetime.date'
```

## Root-causing and fixing the assortment bug

Read `scripts/build_city_assortment_from_sales.py:build_layers()` fully.
Found: `combined["valid_from"] = pd.to_datetime(valid_from).date().isoformat()`
— produces a Python `str`. This function's output DataFrame gets inserted
directly into ClickHouse via `client.insert_df()`, into a `Date`-typed
column. `clickhouse-connect`'s Date serializer does `(value - epoch).days`
per cell — raises exactly the observed error when `value` is a string
instead of a `datetime.date`.

Confirmed this diagnosis is right by reproducing the exact traceback
before attempting any fix:
- First tried reproducing directly against the real `bakeable_products`
  table with a clearly-tagged sentinel test row and immediate cleanup —
  **the auto-mode safety classifier correctly blocked this** (writing to
  shared production state for an ad-hoc hypothesis test, when a
  documented dev environment exists for exactly this).
- Re-ran the reproduction against a throwaway `Memory`-engine scratch
  table via `.env.dev` (the documented `_dev`-suffixed ClickHouse
  environment) instead — got the identical traceback with the buggy
  (string) value, and a clean insert with the fixed (real `date`) value.

Also discovered the file itself isn't some other session's unreviewed
WIP as previously assumed — `scripts/build_city_assortment_from_sales.py`
is already-committed, shipped code (`71465a1`, the 2026-07-06 "sales-based
bakeable assortment" feature). The VM's copy only *looked* uncommitted
because the VM's own git HEAD (`2c38e80`) predates that commit — this is
the same VM-git-drift issue already flagged in `CURRENT_STATE.md`'s
"Known issue" note, not a second, separate problem.

### Fix

`scripts/build_city_assortment_from_sales.py`: changed
`.date().isoformat()` to just `.date()`, keeping `valid_from` a real
`datetime.date` object. Added `tests/test_build_city_assortment_from_sales.py`
asserting this stays a date, not a string. Committed `1b29184`, pushed to
`origin/master`.

## Deploy and manual verification

Deployed via the same SFTP+backup+preflight pattern established in the
2026-07-13 session (VM git is still blocked, see "Known issue" note):
backed up `scripts/build_city_assortment_from_sales.py.bak_20260714_073303`,
uploaded the fixed file, verified `py_compile` and a live import
confirming the buggy `.date().isoformat()` pattern is gone.

Rather than waiting for the 2026-07-15 nightly timer, manually triggered
`systemctl start forecast-production.service` (explicit user
confirmation obtained first) — a full ~9-minute run that regenerates and
re-activates the day's forecast for every bakery.

### Results — both fixes confirmed working

**Assortment (this session's fix):**
```
Assortment refresh: city=318 bakery=2170 inserted=2488 valid_from=2026-07-13
```
No more `FAILED`. For Казань specifically, all 5 originally-flagged
low-share SKUs (product 5105/10670/10628/5106/11213) now resolve to
`scope='bakery'` via the new `sales_window` source — correctly excluded
from the 80% city-wide threshold, present only because a specific bakery
sells them.

**Important — wider effect than expected:** `sales_window` rows landed
for all 9 cities in one run (`318` total city-scope rows), each dated
`valid_from=2026-07-13` — newer than the old `forecast_category_filter`/
`partner_baking_markup` rows (last updated `2026-06-30`).
`get_bakeable_products()` selects by "freshest `valid_from` per city," so
this immediately switched **every city's** served assortment from the
old, unfiltered ~110-product set to the new ~52-product threshold-checked
city layer plus per-bakery additions. This is the fix working as
intended, but it's a live, immediate change across the whole embedded
app — not confined to bakery 16. Flagged in both `CURRENT_STATE.md` and
`DECISIONS.md` for whoever notices SKUs missing from a plan over the
next few days.

**SKU-hour fallback (2026-07-13's `e3f39e6`, re-verified on this fresh
run):**

| SKU | Before fix | After fix (this run) | Actual (30d avg) |
|---|---|---|---|
| Пирог с Манго (11465) | 0.043/day, 1 hour (22:00) | 2.97/day, 3 hours (7-12) | ~6.9/day |
| Роллы Вулкан с курицей (11213) | 0.30/day, 1 hour (22:00) | 0.048/day, 16 hours (6-21) | ~2.0/day |

Both SKUs' forecasts are now properly spread across real active hours
instead of dumped into a single near-dead hour — the specific bug is
fixed. Both still under-forecast relative to actual demand, which is a
**separate, not-yet-investigated** issue in the recent-sales correction
blend formula (SKUs whose recent share falls below both the "runner"
0.5% and "core" 1% boost thresholds get no lift at all, per the
2026-07-13 investigation) — flagged as a follow-up, not fixed this
session.

## Pending Issues

- **Watch over the next few days**: the assortment-threshold fix's wide
  blast radius (all 9 cities' assortments shifted at once) means some
  bakery, somewhere, may have a SKU quietly vanish from its baking plan
  because it no longer clears either the city threshold or has its own
  bakery-scope row. This is expected/correct behavior, not a regression
  — but will look like one without this context.
- **Recent-correction under-forecast for thin SKUs** (Пирог с Манго,
  Роллы Вулкан с курицей): both still forecast well below actual demand.
  Likely needs a lower-tier boost threshold in
  `_build_recent_correction_targets` (`apply_bakery_profiles_clickhouse.py`)
  for SKUs between roughly 0.1% and 0.5% recent share. Not investigated.
- Carried over from 2026-07-13, still open: VM git-pull is blocked
  (root-owned `docs/ops/*.md`, VM HEAD stuck at `2c38e80`); 4 pre-existing
  unrelated test failures flagged in background task `task_b87fbf4a`.

## Commits

| Hash | Message |
|---|---|
| `1b29184` | fix: keep bakeable_products.valid_from a real date, not a string, in build_layers |
| `6376930` | docs: record SKU-hour fallback fix deploy and assortment-threshold audit findings (2026-07-13, prior session) |
| `e3f39e6` | fix: exclude single-observation outlier rows from SKU-hour fallback profile (2026-07-13, prior session) |
