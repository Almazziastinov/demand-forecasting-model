# Session Handoff - 2026-06-17 - Git Actual State After Smoothing

## Scope

This handoff supersedes the stale parts of the 2026-06-16 handoffs by checking
the actual git state and local generated artifacts.

Current local branch:

```text
master == origin/master == origin/HEAD
```

Current HEAD before this handoff commit:

```text
4d1ac31 fix: make sku hour smoothing chunk-safe
```

The previous handoff `SESSION_HANDOFF_2026-06-16_pilot_uplift_chunk_fix.md`
described the chunk-safe smoothing fix as prepared and the heavy rebuild as
interrupted. That is now outdated.

## What Is Already Done In Git

### Dev environment

Commit:

```text
41c0a3b feat: add forecast dev environment
```

This introduced:

- `.env.dev` workflow;
- `_dev` ClickHouse serving tables;
- `scripts/dev_run_embedded_api.ps1`;
- `scripts/dev_run_inference.ps1`;
- dev UI badge;
- table-name suffixing for embedded and publish paths.

### Chunk-safe SKU-hour smoothing

Commit:

```text
4d1ac31 fix: make sku hour smoothing chunk-safe
```

This committed:

- `src/experiments_v2/smooth_sku_hour_share_profile.py`
- `src/experiments_v2/apply_bakery_profiles.py`
- `reports/dev_pilot_lead1_audit/tukaya_uplift_data_audit.md`
- `handoffs/SESSION_HANDOFF_2026-06-16_pilot_uplift_chunk_fix.md`

The technical root cause was chunk-local normalization in
`smooth_sku_hour_share_profile.py`. If a `date x bakery_id x hour` group was
split across pandas chunk boundaries, each chunk part normalized separately to
`1.0`. For Tukaya 62A and Sibirsky Trakt this produced norm sums near `2.0`,
which suppressed uplift multipliers.

The committed fix uses a two-pass approach:

1. first pass computes adjusted shares and writes an intermediate adjusted file;
2. global denominators are aggregated by `date x bakery_id x hour`;
3. second pass computes final normalized adjusted shares from global
   denominators.

## Local Generated State After The Commit

The expensive smoothing rebuild was later completed locally.

Generated files now present:

```text
data/processed/sku_hour_share_profile_daily_smoothed.csv
data/processed/sku_hour_share_profile_smoothed.csv
data/processed/sku_hour_share_profile_smoothed_summary.json
```

Timestamps observed:

```text
sku_hour_share_profile_daily_smoothed.csv   2026-06-17 09:47
sku_hour_share_profile_smoothed.csv         2026-06-17 09:50
sku_hour_share_profile_smoothed_summary.json 2026-06-17 09:50
```

Current summary:

```json
{
  "profile_rows": 3856275,
  "applied_rows": 36365329,
  "bakeries": 217,
  "products": 1331,
  "mean_norm_share_sum": 1.0,
  "mean_uplifted_row_rate": 0.436985,
  "mean_share_uplift_raw": 0.005242
}
```

This means the local generated profile is no longer in the interrupted/missing
state described in the prior handoff.

## Current Dev Runs And Reports

The dev publish path has already used the rebuilt profile/uplift tables.

Current `reports/dev_production_inference_summary.json` records:

```text
profile_table: sku_hour_share_profile_smoothed_embedded_dev
uplift_table: sku_hour_uplift_multiplier_embedded_dev
uplift_profile_version: dev_chunk_safe_smoothing_20260617
recent_correction_mode: runner_city_prior_soft_weekpart
run_id: dev_uplifted_bakery_norm_uplift_sku_20260617_h14
activated: true
```

The local dataset refresh for that run covered:

```text
history_start_date: 2025-01-01
history_end_date: 2026-06-16
daily_aggregate_rows: 89717
weather_status: refreshed
```

Additional 2026-06-17 dev reports exist:

```text
reports/dev_retrained_uplift_lead1_summary_20260617.json
reports/dev_allowlist_lead1_summary_20260617.json
reports/dev_oldmodel_allowlist_lead1_summary_20260617.json
reports/dev_bakery_day_model_uplifted_retrain_summary_20260617.json
reports/dev_bakery_day_model_uplifted_allowlist_summary_20260617.json
reports/dev_problem_bakery_audit_retrained_uplift/
reports/dev_problem_bakery_audit_allowlist/
```

These reports are generated/ignored artifacts and are not part of this commit.

## Important Finding After The Fix

The chunk-safe fix is technically correct, but a full switch to the fixed uplift
profile creates a modeling issue for the problem bakeries `22` and `222`.

Comparison file:

```text
reports/dev_problem_bakery_audit_allowlist/summary_metrics_with_oldmodel_allowlist.csv
```

Observed lead-1 comparison for 2026-06-01..2026-06-14:

| bakery_id | bakery | scenario | bias_pct | total_wmape_pct | sku_wmape_pct | allocation_wmape_pct |
|---:|---|---|---:|---:|---:|---:|
| 22 | Sibirsky Trakt 25 Kazan | fixed_all_retrain | 22.82 | 22.82 | 36.83 | 28.62 |
| 22 | Sibirsky Trakt 25 Kazan | oldmodel_allowlist | 0.57 | 0.57 | 29.47 | 28.62 |
| 222 | Gabdully Tukaya 62A Kazan | fixed_all_retrain | 26.44 | 26.44 | 49.71 | 38.21 |
| 222 | Gabdully Tukaya 62A Kazan | oldmodel_allowlist | 0.13 | 0.13 | 39.87 | 38.21 |

Interpretation:

- the full fixed profile restores uplift for `22` and `222`, but over-uplifts
  bakery totals by roughly `+23..26%`;
- the best current local result for those two stores is the
  `oldmodel_allowlist` hybrid;
- this hybrid appears to be an experimental dev-table/profile version, not a
  committed production feature.

## Current Working Tree Before This Handoff Commit

Intentional small local code/config changes:

```text
M .gitignore
M scripts/dev_run_inference.ps1
```

`.gitignore` adds:

```text
data/raw/bakery_daily_sales_clickhouse.csv
```

`scripts/dev_run_inference.ps1` now explicitly passes suffixed dev profile and
uplift tables:

```text
--profile-table sku_hour_share_profile_smoothed_embedded$FORECAST_TABLE_SUFFIX
--uplift-table sku_hour_uplift_multiplier_embedded$FORECAST_TABLE_SUFFIX
```

Reason:

- dev inference must use `_dev` profile/uplift tables;
- otherwise a dev run can accidentally read shared/prod-named profile tables.

Important untracked files/directories:

```text
.codex/
notebooks/bakery_fact_vs_forecast_review.ipynb
```

Do not commit `.codex/`; it may contain environment/secrets. The notebook is a
local analysis artifact and should be cleaned before any future commit.

The older untracked handoff:

```text
handoffs/SESSION_HANDOFF_2026-06-16_git_actualization.md
```

is stale and should not be treated as current source of truth.

## Current Stage

We are no longer at "fix chunk-safe smoothing" stage. That part is complete in
git and the local heavy rebuild has already run.

Current stage:

1. Decide how to turn the `oldmodel_allowlist` result for bakeries `22` and
   `222` into a reproducible pipeline option, or decide not to use it.
2. If using it, implement explicit code/config support rather than relying on
   manual dev profile table manipulation.
3. Re-run focused dev audit for the pilot/problem bakeries after the chosen
   strategy is encoded.
4. Only after that consider VM/prod rollout. Prod was not changed during this
   local investigation.

## Suggested Next Implementation Direction

Make the hybrid explicit and testable:

- add a named uplift profile strategy or override config for selected bakeries;
- keep default chunk-safe smoothing for the general case;
- for `bakery_id in {22, 222}`, compare:
  - fixed chunk-safe profile;
  - old profile/uplift behavior;
  - capped uplift multiplier;
  - bakery-level uplift cap by recent realized sales;
- choose the least manual approach that preserves SKU allocation improvement
  without forcing bakery-total overforecast.

Minimum acceptance checks:

- dev runner reads only `_dev` profile/uplift tables;
- `sku_day == sum(sku_hour)` remains exact;
- no pre-06 spike regression for pilot bakeries;
- `22` and `222` do not get forced `+23..26%` bakery-total uplift unless the
  business intentionally wants that.

## Verification During This Actualization

Commands run locally:

```text
git status --short --branch
git log --oneline --decorate -25
git show --stat --name-status 4d1ac31
git show --stat --name-status 41c0a3b
git diff -- .gitignore scripts/dev_run_inference.ps1
```

Also inspected:

```text
data/processed/sku_hour_share_profile_smoothed_summary.json
reports/dev_production_inference_summary.json
reports/dev_*_summary_20260617.json
reports/dev_problem_bakery_audit_allowlist/summary_metrics_with_oldmodel_allowlist.csv
```

No test suite was run as part of this handoff creation.
