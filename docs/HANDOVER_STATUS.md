# Handover status

Last verified: 2026-11-20 (read-only audit)

This file records blockers to an independent developer handover. It is not a
replacement for `docs/ops/CURRENT_STATE.md`; live production state must always
be verified again immediately before a deployment.

## Current blockers

### 1. Forecast publication failed on 2026-11-20

The forecast VM timer is enabled and active. Its 2026-11-20 base source run
completed, but the weighted-weekday post-processing step failed before a new
served run could be activated.

The failure is a ClickHouse schema-contract mismatch: the production query for
`fct_write_offs` groups by `write_off_doc_num`, while the current table does
not expose that column. The database returned `UNKNOWN_IDENTIFIER` (code 47).

Required resolution before formal handover:

1. Inspect the current `fct_write_offs` schema and the source-system business
   key with the data owner.
2. Update the deduplication query to use the verified stable document key, or
   restore the intended column through a reviewed migration.
3. Add a schema-contract test/health check that fails before publication.
4. Run the forecast as a controlled recovery, verify the active run and all
   daily/hourly conservation checks, and document the result in
   `docs/ops/CURRENT_STATE.md`.

Do not attempt a blind rerun or activate the incomplete base source run.

Tracked in [issue #2](https://github.com/Almazziastinov/demand-forecasting-model/issues/2).

### 2. CI lint baseline is red

The configured repository-wide Ruff command reports 3,094 existing E/F/W
violations as of 2026-11-20. This prevents CI from serving as an effective
merge gate. Targeted tests for the recently added operating-hours and regional
assortment code passed (6 tests), but that is not a replacement for a green
baseline.

Required resolution:

1. Create a dedicated lint-debt PR/series, or explicitly scope the CI lint job
   to supported production modules while keeping a separate debt tracker.
2. Keep changed files lint-clean in every interim PR.
3. Once CI is green, enable branch protection requiring its success.

Tracked in [issue #3](https://github.com/Almazziastinov/demand-forecasting-model/issues/3).

### 3. Repository and deployed Blackhole code require reconciliation

The local worktree contains unrelated uncommitted application, pipeline, test,
and operations-document changes. The operations record also notes targeted
regional fixes on Blackhole that may not be identical to this checkout.

Required resolution:

1. Review the worktree diff and split it into independently testable pull
   requests.
2. Compare each deployed Blackhole file with the corresponding reviewed Git
   revision; preserve or commit every intentional difference.
3. Tag the verified revision after the deployed source and `master` agree.

Tracked in [issue #4](https://github.com/Almazziastinov/demand-forecasting-model/issues/4).

## Positive checks from the audit

- The GitHub repository is public and the current operator has administrator
  access.
- The production forecast timer is enabled and active.
- Targeted tests passed: `tests/test_operating_hours.py` and
  `tests/test_weighted_weekday_regional_scope.py` (6/6).
- `master` currently has no branch-protection rule. Enable it only after the
  CI baseline is made green, otherwise normal development will be blocked.
