# Handover status

Last verified: 2026-11-20 (read-only audit)

This file records blockers to an independent developer handover. It is not a
replacement for `docs/ops/CURRENT_STATE.md`; live production state must always
be verified again immediately before a deployment.

## Current blocker

### Repository and deployed Blackhole code require reconciliation

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

## Resolved during handover audit

### CI and protected main branch

- PR [#6](https://github.com/Almazziastinov/demand-forecasting-model/pull/6)
  established the Ruff baseline and made the demo-store test independent of
  unversioned artifacts.
- CI on `master` passed `ruff check src/ web/ tests/` and the full suite:
  537 passed and 3 skipped.
- `master` now requires a pull request, one approval, a passing
  `lint-and-test` check and resolved discussions. These requirements apply to
  administrators too.
- Issue [#3](https://github.com/Almazziastinov/demand-forecasting-model/issues/3)
  is closed.

### Forecast publication failed on 2026-11-20

The forecast VM timer is enabled and active. Its 2026-11-20 base source run
completed, but the weighted-weekday post-processing step failed before a new
served run could be activated.

The failure is a ClickHouse schema-contract mismatch: the production query for
`fct_write_offs` groups by `write_off_doc_num`, while the current table does
not expose that column. The database returned `UNKNOWN_IDENTIFIER` (code 47).

Resolution:

- The current schema has `document_number`, not `write_off_doc_num`. The
  correct deduplication key is `document_number + line_id`.
- The production writer was patched after a timestamped backup; the local
  regression test covers legacy and regional write-off queries.
- A controlled manual recovery activated
  `draft_normalized_demand_v1_20261120_h14` for 2026-11-20 through 2026-12-03.
- The active run has 3,500 bakery-days, 342,595 SKU-days and 4,722,101
  SKU-hours. Maximum bakery-to-SKU and SKU-to-hour reconciliation deltas were
  `1.82e-12` and `8.53e-14` respectively.

Issue #2 is closed. Issue #5 is also closed: `verify_prod_deploy` now
recognizes an inactive source run when it is explicitly referenced by the
active weighted-weekday run. The corrected checker was deployed to the VM and
returned `VERIFY OK` for the active run above.

### Blackhole baking-plan write-off schema alignment

The deployed Blackhole `/opt/baking_plan/simple_plan.py` had the same obsolete
`write_off_doc_num` reference. It was backed up to
`/opt/backups/baking_simple_plan_before_writeoff_schema_20261120.py`, changed
to `document_number + line_id`, compiled, and the app was restarted. Its
`/health` endpoint returned production OK.

This was a targeted live repair. The complete Blackhole source is still not a
Git working tree and its larger modules differ from this local checkout; it
remains covered by issue #4.

## Positive checks from the audit

- The GitHub repository is public and the current operator has administrator
  access.
- The production forecast timer is enabled and active.
- Targeted tests passed: `tests/test_operating_hours.py` and
  `tests/test_weighted_weekday_regional_scope.py` (6/6).
- `master` currently has no branch-protection rule. Enable it only after the
  CI baseline is made green, otherwise normal development will be blocked.
