# Project handover checklist

Use this checklist before granting independent production access to additional
developers.

## Repository baseline

- [ ] Inspect every current uncommitted change and split it into coherent PRs.
- [ ] Restore a green GitHub Actions baseline. The current repository-wide
      Ruff job has legacy violations; see `HANDOVER_STATUS.md`.
- [ ] Reconcile the checked-in Blackhole app with the deployed files; record
      any intentional drift or move it into a reviewed PR.
- [ ] Tag the verified handover revision, for example `handover-2026-11`.
- [ ] Protect `master`: require PR, passing CI, and one approving review.

## Operational handover

- [ ] On the forecast VM, run the read-only production verification from
      `docs/ops/RUNBOOK.md` and update `CURRENT_STATE.md` with the result.
- [ ] Verify the active run, model version, forecast horizon, and timers.
- [ ] Resolve the current failed weighted-weekday publication before declaring
      the forecast service handed over; see `HANDOVER_STATUS.md`.
- [ ] Verify Blackhole `/health`, its writer timers are disabled, and the app
      reads the same active run.
- [ ] Verify ClickHouse facts and serving tables are reachable from both
      services.
- [ ] Confirm backups and rollback locations are accessible to the designated
      operator.

## Access and ownership

- [ ] Create individual GitHub accounts/team membership for each developer.
- [ ] Create individual ClickHouse accounts: read-only by default, `_dev`
      write scope where required.
- [ ] Create named SSH accounts/keys for the forecast VM; remove shared keys
      after migration.
- [ ] Store production secrets in the approved vault and rotate any secret
      previously shared through a local `.env` or chat.
- [ ] Name primary and backup owners for forecasting, embedded app, data,
      infrastructure, and business access requests.

## Knowledge transfer

- [ ] Each developer completes `docs/DEVELOPER_ONBOARDING.md`.
- [ ] Run one supervised dry-run forecast and one app deployment rehearsal.
- [ ] Demonstrate active-run verification, incident triage, and rollback.
- [ ] Record unresolved risks, owners, and due dates in an issue tracker.
