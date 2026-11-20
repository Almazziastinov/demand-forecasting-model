# Contributing

## Before you start

1. Read [the developer onboarding guide](docs/DEVELOPER_ONBOARDING.md).
2. For any production-related change, read `docs/ops/CURRENT_STATE.md`,
   `docs/ops/SERVICES.md`, `docs/ops/RUNBOOK.md`, and `docs/ops/LLM_WORKFLOW.md`.
3. Run `git status --short --branch`. Do not mix someone else's uncommitted
   work into your branch or deployment.

## Development workflow

- Create a branch from `master`: `codex/<short-description>` or
  `<developer>/<short-description>`.
- Make one coherent change per pull request.
- Add or update tests whenever behaviour changes.
- Before opening a pull request, run tests relevant to your change. For
  example:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/test_<affected_area>.py -v
  .\.venv\Scripts\python.exe -m ruff check <changed_python_files> --select=E,F,W
  ```

- Use English conventional commit messages, for example:
  `fix: preserve net transfers in bakery totals`.
- At least one other developer reviews every production-affecting pull request.

### Current CI baseline

The repository-wide Ruff command configured in CI currently reports legacy
violations outside an ordinary change's scope. Do not silently weaken or bypass
CI. Until the lint debt is separated into an explicit baseline-remediation
task, a PR must at minimum pass Ruff on every Python file it changes and must
not add new violations. Restoring a green repository-wide CI is a formal
handover blocker tracked in `docs/HANDOVER_STATUS.md`.

## Production boundaries

- The forecast VM is the only writer of production forecast runs.
- Blackhole/VibeCode is a read-only UI/API consumer; never enable forecast
  writers there.
- Do not copy an entire local module to Blackhole unless it was reconciled with
  the deployed version. The app contains regional fixes that historically did
  not always exist in the local checkout.
- Never commit `.env`, `.codex/`, tokens, passwords, database exports, or
  personally identifying audit files.
- Use individual accounts. Production database access is read-only by default;
  write access and VM deployment access require a reviewed change.

## Pull-request checklist

- [ ] Scope and expected business effect are described.
- [ ] Tests and Ruff pass locally and in CI.
- [ ] No secrets or generated reports were added.
- [ ] `docs/ops/` was updated if production behaviour, service ownership, or
      recovery procedure changed.
- [ ] Deployment and rollback steps are stated for production changes.
