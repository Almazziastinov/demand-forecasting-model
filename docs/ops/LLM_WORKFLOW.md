# LLM Workflow

Last updated: 2026-06-28

These rules are for Codex, Claude, and other code agents working in this repo.

## Start Every Session Here

Before changing code or production state:

1. Read `docs/ops/CURRENT_STATE.md`.
2. Read `docs/ops/SERVICES.md`.
3. Read only the relevant runbook sections in `docs/ops/RUNBOOK.md`.
4. Run `git status --short --branch`.
5. If the task touches production, verify live state before acting.

## Treat Handoffs As History

Files in `handoffs/` are useful session logs. They can explain how the project
got here, but they are not the current source of truth. If a handoff conflicts
with `docs/ops/CURRENT_STATE.md`, verify live state and update ops docs.

## Production Rules

- The VM is the only forecast writer.
- VibeCode/Blackhole is read-only API/UI.
- Do not enable or run forecast timers on Blackhole.
- Do not print secrets.
- Do not overwrite active forecast runs unless the intended run id is verified.

## Scope Control

Work in the smallest relevant area:

- Production operations: `pipelines/forecast_publish/`, `scripts/`, `deploy/`,
  `docs/ops/`
- Embedded app: `apps/forecast_embedded/`
- Legacy local demo: `web/`
- Legacy baseline ML: `src/`, `run_pipeline.py`
- Historical context: `handoffs/`

Do not refactor across these areas unless the user explicitly asks for it or the
change is required to fix the issue safely.

## After Production Changes

Update docs in the same change:

- `CURRENT_STATE.md` for state changes.
- `SERVICES.md` for ownership/service changes.
- `RUNBOOK.md` for reusable commands.
- `DECISIONS.md` for durable architecture decisions.

If no code changed but production state changed, still update `CURRENT_STATE.md`
or create a handoff note if the user wants a session log.

## Verification

For production work, a final answer should state:

- what was changed;
- what was verified;
- which service is now the source of truth;
- any remaining risk.

For code-only work, use the test commands from `AGENTS.md` and mention any tests
that could not be run.
