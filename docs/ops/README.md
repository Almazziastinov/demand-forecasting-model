# Ops Documentation

This directory is the current operational source of truth for the project.
It is intentionally short and practical: a new maintainer or LLM agent should
be able to understand the live system from these files before reading session
handoffs.

Start here:

1. `CURRENT_STATE.md` - what is true right now.
2. `SERVICES.md` - live services, ownership, and allowed writers.
3. `RUNBOOK.md` - verification and incident commands.
4. `LLM_WORKFLOW.md` - rules for code agents.
5. `DECISIONS.md` - durable architecture decisions.
6. `DATA_CONTRACTS.md` - production tables and artifact contracts.

Handoffs in `handoffs/` are historical logs. They are useful context, but they
must not override this directory without a fresh verification.
