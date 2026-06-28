# Decisions

This file records durable project decisions. It is not a session log.

## 2026-06-28 - VM Is The Only Production Forecast Writer

Decision:

- Production forecast generation and ClickHouse forecast publishing run only on
  the VM `201.51.7.24`.
- VibeCode/Blackhole serves only the embedded read-only API/UI.
- Forecast timers on VibeCode/Blackhole must stay disabled.

Context:

- The project previously had forecast generation installed on Blackhole at
  `/opt/forecast_job`.
- On 2026-06-28, a Blackhole `forecast-production.timer` was found overwriting
  the active ClickHouse run with stale run
  `prod_uplifted_bakery_norm_uplift_sku_20260601_h14`.
- The stale timer was disabled and the fresh VM-generated run
  `prod_uplifted_bakery_norm_uplift_sku_20260623_h14` was re-activated.

Implication:

- Any future production writer migration must update `CURRENT_STATE.md`,
  `SERVICES.md`, this decision log, and the runbook before or during rollout.

## 2026-06-28 - Ops Docs Are The Current State Layer

Decision:

- `docs/ops/` is the operational source of truth for live state.
- `handoffs/` remains historical context only.

Implication:

- New LLM sessions should begin with `docs/ops/CURRENT_STATE.md`.
- Any production-state change must update `docs/ops/`.
