# Developer onboarding

This guide is the entry point for a developer joining the demand-forecasting
project. It supplements, but does not replace, `docs/ops/` for production
operations.

## 1. Project map

| Area | Location | Purpose |
| --- | --- | --- |
| Production forecast writer | `pipelines/forecast_publish/` | Builds and activates forecast runs on the forecast VM. |
| Served model | `weighted_weekday_production.py` | Weighted same-weekday forecast on normalised/restored demand. |
| Production UI/API | `apps/forecast_embedded/` | Read-only Bitrix embedded application backed by ClickHouse. |
| Baking-plan export | `apps/baking_plan/` | Calculates the downloadable bakery plan from forecast and observed flows. |
| Research | `src/experiments_v2/`, `src/model_tournament/` | Model experiments and fixed-origin comparisons. |
| Legacy baseline | `src/`, `run_pipeline.py`, `web/` | Historical/reference tooling, not the production source of truth. |

The authoritative operational documents are:

1. `docs/ops/CURRENT_STATE.md` — verified incidents and current state.
2. `docs/ops/SERVICES.md` — service topology and writer ownership.
3. `docs/ops/RUNBOOK.md` — checks, recovery and rollback.
4. `docs/ops/DECISIONS.md` — durable architectural choices.
5. `docs/ops/DATA_CONTRACTS.md` — production data contracts.

Historical files in `handoffs/` are context only, not a source of truth.

## 2. Local setup

Prerequisites: Git, Python 3.12, and a personal ClickHouse development account.

```powershell
git clone https://github.com/Almazziastinov/demand-forecasting-model.git
cd demand-forecasting-model
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
Copy-Item .env.example .env.dev
```

Fill only `.env.dev`. Keep `FORECAST_TABLE_SUFFIX=_dev`; it prevents local
experiments from writing to serving tables. Do not create or share a copied
production `.env` file.

Useful local commands:

```powershell
ruff check src/ apps/ pipelines/ scripts/ tests/ --select=E,F,W
pytest tests/ -v
$env:ENV_FILE = '.env.dev'
uvicorn apps.forecast_embedded.app.main:app --host 127.0.0.1 --port 3001 --reload
```

For the app-specific workflow see `apps/forecast_embedded/README.md` and
`docs/dev_environment.md`.

## 3. Environments and access

| Resource | Normal developer access | Elevated/operator access |
| --- | --- | --- |
| GitHub repository | Write via feature branches and PRs | Repository administrator manages teams, rules and secrets. |
| ClickHouse | Personal read-only credentials or `_dev` tables | Time-limited write permissions for reviewed migrations/backfills. |
| Forecast VM | No shared root account | Named SSH account, reviewed deployment rights. |
| Blackhole/VibeCode | Source/review access | Named deployment credential; application service only. |
| Bitrix | Test or business-view access as needed | App owner manages portal and production settings. |

Secrets belong in the approved password manager or platform secret storage,
never in Git, tickets, chat messages, reports, screenshots, or `.env.example`.

## 4. Production model and safety rules

The forecast VM is the only allowed writer of serving forecast runs. The
embedded app is a consumer of active ClickHouse runs. Before a production
change, verify live state rather than relying on an old run identifier in a
document.

Production commands, service names, timers, and rollback instructions are in
`docs/ops/RUNBOOK.md`. New developers must first perform a dry-run or read-only
health check alongside an existing operator.

In particular:

- do not activate a run merely because it exists;
- do not start forecast timers on Blackhole;
- preserve the distinction between forecast allocation and downstream stock,
  transfer, multiplicity, and baking-plan calculations;
- use the regional ID-offset rules documented in `DATA_CONTRACTS.md`;
- back up targeted ClickHouse rows before any mutation and verify afterwards.

## 5. First-week checklist

- [ ] Clone the repository and complete a local test run.
- [ ] Read the five operational documents above.
- [ ] Open a small documentation-only PR and receive a review.
- [ ] Observe a read-only production health check.
- [ ] Run an inference only against `_dev` tables.
- [ ] Review one app change and one forecasting-pipeline change with an owner.

## 6. Known handover risks

The repository can contain local work that has not yet been committed, and the
deployed Blackhole service has historically accumulated targeted regional
repairs. Before a broad deployment, reconcile the exact deployed files with
the reviewed Git revision. Never treat an arbitrary developer workstation as a
deployment source.

The live operational state must be freshly verified before the formal handover
meeting; `CURRENT_STATE.md` records the latest verified snapshot available in
the repository, not a guarantee about the state at the moment you read it.
