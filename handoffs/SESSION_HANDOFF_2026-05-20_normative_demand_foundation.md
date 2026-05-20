# Session Handoff - 2026-05-20 - Normative Demand Foundation

## Scope

This session established the technical and research foundation for the
`normative_demand` workstream.

The main outcome is that the project now has:

- a Russian research plan for normative demand;
- a new SKU-day research base builder on top of the deduplicated ClickHouse
  sales snapshot;
- raw export templates for new operational sources;
- a first segmentation experiment (`N1`) wired to the new research layer;
- a panel version of the SKU-day dataset for proper zero-day / intermittent
  analysis.

## Conceptual Direction Agreed

We explicitly reframed the task away from "recover true demand exactly" and
toward building a **planning-oriented normative demand signal**.

Key distinction:

- `observed_sales` = what was actually sold
- `attainable_demand` = conservative estimate of what could have been sold
- `normative_demand` = structured operational target for planning, production,
  and procurement

Important business interpretation:

- the goal is not to reproduce all historical chaos;
- the goal is to create a more systematic target that still respects bakery and
  SKU behavior;
- this target should later become the base layer for planning, with holidays
  and events applied as separate corrections.

## New / Changed Files

### Research plan

- `src/experiments_v2/NORMATIVE_DEMAND_RESEARCH_PLAN.md`
  - Russian research blueprint
  - definitions, metrics, segmentation logic, and experiment roadmap

### ClickHouse export helpers

- `scripts/export_clickhouse_generic.py`
- `scripts/clickhouse_export_production_release.sql`
- `scripts/clickhouse_export_moves.sql`
- `scripts/clickhouse_export_dim_kkt.sql`

Purpose:

- reuse the existing ClickHouse connection approach from the sales exporter;
- export production release, moves, and `dim_kkt` into local raw CSVs;
- keep raw extraction separate from downstream cleaning and audit.

### New research base builder

- `src/experiments_v2/build_sku_daily_research_base.py`
- `tests/test_build_sku_daily_research_base.py`

Builder responsibilities:

1. aggregate sales from `sales_hrs_all_clickhouse.csv`
2. apply the existing strict sales dedup logic
3. aggregate release data
4. aggregate move data
5. build partner map from `dim_kkt`
6. merge all layers into a SKU-day research dataset
7. create a panel version with zero-filled days inside active windows
8. write audit artifacts for unresolved conflicts

## Dedup / Cleaning Rules Implemented

### Sales

Already-existing strict dedup logic reused:

- `check_datetime`
- `bakery_id`
- `product_id`
- `quantity`
- `price`
- `line_amount`
- `cash_event_type`

### Production release

Important agreed rule:

- `_updated_at` is **not** treated as a reliable version selector
- `_UUID` must be included in exact duplicate handling

Implemented approach:

- exact duplicate removal on:
  - `_UUID`
  - `release_id`
  - `line_id`
  - `release_date`
  - `bakery_id`
  - `product_id`
  - `quantity`
  - `baker_name`
- conflict audit on:
  - same `release_id + line_id`
  - but differing remaining fields

### Moves

- exact duplicate removal on:
  - `move_id`
  - `move_date`
  - `product_id`
  - `sender_id`
  - `receiver_id`
  - `quantity`
- conflict audit on repeated `move_id` with differing payload

### Partner map (`dim_kkt`)

- exact duplicates removed
- bakery-to-organization conflicts are preserved in audit
- main map chooses the most frequent organization pair per bakery
- `organization_conflict_flag` is preserved in the merged dataset

## New Dataset Outputs

The builder now writes:

- `data/processed/sku_daily_research_base.csv`
- `data/processed/sku_daily_research_panel.csv`
- `data/processed/sku_daily_research_base_summary.json`

Audit outputs:

- `reports/sku_daily_research_base_audit/release_conflicts.csv`
- `reports/sku_daily_research_base_audit/moves_conflicts.csv`
- `reports/sku_daily_research_base_audit/partner_conflicts.csv`

### Difference between base and panel

`sku_daily_research_base.csv`

- observed-centered
- only days with observed sales rows
- useful for explainability and operational joins

`sku_daily_research_panel.csv`

- panelized inside active windows per `bakery x SKU`
- missing days are added with zero sales
- necessary for correct `zero_share`, `intermittent` detection, and proper
  time-series segmentation

## N1 Segmentation Experiment

### Files

- `src/experiments_v2/75_normative_demand_map/run.py`
- `tests/test_normative_demand_map.py`

### What N1 now does

It builds a `predictability_and_structure_map` for each `bakery x SKU` pair,
using:

- observed sales behavior
- weekly structure
- lag-7 predictability proxy
- release coverage and release correlation
- move behavior
- bakery dependence
- partner metadata
- row quality / conflict signals

Artifacts written under:

- `src/experiments_v2/75_normative_demand_map/`

### First result on observed-centered base

Segment counts before panel:

- `noisy`: `45,206`
- `amplitude_unstable`: `7,271`
- `trend_dominated`: `5,143`
- `bakery_driven`: `521`
- `stable`: `212`
- `high_censoring`: `10`

Interpretation:

- useful first cut
- but clearly missing intermittent behavior because only observed sales days
  were present

### Second result on full panel

Command used:

```powershell
.venv\Scripts\python.exe -m src.experiments_v2.75_normative_demand_map.run --daily-path data/processed/sku_daily_research_panel.csv
```

Segment counts after panel:

- `noisy`: `30,197`
- `intermittent`: `13,180`
- `amplitude_unstable`: `12,005`
- `trend_dominated`: `2,280`
- `bakery_driven`: `433`
- `stable`: `257`
- `high_censoring`: `11`

Interpretation:

- panelization worked as intended
- a large chunk of previous "noise" was actually intermittent behavior
- `amplitude_unstable` became a major segment
- `high_censoring` remains underdeveloped and needs new rules

## Current Interpretation of Segments

- `stable`
  - candidate for `Normative V1`
- `amplitude_unstable`
  - main candidate for `Normative V2`
- `bakery_driven`
  - candidate for bakery-anchored normative construction
- `trend_dominated`
  - candidate for a trend-first normative variant
- `intermittent`
  - separate sparse / fallback branch
- `noisy`
  - still too broad, needs more splitting later
- `high_censoring`
  - current definition is too weak and should be redesigned

## Next Recommended Step

Implement the first normative construction experiment:

- `src/experiments_v2/76_normative_v1_v2/run.py`

Scope:

1. read `sku_daily_research_panel.csv`
2. read the N1 segment map
3. keep only:
   - `stable`
   - `amplitude_unstable`
4. build:
   - `Normative V1 = level + fixed weekday structure`
   - `Normative V2 = V1 + adaptive weekly amplitude`
5. compare their structural behavior and business plausibility

## Tests Run

Verified locally:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_build_sku_daily_research_base.py tests\test_normative_demand_map.py -v
```

Result:

- `4 passed`

## Important Git Notes

Do **not** mix these changes with:

- local notebook changes:
  - `notebooks/bakery_day_backtest.ipynb`
  - `notebooks/hourly_sales_day.ipynb`
- raw exported CSV files in `data/raw/`

Those should remain outside this commit unless explicitly requested.
