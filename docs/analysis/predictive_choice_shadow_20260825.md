# Prospective predictive-choice shadow — 2026-08-25

## Scope

The predictive-choice allocator was applied locally to the production plan
`prod_base_bakery_norm_recent_20260825_h14` for 2026-08-25. The model was
trained only on information available through 2026-08-23. The input contains
40,398 SKU rows, 214 bakeries and 628 products.

This is a prospective shadow: the 2026-08-25 fact was not available when the
allocation was generated, so this report checks allocation invariants and
risk indicators, not forecast accuracy.

## Conservation and concentration

| Metric | Incumbent | Predictive choice |
|---|---:|---:|
| Network total | 199,963.23 | 199,963.23 |
| p95 largest SKU share | 28.57% | 17.69% |
| Maximum largest SKU share | 39.02% | 38.60% |
| Bakery-days with largest SKU >=20% | 69 | 6 |
| Bakery-days with largest SKU >=30% | 6 | 1 |
| Bakery-days with largest SKU >=40% | 0 | 0 |

The network total is unchanged. Every bakery/category total is conserved;
the maximum absolute reconciliation difference is
`1.14e-13` units. Predictions contain no null or negative quantities.

## Coverage risks

- 7,882 rows have fewer than seven historical sales days; incumbent mass on
  those rows is 11,798.75 units.
- 5,862 rows have no prior recorded sales.
- 38 bakeries have no observed sales in the recent source window; their
  incumbent forecast mass is 21,002.88 units.

These rows are retained in the shadow output and explicitly flagged. The
prospective fact comparison must report established and cold-start segments
separately; the 38 unobservable bakeries must not be silently scored as zero
demand.

## Decision

The shadow passes the structural gate and materially reduces the concentration
incident without changing bakery/category volumes. It does not yet authorize a
production rollout. After the 2026-08-25 fact closes, compare incumbent and
predictive WAPE on the observable universe, with separate cold-start and
unobservable coverage diagnostics.

Artifacts:

- `scripts/run_predictive_choice_shadow.py`
- `reports/predictive_choice_shadow_20260825/shadow_predictions.parquet`
- `reports/predictive_choice_shadow_20260825/forecast_override_20260825.csv`
- `reports/predictive_choice_shadow_20260825/summary.json`

Production writes: none.
