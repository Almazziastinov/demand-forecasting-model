# Current base_norm_recent versus causal seven-day mean — corrected 2026-08-25

## Correction notice

The original 2026-08-24 conclusion was invalid. It interpreted forecast-only
bakery-days without sales rows as zero demand. This mixed model error with an
unresolved production-universe/data-availability problem and inflated the
reported bakery-level WAPE.

The corrected evaluation includes a bakery-day only when aggregated observed
sales are positive. Forecast-only bakery-days are reported separately and are
not assigned a synthetic zero target.

## Scope

- Eight valid production dates: 2026-08-11, 12, 13, 17, 18, 21, 22 and 23.
- One explicit `base_norm_recent` production run per date.
- 176 observable bakeries and 1,406 evaluable bakery-days.
- 304 excluded forecast-only bakery-days across 38 bakeries.
- Excluded incumbent forecast mass: 158,267.15 units.
- Targets: deduplicated observed sales and conservative strict demand.

## Corrected bakery-level results

| Target | Current base_norm_recent | Previous 7-day mean |
|---|---:|---:|
| Observed sales WAPE | **7.0755%** | 12.4591% |
| Observed sales bias | **-0.4697%** | +7.0149% |
| Strict-demand WAPE | **7.5902%** | 12.6066% |
| Strict-demand bias | **-3.3387%** | +7.0631% |

The current bakery-level ML forecast beats the direct seven-day mean. The
previous claim that Mean7 won on all eight dates is withdrawn. On observed
sales, ML wins seven of eight dates; the only narrow loss is 2026-08-18.

The independent 2026-07-17..2026-08-02 check is consistent with this result:
bakery-level production WAPE is 7.25% on observable bakery-days. There is no
evidence of a July-to-August collapse of the bakery model.

## Layer decomposition

Direct inspection of production snapshots shows:

- August `forecast_base` WAPE: 6.7632%, bias: -1.6032%.
- August `forecast_final` WAPE: 7.0972%, bias: -0.6528%.
- The recent correction improves aggregate bias but worsens WAPE by 0.3340 pp.
- SKU-day forecasts conserve the bakery `forecast_final` total to floating
  point precision.

The large SKU errors therefore belong to SKU allocation/mix, not to the
bakery-volume ML model or a failure to conserve total volume.

## Data-quality finding

All 38 excluded bakeries lack sales rows on every evaluated date while still
receiving forecasts. They must not be treated automatically as zero-demand
bakeries. Each requires classification as inactive/closed, missing ETL data,
or identifier mismatch. Until resolved, quality reports must publish this
population and its forecast mass separately.

## Revised decision

1. Retain bakery-level ML as the production volume model.
2. Withdraw Mean7 as a proposed primary replacement.
3. Audit the 38 forecast-only bakeries and add an explicit observability gate
   to evaluation reports.
4. Evaluate SKU allocation methods under the same conserved bakery/category
   totals.
5. Review the recent correction separately: it reduces bias but slightly
   worsens WAPE.

## Corrected artifacts

- `reports/base_norm_recent_vs_mean7_20260824/corrected_active_universe_summary.json`
- `reports/base_norm_recent_vs_mean7_20260824/corrected_active_metrics_by_date.csv`
- `scripts/recalculate_active_bakery_universe.py`

Production writes: none.
