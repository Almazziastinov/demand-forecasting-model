# Daily and predictive SKU allocation backtest — corrected 2026-08-25

## Correction notice

The original end-to-end and concentration summaries included forecast-only
bakery-days with zero aggregate target. Those rows are now excluded from model
quality metrics and reported as a separate data-quality population.

The corrected universe requires `date × bakery` aggregate `demand_mid > 0`.
This leaves 3,252 evaluable bakery-days across 184 bakeries and excludes 492
forecast-only bakery-days across 37 bakeries. Excluded incumbent forecast mass
is 256,384.55 units.

## Architecture under test

Every method receives exactly the same incumbent bakery/category total. The
experiment evaluates only the SKU mix:

1. stored incumbent SKU allocation;
2. causal daily historical profile;
3. forecast-conditioned predictive allocation with daily fallback.

No method changes the bakery/category volume supplied by the bakery-level ML
model. Hourly scheduling is downstream and must conserve each SKU-day total.

## Corrected results

| Method | End-to-end SKU WAPE | Allocation WAPE | Bias |
|---|---:|---:|---:|
| Incumbent allocation | 39.5920% | 32.0778% | +11.8964% |
| Daily profile | 41.1765% | 33.6694% | +11.8964% |
| Predictive allocation | **38.6535%** | **31.0759%** | +11.8964% |

The predictive method improves SKU WAPE by 0.9385 pp end-to-end and 1.0018 pp
in the equal-total allocation diagnostic. The daily historical profile is
worse than the incumbent and remains fallback-only.

The common +11.8964% bias is inherited from category totals in this historical
`base_bakery_raw_uplift_sku` scenario. It is not evidence that the current
`base_norm_recent` bakery model has the same bias; the corrected August audit
shows current bakery-level observed-sales bias of -0.4697%.

## Concentration reinterpretation

On the corrected observable universe, none of the three methods has a bakery-day
with one SKU at or above 20%:

| Method | p95 top SKU share | Maximum | >=20% | >=30% | >=40% |
|---|---:|---:|---:|---:|---:|
| Incumbent | 13.5156% | 16.9047% | 0 | 0 | 0 |
| Daily profile | 13.3208% | 15.5196% | 0 | 0 | 0 |
| Predictive | 13.4630% | 15.6875% | 0 | 0 | 0 |

Therefore the original claim that this historical interval independently
proved the >=30% concentration incident is withdrawn. The August production
incident remains a separate, directly observed case and must be evaluated on
its own observable bakery-day universe.

## Revised interpretation

- Bakery-level ML remains the source of total volume.
- Predictive allocation remains the best challenger in this experiment, but
  its gain is modest and requires a fresh blocked validation.
- Direct Mean7 quantity replacement is no longer supported by this research.
- The original segment table and simple-average control are withdrawn until
  regenerated after the observability filter.
- SKU 1071 and the August hourly-profile amplification require a dedicated
  current-scenario analysis; this historical concentration table is not valid
  evidence for them.

## Next validation

1. Use current `base_norm_recent` bakery/category totals.
2. Restrict quality metrics to observable bakery-days and publish excluded
   forecast mass separately.
3. Compare incumbent, predictive, daily and guarded blended SKU allocation.
4. Report full SKU WAPE, equal-total allocation WAPE, missing actual SKU,
   forecast-only SKU, SKU 1071 and top-share risk.
5. Preserve SKU-day totals before applying any hourly schedule.

## Corrected artifacts

- `reports/daily_sku_allocation_backtest_20260824/corrected_active_universe_summary.json`
- `scripts/recalculate_active_bakery_universe.py`

Production writes: none.
