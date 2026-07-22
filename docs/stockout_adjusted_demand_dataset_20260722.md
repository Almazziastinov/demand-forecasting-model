# Stockout-adjusted demand dataset v1

Date: 2026-07-22

Status: offline data layer; production unchanged

## Purpose

This dataset implements the simplified direction: restore censored demand on
accepted stockout SKU-days without assuming that another SKU pulled volume
away. No donor, reallocation, or bakery-volume regime label is used.

The grain is `date x bakery_id x product_id`. Version 1 contains SKU-days with
positive observed sales from the hourly source. It does not yet create the
complete assortment grid for zero-sale SKU-days.

## Target contract

- `demand_lower_bound`: observed sales. It is exact on clean rows and only a
  lower bound on clear-stockout rows.
- `imputed_demand`: estimated demand after the last sale hour, based on earlier
  non-stockout days for the same bakery, SKU, weekday, and hour.
- `demand_point_estimate`: observed sales plus imputed demand.
- `demand_upper_guardrail`: observed sales plus the configured reconstruction
  cap.
- `target_source`: provenance of the row, including observed,
  reconstructed, and censored-without-estimate states.
- `reconstruction_confidence`: confidence in the reconstruction, not in the
  stockout detector. Five or more reference days is `high`, three or four is
  `medium`, and fewer than three is `insufficient`.
- `suggested_training_weight`: an explicit offline starting weight: 1.0 for
  observed rows, 0.8 for high-confidence reconstruction, 0.5 for medium, and
  0 for censored rows without an estimate. These weights have not yet been
  validated as a model policy.

The point estimate remains capped at the lower of 20 units and 75% of the
observed case volume, with the existing four-unit floor for low-volume cases.
Both raw imputation and cap information remain in the dataset for sensitivity
analysis.

## Materialized sample

The read-only build covers 2026-05-03 through 2026-07-19 for 11 pilot bakeries:

- 114,852 SKU-day rows and 388 products;
- all 1,296 accepted clear-stockout rows are represented;
- 1,260 rows receive a positive reconstruction;
- 33 have insufficient history and 3 have no positive post-cutoff estimate;
- high / medium / insufficient reconstruction counts are 868 / 395 / 33;
- observed demand is 1,018,682.0 units;
- reconstructed demand adds 8,305.8 units, or 0.815% overall;
- there are no duplicate keys, negative targets, or point estimates below
  observed sales.

The cap is binding for 661 stockout rows, or 51.0%. Therefore the 8,305.8-unit
point estimate is materially policy-dependent and must not be treated as
ground truth. The cap grid gives:

| Maximum uplift ratio | Absolute cap | Imputed units |
| ---: | ---: | ---: |
| 0.50 | 10 | 6,204.0 |
| 0.50 | 20 | 6,872.7 |
| 0.75 | 10 | 7,277.8 |
| 0.75 | 20 | 8,305.8 |
| 1.00 | 10 | 7,973.3 |
| 1.00 | 20 | 9,214.3 |

This sensitivity is the main data-stage warning. The next modelling stage
should compare lower-bound, weighted-point, and cap variants rather than train
one unconditional adjusted target.

## Outputs

- Builder: `scripts/build_stockout_adjusted_demand_dataset.py`
- Main dataset: `reports/stockout_adjusted_demand_dataset/sku_day_demand.csv`
- Reconstruction audit and hourly estimates in the same report directory
- Machine-readable field contract: `schema.json`
- Diagnostics by confidence, bakery, and product
- Cap sensitivity: `cap_sensitivity.csv`
