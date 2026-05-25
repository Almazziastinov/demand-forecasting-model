# Session Handoff 2026-05-25 - Kazan Temporal Normative

## Scope

This slice focuses on:

- city: `Kazan`
- category: `Sitnaya bakery`
- bakeries: top `30` by completeness
- target level: `bakery x sku x day`

The goal was to move from noisy level reconstruction toward a readable
temporal normative series with:

- stable weekly structure
- slow-moving level
- adaptive weekly amplitude

## What Was Built

Sample and support layers:

- `data/processed/kazan_bakery_daily_sample.csv`
- `data/processed/kazan_sitnaya_bakery_category_daily_sample.csv`
- `data/processed/kazan_sitnaya_daily_sample.csv`
- `data/processed/kazan_sitnaya_hourly_sample.csv`
- `data/processed/kazan_bakery_profile_map.csv`
- `data/processed/kazan_sitnaya_sku_profile_map.csv`
- `data/processed/kazan_bakery_clusters.csv`
- `data/processed/kazan_sitnaya_sku_clusters.csv`

Diagnostics and structural layers:

- `data/processed/kazan_anchor_suitability_map.csv`
- `data/processed/kazan_decomposition_path_scores.csv`
- `data/processed/kazan_reconstructed_sku_day.csv`

Temporal normative layer:

- `data/processed/kazan_temporal_normative_sku_day.csv`
- `data/processed/kazan_temporal_normative_sku_day_summary.json`

Visual review assets:

- `notebooks/kazan_reconstruction_review.ipynb`
- `notebooks/kazan_temporal_normative_review.ipynb`
- `src/analysis/plot_kazan_reconstruction.py`

## Structural Reading

The strongest default structural branch remains:

- `bakery_category -> sku_cluster -> sku`

A close alternative is:

- `bakery_total -> category -> sku_cluster -> sku`

`city_sku -> bakery` and `bakery_cluster_sku -> bakery` still matter for
smaller subsets, but they are not the default backbone on the current Kazan
sample.

## Temporal Normative Contract

The current temporal series is built on top of `reconstructed_sales_qty`.

Per `bakery x sku`:

1. weekly totals are computed from the structural backbone
2. weekly totals are smoothed with EWMA
3. a stable weekday profile is estimated from recent weeks
4. the final daily series is reconstructed as:

`temporal_normative_qty = week_total_smoothed * weekday_share_normative`

Adaptive amplitude is represented by:

`weekly_amplitude_factor = week_total_smoothed / long_run_week_mean`

This means:

- weekday shape is repeatable
- weekly scale can expand or contract
- short noisy day-level movement is suppressed

## Current Metrics

From `data/processed/kazan_temporal_normative_sku_day_summary.json`:

- rows: `231,695`
- bakeries: `30`
- sku: `29`
- mean observed sales: `38.1709`
- mean reconstructed sales: `38.1098`
- mean temporal normative sales: `37.3343`
- mean temporal absolute gap: `8.2006`
- mean temporal bias: `-0.8366`

Interpretation:

- the temporal series is more conservative than raw observed sales
- this is expected because temporal smoothing removes irregular local noise
- the main evaluation mode should be visual and operational, not only fit to
  observed sales

## Validation

Relevant Kazan pipeline tests were run and passed:

- `tests/test_build_kazan_sitnaya_sample.py`
- `tests/test_build_kazan_profile_maps.py`
- `tests/test_build_kazan_clusters.py`
- `tests/test_build_kazan_anchor_suitability.py`
- `tests/test_build_kazan_decomposition_levels.py`
- `tests/test_build_kazan_share_stability_maps.py`
- `tests/test_build_kazan_decomposition_path_scores.py`
- `tests/test_build_kazan_reconstructed_sku_day.py`
- `tests/test_build_kazan_temporal_normative_sku_day.py`
- `tests/test_plot_kazan_reconstruction.py`

## Next Step If Continued

If this workstream continues, the next iteration should focus on:

- blending local weekday shape with upper-level weekday shape
- stronger handling for sparse or special SKU
- explicit trend smoother separate from weekly amplitude
