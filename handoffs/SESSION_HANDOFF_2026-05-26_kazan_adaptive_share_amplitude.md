# Session Handoff 2026-05-26 - Kazan Adaptive Share and Amplitude

## Scope

This session continued the `Kazan temporal normative` slice and focused on two
questions:

- how to make weekly share adaptive to recent SKU behavior;
- how to make SKU-level weekly amplitude variants visibly affect the temporal
  series.

The work stayed on the existing sample:

- city: `Kazan`
- category: `Sitnaya bakery`
- bakeries: top `30` by completeness
- target level: `bakery x sku x day`

## What Was Implemented

### 1. Three SKU-level amplitude variants were added

File:

- `src/experiments_v2/build_kazan_temporal_normative_sku_day.py`

The temporal builder now keeps the legacy weekly-total formulation and also
builds three SKU-level adaptive-amplitude variants:

- `weekly_cv`
- `flat_deviation`
- `reference_profile`

All variants are written into the same output dataset with separate columns:

- `temporal_normative_legacy_qty`
- `temporal_normative_weekly_cv_qty`
- `temporal_normative_flat_deviation_qty`
- `temporal_normative_reference_profile_qty`

Default exported `temporal_normative_qty` currently points to:

- `reference_profile`

### 2. Adaptive weekday share was introduced

The weekday profile is no longer just a single long-run profile.

Current construction:

- `long_weekday_share`
  - median weekday profile from the full local `bakery x sku` history
- `short_weekday_share`
  - median weekday profile from the last `28` days
- `weekday_share_normative`
  - blended share:
    - `0.5 * long + 0.5 * short`

Support columns now included in the temporal dataset:

- `long_weekday_share`
- `short_weekday_share`
- `share_delta_short_vs_long`
- `weekday_share_normative`
- `weekday_factor_normative`

Interpretation:

- short profile does not replace the long profile;
- it corrects the long profile toward the recent month.

### 3. Amplitude effect was made stronger

The first implementation of SKU-level amplitude variants produced almost no
visible deviation from the baseline.

To test stronger behavior, the builder now:

- uses wider `z-score -> multiplier` ranges;
- converts raw amplitude multipliers into stronger
  `*_effective_strength` values before applying them to weekday factors.

New diagnostic columns:

- `weekly_cv_amplitude_multiplier`
- `flat_deviation_amplitude_multiplier`
- `reference_profile_amplitude_multiplier`
- `weekly_cv_effective_strength`
- `flat_deviation_effective_strength`
- `reference_profile_effective_strength`

## Notebook Work

New notebook:

- `notebooks/kazan_temporal_amplitude_variants_review.ipynb`

Purpose:

- compare `legacy` vs the three SKU-level amplitude variants;
- inspect adaptive weekday share behavior;
- review pairs with the largest `short vs long` share shifts;
- inspect amplitude multipliers and effective strengths.

Notebook updates in the latest saved version:

- reads `kazan_temporal_normative_sku_day_summary.json`
- shows:
  - `short_share_days`
  - `short_share_weight`
- sorts pair index by:
  - `max_share_shift`
- default example pair:
  - `bakery_id = 142`
  - `product_id = 1071`

## Test / Build Status

Validated:

- `tests/test_build_kazan_temporal_normative_sku_day.py`

Command used:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_build_kazan_temporal_normative_sku_day.py -v
```

Result:

- `1 passed`

The Kazan temporal dataset was also rebuilt successfully via:

```powershell
.venv\Scripts\python.exe src\experiments_v2\build_kazan_temporal_normative_sku_day.py
```

## Current Metrics

From `data/processed/kazan_temporal_normative_sku_day_summary.json` after the
latest rebuild:

- `short_share_days`: `28`
- `short_share_weight`: `0.5`

Variant mean absolute gap summary:

- `legacy`: `8.161302`
- `weekly_cv`: `8.889444`
- `flat_deviation`: `9.356082`
- `reference_profile`: `9.66329`

## Current Interpretation

### About adaptive share

- moving from `14` to `28` days reduced some overreaction from the short-share
  correction;
- however, the main quality issue is not only the short-share window itself.

### About amplitude

- earlier versions were too weak and visually close to baseline;
- the stronger version now clearly changes the temporal series;
- but the current stronger formulation overshoots and degrades aggregate gap
  metrics versus `legacy`.

In other words:

- weak amplitude produced too little visible effect;
- strong amplitude produced visible effect, but in the wrong direction.

## Practical Stopping Point

The team decided to stop here for this slice and keep the current state
recorded.

Most relevant current takeaways:

1. adaptive share logic is now implemented and inspectable;
2. stronger amplitude logic is implemented and inspectable;
3. the current stronger amplitude setup is too aggressive;
4. `legacy` remains the best aggregate variant on the current Kazan slice.

## Most Likely Next Step If Continued

If this work resumes, the next move should not be to strengthen the signal even
further.

Recommended direction:

- keep `short_share_days = 28`
- keep share blending logic available for inspection
- reduce or gate the amplitude effect instead of increasing it further

The most likely useful follow-up would be:

- confidence-weighted short-share correction;
- amplitude gating by signal stability / weekly confidence;
- local review of pairs with the largest share shifts before any new global
  tuning.
