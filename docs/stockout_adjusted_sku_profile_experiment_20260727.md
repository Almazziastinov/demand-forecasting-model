# Stockout-adjusted SKU/profile experiment

Date: 2026-07-27

Status: offline end-to-end diagnostic; production unchanged

## Question

The conservative stockout-adjusted bakery target improved bakery-day bias.
This experiment tests whether that added total reaches affected stockout SKUs
through the current normalized SKU-hour profile, without assuming explicit
donor-to-recipient transfers.

Two non-overlapping 14-day holdouts follow cutoffs 2026-06-21 and 2026-07-05.
All profiles and adjustments use history available by the cutoff. The bakery
predictions come from the preceding bakery-target experiment. Actual hourly
bakery shape is used as an oracle control so the result isolates daily total
and SKU-profile effects from hour-level forecast error.

## Variants

- observed total and observed-sales profile: baseline;
- conservative bakery total with the observed-sales profile;
- observed bakery total with the conservative reconstructed-demand profile;
- conservative total and conservative profile together;
- conservative total and a guarded profile that retains baseline exact routing
  and fallback.

The primary evaluation target is reconstructed demand over the entire holdout:
clean SKU-days use observed sales and stockout SKU-days use observed sales plus
reconstruction. Conservative and full reconstruction variants are both
reported. Stockout-only and clean-only scopes remain diagnostics.

All variants are aligned to the same union of SKU-days before scoring. Missing
variant predictions are treated as zero. The resulting supports are 24,049 and
21,582 SKU-days in the two windows.

## Result

The reconstructed profile contains a positive distribution signal, but the
full conservative bakery-total uplift is not yet stable at SKU level.

### Bakery total alone

The conservative bakery model adds 833.1 units per holdout on average, but the
unchanged profile sends only 46.6 units to stockout SKU-days. Pooled across
both windows, this closes only 3.2% of the reconstructed stockout gap. About
94% of the added total goes to clean SKU-days.

### Reconstructed profile alone

With the bakery total held approximately fixed, the reconstructed profile
moves 251.2 units per window toward stockout SKU-days. This is an implicit
reallocation: almost the same amount is removed from clean SKU-days.

Against reconstructed demand over all SKU-days, MAE improves in 2/2 windows
with a mean delta of -0.0196 units per SKU-day. Aggregate bias cannot change
because the bakery total is fixed. MAE on historically adjusted bakery/SKU
pairs and on the stockout-only subset worsens in 2/2, so the gain is broad
distribution improvement rather than precise delivery to the current
stockout rows.

### Conservative total and full profile

The end-to-end variant delivers 299.3 units per window to stockout SKU-days.
This closes 20.6% of the pooled reconstructed gap. Delivery is unstable:
9.1% after the 2026-06-21 cutoff and 34.9% after 2026-07-05.

Against reconstructed demand over all SKU-days, absolute aggregate bias
improves in 2/2 windows by 747.7 units per window on average. SKU-day MAE
improves only 1/2 and is effectively neutral on average (-0.0001). The
stockout-only and adjusted-pair scopes still worsen in 2/2. Roughly 65% of net
added volume goes to clean SKU-days.

### Guarded profile

The guarded end-to-end variant closes 16.8% of the pooled stockout gap and
sends about 71% of net added volume to clean SKU-days. It improves overall
reconstructed-demand absolute bias in 2/2 windows. SKU-day MAE improves only
1/2, although its mean delta is favourable at -0.0195 because the second
window improves strongly. Stockout-SKU MAE and adjusted-pair clean-SKU MAE
both worsen in 2/2.

The largest clean-day recipients include Kystybyi P, chicken and beef
triangles, sausage pastry, cabbage bekken, Makovka, and sausage-under-coat.
This confirms that added bakery volume is broadly dispersed through mature
profile shares rather than targeted at the censored SKU demand.

## Decision

The earlier observed-sales-primary interpretation was too negative. When the
entire holdout is evaluated on reconstructed demand, the adjusted profile alone
improves SKU-day MAE consistently and should remain an offline candidate.

Do not promote the full end-to-end variant yet: the complete bakery uplift
improves aggregate reconstructed-demand bias but only wins SKU-day MAE in one
window. The next experiment should keep the reconstructed profile and sweep a
partial bakery-total uplift, for example 0%, 25%, 50%, 75%, and 100% of the
conservative bakery correction. Selection must use all-SKU reconstructed
demand on equal support, with stockout-only and adjusted-pair scopes as
guardrails.

## Artifacts

- Experiment: `scripts/experiment_stockout_adjusted_sku_profiles.py`
- Tests: `tests/test_experiment_stockout_adjusted_sku_profiles.py`
- Reports: `reports/stockout_adjusted_sku_profile_experiment/`
