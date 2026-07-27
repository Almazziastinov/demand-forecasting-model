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

Stockout SKU-days are scored against observed sales as a lower bound, the
conservative reconstructed target, and the full reconstructed point. Clean
SKU-days use observed sales.

## Result

No SKU/profile variant is suitable for promotion.

### Bakery total alone

The conservative bakery model adds 833.1 units per holdout on average, but the
unchanged profile sends only 46.6 units to stockout SKU-days. Pooled across
both windows, this closes only 3.2% of the reconstructed stockout gap. About
94% of the added total goes to clean SKU-days.

### Reconstructed profile alone

With the bakery total held approximately fixed, the reconstructed profile
moves 251.2 units per window toward stockout SKU-days. This is an implicit
reallocation: almost the same amount is removed from clean SKU-days.

Clean-SKU MAE improves in 2/2 windows at the aggregate all-SKU scope, but MAE
on historically adjusted bakery/SKU pairs worsens in 2/2. Stockout-SKU MAE
against the conservative target also worsens in 2/2. The profile therefore
improves aggregate bias while assigning volume to the wrong individual rows.

### Conservative total and full profile

The end-to-end variant delivers 299.3 units per window to stockout SKU-days.
This closes 20.6% of the pooled reconstructed gap. Delivery is unstable:
9.1% after the 2026-06-21 cutoff and 34.9% after 2026-07-05.

Aggregate stockout bias improves in 2/2 windows, but:

- stockout-SKU MAE worsens in 2/2;
- clean-SKU MAE worsens in 2/2;
- adjusted-pair clean-SKU MAE worsens in 2/2;
- roughly 65% of net added volume still goes to clean SKU-days.

### Guarded profile

The guarded end-to-end variant closes 16.8% of the pooled stockout gap and
sends about 71% of net added volume to clean SKU-days. It improves overall
clean-SKU MAE in 2/2 windows, but stockout-SKU MAE and adjusted-pair clean-SKU
MAE both worsen in 2/2. Guarding routing reduces damage but does not solve
delivery.

The largest clean-day recipients include Kystybyi P, chicken and beef
triangles, sausage pastry, cabbage bekken, Makovka, and sausage-under-coat.
This confirms that added bakery volume is broadly dispersed through mature
profile shares rather than targeted at the censored SKU demand.

## Decision

Do not feed reconstructed demand directly into the current normalized profile
as a production candidate. Normalization necessarily converts part of the
correction into implicit transfers, while an increased bakery total is spread
mostly across existing mature shares.

The next useful experiment is an independent SKU-demand model trained on the
stockout-adjusted target. SKU forecasts should be produced without a fixed-sum
share constraint; the bakery total can then be obtained by summing or
reconciling those SKU forecasts. This preserves the intended semantics:
restoring one SKU's censored demand does not require subtracting volume from
another SKU.

## Artifacts

- Experiment: `scripts/experiment_stockout_adjusted_sku_profiles.py`
- Tests: `tests/test_experiment_stockout_adjusted_sku_profiles.py`
- Reports: `reports/stockout_adjusted_sku_profile_experiment/`
