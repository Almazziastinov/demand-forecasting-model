# Two-day carryover economics and automation assessment (2026-08-26)

A chronological FIFO simulation was run on the 20-date controlled SKU scope.
Yesterday's unsold fresh quantity is sold first on the next calendar day and
expires after that day. Forecast gaps start from the observed opening-stock
proxy and are not joined across missing dates. Actual transfers are held fixed
for every strategy.

| Strategy | Production | Served demand | Remaining lost | Expired | Service level |
|---|---:|---:|---:|---:|---:|
| Actual-state simulation | 2,583,131 | 2,488,435 | 1,085,858 | 65,070 | 69.62% |
| P50 + Predictive | 3,043,379 | 2,679,480 | 894,813 | 203,614 | 74.97% |
| P50 + Predictive + floor | 3,576,546 | 2,963,941 | 610,352 | 349,523 | 82.92% |

Relative economics set sale price to 1.0 and disposal cost to 0.05 per expired
unit. Production cost is varied as a fraction of sale price.

| Production cost | P50 profit delta vs actual | Floor profit delta vs actual |
|---:|---:|---:|
| 0.20 | +92,068 (+4.68%) | +262,600 (+13.34%) |
| 0.35 | +23,031 (+1.46%) | +113,588 (+7.18%) |
| 0.50 | -46,007 (-3.85%) | -35,424 (-2.97%) |
| 0.65 | -115,044 (-14.27%) | -184,437 (-22.88%) |
| 0.80 | -184,081 (-43.97%) | -333,449 (-79.64%) |

Break-even production cost versus actual is 0.400 for P50 and 0.464 for
floor. Floor remains more profitable than P50 while production cost is below
approximately 0.520 of sale price.

This is not a ruble business case. It assumes deterministic reconstructed
demand, identical unit price, identical transfers, no capacity/batch/labor
constraints, and a two-day shelf life for every SKU. The actual-state
simulation is affected by the known opening-stock reconciliation gap, so its
served quantity is a modelled baseline rather than accounting truth.

Both candidates are technically automatable because every input is causal and
machine-computable. Removing humans from plan creation still requires hard
guards for assortment, bakery/SKU capacity, batch multiples, opening-stock
freshness, missing data, and maximum daily volume changes. P50 + Predictive is
the safer default; floor is economically attractive only when production cost
and waste economics fall below the measured break-even region.

Artifacts:

- `scripts/simulate_two_day_economics.py`
- `reports/two_day_economics_20260826/`

Production was unchanged.
