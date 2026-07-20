# Stockout Direction Offline Results — 2026-07-20

## Boundary

All ClickHouse access was read only. No production services, tables, runs,
profiles, timers, or environment variables were changed. All candidate
transformations and forecasts were written only to local `reports/` files.

## Architecture tested

The work deliberately separates two mechanisms:

1. **Allocation:** bakery volume is preserved and only SKU shares may change.
2. **Demand loss:** censored SKU demand is reconstructed before profiles,
   rolling features, and bakery targets are built; bakery volume increases by
   exactly the reconstructed SKU demand.

Cases that cannot be identified confidently are not transformed.

## 1. Case classification

The 397 confirmed model-underforecast stockouts were compared with a trailing,
same-weekday bakery counterfactual built only from prior dates. The primary
classification produced:

| Case type | Cases | Confirmed shortfall |
| --- | ---: | ---: |
| Allocation | 240 | 961.2 |
| Demand loss | 57 | 213.5 |
| Uncertain | 100 | 334.6 |

Threshold sensitivity was evaluated across bakery-normal ratios 0.90–1.00,
demand-loss ratios 0.80–0.90, and substitution thresholds 0.25–0.75. The
high-precision intersection used downstream contains:

- 231 robust allocation cases;
- 26 robust demand-loss cases;
- 140 uncertain cases.

The strict demand-loss set requires bakery volume below 80% of its expected
same-weekday level and a statistically material negative gap. It should be
interpreted as a strong co-occurrence of stockout and low bakery volume, not
proof that one SKU caused the entire bakery gap.

## 2. Demand-adjusted preprocessing

Only the 26 robust demand-loss cases were eligible. Expected post-stockout
hourly demand was estimated from prior non-stockout weekdays using both direct
SKU-hour demand and the SKU's share of bakery-hour traffic. Guardrails:

- at least three reference days;
- bakery remains active after the SKU's last sale;
- per-hour estimate capped at 2× the observed positive-hour rate;
- total case uplift capped at 75% of observed SKU sales and 20 units;
- original sales retained in separate columns;
- no correction for allocation or uncertain cases.

Result:

- 25/26 cases received an adjustment;
- 140.3 units reconstructed;
- median 3.75 units per case, maximum 15;
- 96.2% reference coverage;
- 17/825 bakery-days affected;
- total target change: 0.014% of 984,766 observed units;
- maximum mean profile-share change: 0.179 percentage points.

Daily SKU history, bakery targets, lag/rolling variants, and adjusted share
profiles were generated locally. The adjustment is safe but currently too
sparse to materially retrain the bakery model. A longer shadow history is
needed.

## 3. Reconstruction backtest

Synthetic stockouts were created by hiding known non-stockout sales after the
last visible hour. The existing bakery-share method recovered approximately
75–81% of hidden demand. A guarded hybrid recovered 82–87% for higher-volume
SKU ending two or three hours early, but was worse for small SKU and four-hour
gaps. Therefore no universal hybrid was selected; the shadow configuration
keeps conservative caps and records the segment for later calibration.

WMAPE was not used as the decision metric. Recovery bias, imputed volume, and
false-uplift exposure were used instead.

## 4. Dynamic allocation model

A leakage-free LightGBM model predicted a capped log correction to the current
SKU share. Training used only prior confirmed non-stockout rows. Features
included bakery, SKU, category, weekday, baseline share, lagged local ratios,
network product prior, local category prior, dispersion, and history length.
Every scenario was renormalized to preserve the original bakery-day total.

Best candidate: `model_log_ratio_strength_0.25`.

| Metric | Baseline | Dynamic candidate |
| --- | ---: | ---: |
| Stockout shortfall | 1,530.3 | 1,553.1 |
| Normal-day MAE | 4.681 | 4.696 |
| Cases removed | — | 2 |
| New underforecast cases | — | 5 |
| Maximum bakery-total change | — | ~0 |

The candidate failed both stockout and normal-day gates and is rejected from
shadow. Stronger model and direct pair-calibration variants were worse.

This does not contradict the 33% improvement in the current-profile replay:
that profile includes data through the evaluated period and is diagnostic,
whereas the model experiment is strictly walk-forward.

## 5. Combined replay

| Scenario | Shortfall | Fixed cases | Improved | Worsened |
| --- | ---: | ---: | ---: | ---: |
| Historical baseline | 1,509.3 | 0 | 0 | 0 |
| Demand preprocessing only | 1,453.6 | 23 | 25 | 0 |
| Current profile diagnostic | 1,005.9 | 133 | 282 | 112 |
| Current profile + demand diagnostic | 973.2 | 143 | 286 | 108 |
| Walk-forward dynamic allocation | 1,525.4 | 2 | 39 | 113 |
| Dynamic allocation + demand | 1,469.5 | 25 | 62 | 106 |

Only demand preprocessing improves cases without creating new underforecasts.
The current-profile scenarios remain diagnostic-only because of look-ahead.

## Decision

### Accepted into local shadow

- robust demand-loss classification;
- conservative demand reconstruction;
- adjusted SKU-day and bakery-day targets;
- adjusted lag/rolling columns and share profiles;
- full case/hour audit.

### Rejected from shadow

- current walk-forward dynamic allocation model;
- direct bakery×SKU ratio calibration.

### Deferred

- mixed-case restoration: no stable mixed class was identified;
- production activation: explicitly out of scope;
- bakery model retraining: adjustment volume is currently too small;
- cold-start allocation: insufficient evaluation history.

## Shadow operation

Run locally:

```bash
python scripts/run_stockout_direction_shadow.py --env-file .env
```

The runner refreshes classification, demand adjustments, and combined replay,
then writes `reports/stockout_direction_shadow/manifest.json`. It has no
ClickHouse write path.

Promotion gates in `config/stockout_direction_shadow.json` require at least 21
shadow days, no normal-day bias regression, no new underforecast cases, and a
manual review before any production proposal.
