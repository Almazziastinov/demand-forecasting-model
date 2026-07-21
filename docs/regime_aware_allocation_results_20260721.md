# Regime-Aware SKU Allocation Results — 2026-07-21

## Boundary

The experiment is offline and read only. It reads historical forecast snapshots
and sales from ClickHouse and writes local reports only. No production table,
run, profile, service, timer, environment variable, or deployment was changed.

## Root cause in the previous allocation experiment

The previous dynamic allocator normalized `baseline_share` inside only the 53
screened SKU, while `observed_share` used total bakery sales across all SKU.
Those shares had different denominators. It also preserved the forecast total
of the screened subset rather than the complete historical bakery-day SKU
forecast. This made the LightGBM residual target internally inconsistent and
is a material explanation for its failed walk-forward result.

The replacement experiment reconstructs the complete forecast universe for
the dominant historical run of every bakery-day:

- 539 bakery-days;
- 97,334 complete-universe SKU-day rows;
- 7,145 screened rows matched to a forecast;
- 92.61% labelled forecast coverage;
- 565 screened rows had zero or missing forecast;
- 47 of the zero/missing rows were clear stockout rows.

The last group is a zero-allocation/assortment problem and cannot be repaired
by multiplicative share redistribution.

## Leakage-safe target and constraints

For each bakery×SKU×date, all features use earlier dates only:

- median non-stockout log share residual over the prior 42 days;
- recent-versus-older residual shift for regime detection;
- residual dispersion and direction consistency;
- prior 14-day stockout rate;
- prior 28-day sales quantiles for donor protection.

The selected allocator is intentionally asymmetric:

1. Only a reliable positive residual may receive volume.
2. Donors must belong to the screened bakery-product domain.
3. A donor keeps at least its trailing sales p90 plus 0.5 units.
4. At most 0.25% of the complete bakery-day SKU forecast may move.
5. A recipient uplift is capped at 20%.
6. The complete bakery-day SKU total is preserved exactly.

This prevents unrelated products such as packaging from becoming donors. In
an intermediate version, `Пакет звезды` became the only newly underforecast
row because its historical p90 was zero; restricting the donor domain removed
that failure.

## Scenario results

### Symmetric residual correction — rejected

Even at 25% strength and a 1% movement budget, symmetric renormalization:

- worsened stockout shortfall by 53.14 units;
- created 31 new stockout underforecasts;
- created 25 new normal-day underforecasts;
- improved normal-day MAE only by removing existing overforecast.

The mechanism treated forecast headroom as freely transferable and repeated
the same structural problem as the earlier allocator.

### Positive-capacity regime correction — accepted into local shadow

Selected scenario:
`positive_capacity_regime_q90m05_strength_1.00_budget_0.0025`.

| Metric | Baseline | Shadow candidate | Delta |
| --- | ---: | ---: | ---: |
| Clear-stockout shortfall | 1,291.245 | 1,289.402 | −1.843 |
| Normal-day MAE | 4.832506 | 4.830447 | −0.002058 |
| Full-universe MAE | 2.017625 | 2.017305 | −0.000319 |
| Full-universe shortfall | — | — | −15.534 |
| Recurrent-pair shortfall | 510.865 | 510.093 | −0.772 |
| New stockout underforecasts | — | 0 | 0 |
| New normal underforecasts | — | 0 | 0 |
| New full-universe underforecasts | — | 0 | 0 |
| Maximum bakery-total delta | — | 0 | 0 |

Only 32.387 units were redistributed over 49 days. The candidate passes every
local shadow gate, but its effect is deliberately small and it fixes no whole
case by itself.

### Explicit stockout-risk overlay — rejected

The best risk scenario used prior stockout frequency and the upper quartile of
non-stockout residuals. It reduced stockout shortfall by 2.498 units and
created no new underforecast, but normal-day MAE regressed by 0.001167.
The top-5 recurrent segment improved by only 0.056 units. The risk overlay is
therefore diagnostic only and is not enabled in shadow.

## Top-5 and other problematic SKU

For the selected shadow candidate:

| Segment | Stockout rows | Shortfall reduction | Normal MAE delta |
| --- | ---: | ---: | ---: |
| Recurrent top-5 | 17 | 0.000 | 0.000 |
| Recurrent other SKU | 234 | 0.772 | +0.001225 |
| Other screened SKU | 998 | 1.072 | −0.002512 |

The four previously flagged top-5 pairs explain why average calibration is
not enough:

- bakery 257 × ЖарПицца Пикантная (10485) has zero forecast in every labelled
  row and belongs to the zero-allocation direction;
- bakery 222 × Кыстыбый П (10340) has a negative typical non-stockout residual;
- bakery 22 × Элеш с курицей (10667) has weak/short usable history and no
  stable positive residual;
- bakery 221 × Треугольник говядина безд (1076) receives only a negligible
  uplift.

Thus the recurrent top-5 stockouts are mostly episodic variance or zero
allocation, not a stable mean-share error. A generic average-share correction
cannot solve them safely. More day-specific exogenous information or a better
stockout-risk target is required.

## Combined replay on the 397 confirmed misses

| Scenario | Shortfall | Improved | Worsened | Fixed |
| --- | ---: | ---: | ---: | ---: |
| Historical baseline | 1,509.285 | 0 | 0 | 0 |
| Demand preprocessing | 1,453.585 | 25 | 0 | 23 |
| Regime-aware allocation | 1,507.629 | 8 | 0 | 0 |
| Regime-aware allocation + demand | 1,451.929 | 33 | 0 | 23 |

The mechanisms are additive and do not conflict. Demand reconstruction
provides most of the gain; allocation adds a small safe improvement.

## Decision

Accepted into local read-only shadow:

- conservative demand-loss preprocessing;
- regime-aware positive-capacity allocation using the p90+0.5 screened donor
  rule and 0.25% bakery movement budget.

Rejected:

- the old LightGBM daily-share allocator;
- symmetric residual renormalization;
- the current explicit stockout-risk overlay.

The new allocator is not a production candidate yet. It must accumulate at
least 21 prospective shadow days, retain zero new underforecasts, and show a
larger stable gain before any production proposal.

## Reproduction

Allocation experiment only:

```powershell
.venv\Scripts\python.exe scripts\experiment_regime_aware_sku_allocation.py --env-file .env
```

Complete read-only direction shadow:

```powershell
.venv\Scripts\python.exe scripts\run_stockout_direction_shadow.py --env-file .env
```
