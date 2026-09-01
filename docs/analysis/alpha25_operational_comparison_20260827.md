# Alpha 0.25 operational comparison — 2026-08-27

> Historical stress test only. The 1,200-unit daily capacity screen used here
> is not supported by measured bakery-capacity data and must not be used for
> candidate selection or production clipping. See the corrected unconstrained
> stateful simulation in `alpha25_stateful_operations_20260831.md`.

## Decision under test

The selected research candidate is:

`Original Direct -> predictive uplift -> Core-SKU protection -> soft normalization alpha=0.25 -> adaptive floor -> causal tail cap`.

No production data, configuration, forecast snapshots, services, or timers were changed.

## Operational replay

All candidates were passed through the same daily execution rules:

1. subtract the positive previous-day closing stock;
2. round the remaining production need up to the SKU multiple;
3. add previous-day stock back to obtain total available to sell;
4. apply the current approximate daily core-production capacity screen (1,200 units per bakery-day).

The comparison covers the three evaluation folds (`2026-07-27`, `2026-08-10`,
`2026-08-17`). Values below use the calibrated lost-demand scenario.

| Candidate | Total to sell | Surplus | Underbake | Total imbalance |
| --- | ---: | ---: | ---: | ---: |
| Current | 2,555,193 | 611,324 | 772,996 | 1,384,319 |
| Direct P50 | 2,729,429 | 634,192 | 621,628 | 1,255,820 |
| Previous final | 2,748,551 | 640,218 | 608,532 | **1,248,750** |
| Alpha 0.25 + tail cap | 2,858,255 | 711,547 | **570,157** | 1,281,704 |

Relative to Current, Alpha 0.25 reduces underbake by 202,839 units and total
imbalance by 102,615 units, at the cost of 100,224 additional surplus units.

Relative to Previous final, Alpha 0.25 reduces underbake by 38,375 units, but
adds 71,329 surplus units and increases total imbalance by 32,954 units. This is
consistent with the stated business priority (minimize underbake first), but it
is not the minimum-imbalance candidate after operational constraints.

## Important interpretation

Rounding the raw forecast directly was invalid. The production publisher rounds
the net need after subtracting yesterday's stock. The corrected replay uses that
ordering.

This remains a historical one-day replay using the actual causal opening stock
known on each forecast date. A full automation/economic simulation must carry
the candidate's own simulated stock from one day to the next and then apply
rounding and shared capacity. Therefore the earlier raw-plan economics must not
be presented as final operational economics.

## Artifacts

- `scripts/select_alpha25_causal_tail_cap.py`
- `scripts/apply_alpha25_operational_constraints.py`
- `scripts/compare_alpha25_operational_candidates.py`
- `reports/alpha25_operational_candidate_comparison_20260827/summary.csv`
- `reports/alpha25_operational_candidate_comparison_20260827/rows.parquet`
