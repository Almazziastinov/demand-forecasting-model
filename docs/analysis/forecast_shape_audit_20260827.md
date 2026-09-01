# Forecast shape audit (2026-08-27)

Research-only audit of current, predictive same-volume, P50+Predictive, and
P50+Predictive+simple-floor forecasts on 3,554 bakery-days / 188 bakeries / 20
saved dates in the controlled SKU scope.

## Known concentration failure

Predictive allocation materially fixes the amplitude of single-SKU
concentration. Bakery-days with a top SKU share >=20% fall from 652 current to
57 predictive and 7 with floor; >=30% falls from 191 to 2 and 0. Mean total
variation distance to reconstructed demand shares improves from 0.2647 to
0.2358 and 0.2266. The p99 top share falls from 40.0% current to 20.7%
predictive and 17.9% floor (reconstructed demand p99 is 22.2%).

For bakery 29 on 2026-08-23, SKU 1071 falls from 441 units / 32.86% current to
301 / 21.96% predictive at same volume, 355 / 21.96% with P50, and 355 /
18.67% with floor. Reconstructed demand is 161 / 8.25%. Thus the extreme
amplitude is reduced but the case is not solved.

## Remaining systematic problems

SKU 1071 remains the top forecast SKU on 2,796 current bakery-days, 2,860
predictive, and 2,838 floor, versus only 1,965 reconstructed-demand days.
Predictive therefore reduces its share but makes no progress on its excessive
ranking frequency.

Predictive produces exact zero on 1,323 rows with positive reconstructed
demand (4,688 units across 1,010 bakery-days), versus 7 rows / 37 units in the
current system. All have zero predictive raw output. Of these, 1,034 have a
floor history count with mean 2.37 and maximum 8; 289 have no qualifying floor
history. Simple floor repairs only five rows and leaves 1,318 zeros / 4,664
demand units. This is the same thin/missing-SKU coverage class that motivated
hour-profile filling and is not solved by the candidate.

Floor also changes total volume broadly: median +17.73% versus P50, p90
+26.35%, p99 +37.49%, max +57.72%. P50 itself has a median +10.52% versus
current but ranges from -28.26% to +133.66% by bakery-day. These tails require
explicit bakery-day guards before operational use.

## Conclusion

Predictive allocation solves most of the catastrophic concentration amplitude
but not the excessive dominance of SKU 1071, thin-SKU disappearance, or all
bakery-day volume tails. The next model iteration should add causal coverage
for zero/thin predictive outputs and a dominance/ranking guard before further
economic optimization.

Outputs: `reports/forecast_shape_audit_20260827/`. Production unchanged.
