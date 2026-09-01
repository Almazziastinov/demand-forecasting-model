# Bakery 29 / SKU 1071 history (2026-08-27)

Research-only daily audit for 2026-06-01..2026-08-23.

Observed SKU sales average 230.3 units across 84 days. Sunday is structurally
lower: mean 195.8, median 196, min 161, max 248. On 2026-08-23 sales are 161,
the period minimum and 17.8% below the Sunday mean, but comparable with 164 on
June 28 and 174 on August 9. The bakery produced 200 and sold 161 on August 23;
the relaxed detector does not classify the day as a stockout. Thus 161 is an
observed low-demand day rather than a censored sell-out under the available
daily data.

Across Sundays the SKU averages 26.53% of its savory-bakery category and
13.23% of all bakery sales. On August 23 the shares fall to 20.99% and 9.99%.
Predictive assigns 27.28% inside the category, close to the historical Sunday
mean. The incumbent assigns 40.03%. Predictive therefore repairs most of the
within-category allocation error.

The remaining absolute overforecast comes from the frozen category total.
On August 23 observed category sales are 767, while current/Predictive category
forecast is 1,101.6 (+43.6%); Predictive preserves that total by design. It
therefore yields 300.6 units for SKU 1071 even with a plausible within-category
share. P50 raises the category to 1,299.4 and SKU 1071 to 354.5. The candidate
cannot solve this case while allocation is constrained to incumbent category
totals.

The lost-demand label is also influential. 46 of 84 days are flagged as clear
stockouts, typically after higher sales (mean 238 vs 221 non-stockout). In the
August-17 fold, calibrated Sunday demand becomes 365, 298, and 410 units on
July 5, 12, and 19 versus observed 201, 220, and 248. This lifts same-weekday
references toward approximately 285-300 units. Those values may represent
real hidden demand, but are high enough that the SKU-level calibration should
be audited before using them as a hard floor.

Conclusion: the original hourly-profile concentration is real in the incumbent
share, and Predictive largely fixes it inside the category. The residual case
is primarily a category-total allocation problem plus potentially aggressive
lost-demand reconstruction, not a failure to learn the SKU's ordinary Sunday
share. The next challenger should allocate the bakery-day total directly over
all SKU rather than preserve incumbent category totals.

Outputs: `reports/bakery29_sku1071_history_20260827/`. Production unchanged.
