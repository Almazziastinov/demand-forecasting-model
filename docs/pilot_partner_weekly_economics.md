# Partner weekly economics (dev)

## Status

The partner-facing weekly economics block is implemented for the dev pilot
management application. It is not enabled in production and is not part of the
daily production report job yet.

## Business meaning

The block compares two production scenarios against the same reconstructed
demand:

1. actual production;
2. raw Direct forecast converted to net production need without SKU-multiple
   rounding.

The simulation uses a two-day FIFO shelf-life model. Yesterday's product is
sold first with a 30% discount; fresh product is sold at full price. Gross
profit is revenue minus production cost from the reviewed SKU economics map.
The displayed difference is an estimate of additional potential, not a causal
or guaranteed revenue claim.

## Guardrails

- Only rows whose `forecast_run_id` starts with
  `prod_direct_alpha_025_` are eligible. Historical plans from the retired
  model must not be presented as evidence for Direct alpha=.25.
- Rows without a published production plan, reconstructed demand, unit price,
  unit cost, or lost-demand eligibility are excluded.
- The UI displays the share of reconstructed demand included in the economic
  calculation.
- Forecast quality, bakery execution and data-quality problems remain separate
  metric families.
- Published multiple-adjusted execution remains an operational KPI and is not
  used in the economic scenario until that downstream mechanism is validated.

## Dev build

Precompute the economic rows after building the pilot management CSV report:

```powershell
.venv\Scripts\python.exe scripts\build_pilot_partner_economics.py
```

This writes ignored runtime files `economics_daily.csv`,
`economics_mapping.csv`, and `economics_metadata.json` into the selected report
directory. The embedded service uses the precomputed daily rows so weekly and
filtered pages do not replay the simulation on every request.

Run the local dev application with:

```powershell
.\scripts\dev_run_embedded_api.ps1 -EnvFile .env.dev.tunnel -Port 3001
```

## Before production

1. Rebuild the management report through dates containing published Direct
   alpha=.25 plans.
2. Verify transfer, opening-stock and write-off completeness for the economic
   subset.
3. Package the approved price/cost mapping as a versioned report input rather
   than relying on the local research directory.
4. Integrate the precompute step into the atomic daily report job.
5. Reconcile several bakery/SKU/week examples manually and obtain business
   approval for the wording and attribution rules.
