# Direct alpha=.25 current shadow — 2026-08-31

One technical shadow run was executed for 2026-09-01 against active source run
`prod_base_bakery_norm_recent_20260831_h14`. The run queried ClickHouse
read-only and wrote local artifacts only.

The first attempt exposed a forecast-origin cutoff bug (history included the
run-generation date). It was rejected before use. The accepted run uses sales
through 2026-08-30 only.

## Accepted scope and guards

- 214 bakeries, 185 products, 12,282 rows;
- selected volume 189,506.67;
- Direct P50 volume 181,584.44;
- one causal tail-cap row;
- no NaN, negative or duplicate predictions;
- no active incumbent row mapped to a near-zero Direct prediction;
- 25 bakery-days without 56-day sales evidence use incumbent fallback;
- all top SKU shares >=30% belong to the fallback group.

## Known incidents

| Bakery / SKU | Current | Shadow |
| --- | ---: | ---: |
| Zorge 101 / Smetannik 108 | 0.043 | 27.593 |
| Zorge 101 / chicken triangle 1071 | 379.811 | 235.282 |
| Bakery 29 / chicken triangle 1071 | 466.815 | 262.935 |

The shadow passes the intended one-run technical gate. Production and dev
database state were not changed.

