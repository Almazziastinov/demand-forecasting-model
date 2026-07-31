# Pilot Expansion Candidate Set — 2026-07-31

The locally configured pilot set is expanded from 10 to 22 bakeries:

`1, 20, 21, 22, 28, 39, 41, 56, 57, 66, 67, 69, 80, 89, 107, 125, 149, 155, 160, 221, 222, 257`.

New bakeries:

| Partner | Bakery IDs |
| --- | --- |
| Захарова Ирина | 1, 57, 67, 125, 149 |
| Нигматова Алина / Тормасова Ксения | 66, 155 |
| Зайнутдинов Раиль | 160 |
| Макарова Татьяна | 39, 41, 56, 69 |

Bakery 16 remains excluded. Матвеева Владлена has no current bakery mapping
and is not included. Тормасова Ксения shares the two Нигматова bakeries and
does not require a separate database identity for the shared pilot workbook.

The same set is used by:

- `scripts/publish_pilot_forecast.py` — daily Bitrix24 pilot workbook;
- `scripts/build_pilots_evening_uplift.py` — pilot uplift-profile builder;
- `scripts/build_stockout_correction.py` — stockout-correction builder;
- `scripts/run_milp_baking_plan.py` and `scripts/export_milp_baking_plan.py`.

The daily publisher applies the same processing to every ID in the set:
forecast selection, cold-start correction, mature-SKU systematic correction,
previous-day stock subtraction, kratnost rounding, and Excel rendering.

The daily publisher was deployed to Blackhole on 2026-07-31. Its pre-deploy
backup is
`/opt/scripts/publish_pilot_forecast.py.backup_20260731_144758_pilot22`.
A remote `--dry-run` for 2026-08-01 produced 1,222 SKU rows across all 22
bakeries and did not send a Bitrix24 message. The publisher timer remains
enabled and active for 03:00 UTC / 06:00 MSK.

The production forecast writer, active forecast run, ClickHouse tables, and
active uplift/stockout profile versions were not changed by this rollout.
