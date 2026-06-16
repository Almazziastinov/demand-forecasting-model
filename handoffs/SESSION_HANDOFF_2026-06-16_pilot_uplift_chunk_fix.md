# Handoff: pilot bakeries, Tukaya uplift root cause, chunk-safe smoothing

Date: 2026-06-16
Workspace: `C:\Users\dns\Desktop\Projects\demand-forecasting-model`

## Current dev/prod state

- Prod was restored earlier to active run `prod_uplifted_bakery_norm_uplift_sku_20260616_h14`.
- Dev active run is `dev_lead1_history_20260601_20260614_hour_guard_v4`.
- `v4` is lead-1 history copied to dev plus pre-06 bakery-hour guard.
- Prod was not changed during the Tukaya investigation.

## What was fixed in dev before stopping

### Pre-06 hourly spike guard

File changed:

- `src/experiments_v2/apply_bakery_profiles.py`

Added guard in `load_bakery_hour_profile()`:

- drop bakery-hour profile rows with `hour < 6` when `n_days < 8`
- renormalize remaining bakery-hour shares inside `bakery_id x dow`

Reason:

- `Гудованцева 27` had a Tuesday `05:00` profile share of `14.81%` from a single historical day (`n_days=1`).
- In dev `v3`, this created ~288-294 forecast units at `05:00`.
- New dev `v4` removed `pre06` forecasts for all pilot bakeries while preserving `sku_day == sum(sku_hour)`.

Verification already done:

- `Гудованцева 27`: `pre06=0` for all dates `2026-06-01..2026-06-14`
- pilot bakeries: no `pre06 >= 10` and no `pre06 >= 1%`
- `sku_day` vs `sku_hour`: max diff `0.0`, sum diff `0.0`

## Tukaya 62A investigation

Bakery:

- `Габдуллы Тукая 62А Казань`
- `bakery_id=222`

Initial observation:

- In active dev `v4`, bakery total is not normatively uplifted.
- `2026-06-01..2026-06-14`:
  - forecast total: `10976.97`
  - fact total: `10985.22`
  - bias: `-0.08%`
- `forecast_final` above `forecast_base`: only `+0.39%`

Comparison to other pilot bakeries in `data/processed/bakery_daily_sales_uplifted.csv`, last 61 days before June:

| bakery_id | bakery | total uplift rate |
|---:|---|---:|
| 20 | Мира 45 Дербышки Казань | 27.64% |
| 21 | Парковая 7 Казань | 26.88% |
| 22 | Сибирский Тракт 25 Казань | 0.13% |
| 28 | Гудованцева 27 Казань | 27.29% |
| 80 | Калинина 63 Казань | 30.53% |
| 89 | Парина 6 Казань | 27.12% |
| 107 | Четаева 46А Казань | 28.98% |
| 221 | Салиха Батыева 15 Казань | 29.14% |
| 222 | Габдуллы Тукая 62А Казань | 0.001% |
| 257 | Ярмарочная 12 Чебоксары | 25.62% |

## Root cause found

The issue is not that Tukaya lacks SKU uplift flags.

Measured from `sku_hour_share_profile_daily_smoothed.csv` before the interrupted rebuild:

| bakery_id | bakery | avg adj sum | avg norm sum | avg ratio |
|---:|---|---:|---:|---:|
| 20 | Мира 45 | 1.228 | 1.000 | 1.228 |
| 28 | Гудованцева 27 | 1.253 | 1.000 | 1.253 |
| 107 | Четаева 46А | 1.268 | 1.000 | 1.268 |
| 222 | Габдуллы Тукая 62А | 1.272 | 1.993 | 0.640 |
| 22 | Сибирский Тракт 25 | 1.284 | 1.981 | 0.652 |

For normal bakeries, `sku_share_in_hour_adj_norm` sums to `1.0` per `date x bakery x hour`.
For Tukaya and Sibirsky Trakt, it summed to almost `2.0`.

Mechanical cause:

- `src/experiments_v2/smooth_sku_hour_share_profile.py` normalized `sku_share_in_hour_adj_norm` inside each pandas chunk.
- If one `date x bakery x hour` group was split by chunk boundaries, each part normalized independently to `1.0`.
- Final CSV then had `norm_sum ~= 2.0`.
- Uplift multiplier is calculated as `raw_share_sum / norm_share_sum`, clipped lower at `1.0`.
- For Tukaya: `1.27 / 1.99 = 0.64 -> clip -> 1.0`, so bakery uplift disappears.

Audit report updated:

- `reports/dev_pilot_lead1_audit/tukaya_uplift_data_audit.md`

## Code change prepared

File changed:

- `src/experiments_v2/smooth_sku_hour_share_profile.py`

New logic:

1. First pass reads `sku_hour_share_profile_daily.csv`, joins profile means, computes:
   - `sku_share_in_hour_adj`
   - `sku_share_uplift_raw`
   - `sku_share_uplift_flag`
2. It writes temporary `sku_hour_share_profile_daily_smoothed.adjusted_tmp.csv`.
3. It aggregates denominators globally by `date x bakery_id x hour`.
4. Second pass reads the temp file and computes:
   - `sku_share_in_hour_adj_norm`
   - `sku_share_uplift_norm_delta`
   - `sku_hour_sales_adj`
5. Then rebuilds `sku_hour_share_profile_smoothed.csv`.

Verification done before stopping:

- `.venv\Scripts\ruff.exe check src\experiments_v2\smooth_sku_hour_share_profile.py --select=E,F,W --line-length 120`
- passed

## Important interruption state

User stopped the full rebuild because it is too expensive to run now.

The rebuild command had started:

```powershell
.venv\Scripts\python.exe src\experiments_v2\smooth_sku_hour_share_profile.py --profile-path data\processed\sku_hour_share_profile.csv --applied-path data\processed\sku_hour_share_profile_daily.csv --output-dir data\processed --chunk-size 1000000
```

It was interrupted and then the lingering Python processes from `21:03` were killed:

- PID `27920`
- PID `50368`

The partial generated files were removed to avoid accidental use:

- `data/processed/sku_hour_share_profile_daily_smoothed.csv`
- `data/processed/sku_hour_share_profile_daily_smoothed.adjusted_tmp.csv`

This means local `data/processed/sku_hour_share_profile_daily_smoothed.csv` is currently missing and must be regenerated before any pipeline step that depends on it.

Existing untouched files:

- `data/processed/sku_hour_share_profile_daily.csv`
- `data/processed/sku_hour_share_profile.csv`
- `data/processed/sku_hour_share_profile_smoothed.csv` still exists from the old run
- `data/processed/sku_hour_share_profile_smoothed_summary.json` still exists from the old run

## Next steps tomorrow

1. Run the fixed smoothing script when full IO rebuild is acceptable.
2. Validate global normalization:

```text
sum(sku_share_in_hour_adj_norm) by date x bakery_id x hour must be 1.0
```

Specifically check:

- `bakery_id=222` Tukaya: avg norm sum should become `1.0`, not `1.99`
- `bakery_id=22` Sibirsky Trakt: avg norm sum should become `1.0`, not `1.98`

3. Rebuild uplift multipliers:

- local path through `pipelines/forecast_publish/sku_hour_profile_store.py`, or
- ClickHouse path if doing deploy-oriented refresh

4. Rebuild `bakery_daily_sales_uplifted.csv`.
5. Confirm Tukaya/Sibirsky now receive bakery uplift comparable to other pilot bakeries.
6. Only after that create a new dev forecast run and repeat pilot audit.

## Git scope for this handoff

Commit should include:

- `src/experiments_v2/apply_bakery_profiles.py`
- `src/experiments_v2/smooth_sku_hour_share_profile.py`
- `reports/dev_pilot_lead1_audit/tukaya_uplift_data_audit.md` if not ignored, otherwise force-add
- this handoff file

Do not commit:

- partial/generated `data/processed/*`
- unrelated `.codex/`
- unrelated notebooks
- unrelated existing handoff `SESSION_HANDOFF_2026-06-16_git_actualization.md`
