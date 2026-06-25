# Session Handoff - 2026-06-24 - VM Migration + Pie Cap Fix

## Summary

Two things done this session:

1. **Fixed a bug** where products in "пироги сладкие/сытные" category were missing from the
   forecast for bakeries with no tier1 profile entry (thin bakeries).
2. **Migrated production to a new VM** — old VM (89.108.76.196) died from OOM.

---

## Bug Fix: Pie Category Cap Zeroing Out Thin Bakeries

### Root Cause

In `_apply_category_upward_cap` (apply_bakery_profiles_clickhouse.py), the upward cap formula:

```python
cap = base * max_multiplier  # = 0 * 1.0 = 0 for thin bakeries
cap = min(cap, recent_dow_avg_qty)  # = min(0, positive) = 0
capped = min(corrected, cap)  # = min(positive, 0) = 0
```

For protected-category products where a bakery has no tier1 profile entry
(`base_daily_forecast = 0`), the cap collapsed to 0, zeroing out any
correction computed from recent sales. This meant the product never appeared
in `new_daily` and got no forecast.

Affected: ~126 bakeries for product 11473 (сметанник маковый, category "пироги сладкие").
Pattern applies to all SKUs in "пироги сытные|пироги сладкие" for thin bakeries.

### Fix

In the DOW-aware branch of `_apply_category_upward_cap`, treat `base=0` rows
the same way the `elif` (non-DOW) branch already did — use `recent_avg_cap`
directly instead of `min(0, recent_avg_cap)`:

```python
# Before:
cap = pd.Series(np.where(has_recent_cap, np.minimum(cap, recent_avg_cap), cap), ...)

# After:
cap = pd.Series(
    np.where(
        has_recent_cap & base.gt(0),
        np.minimum(cap, recent_avg_cap),
        np.where(has_recent_cap, recent_avg_cap, cap),
    ), ...
)
```

### Verification

Debug output before fix:
```
[DEBUG 11473] bakeries_in_targets=182 base_zero=126 corrected_pos=56 new_daily_bakeries=0
```

Debug output after fix:
```
[DEBUG 11473] bakeries_in_targets=182 base_zero=126 corrected_pos=176 new_daily_bakeries=156
```

Product 11473 now gets a forecast in 176 bakeries (was 56).

### Commits

- `1ecc549` — fix: cap formula zeros out profile-absent SKUs in protected category
- `0e6f275` — chore: remove debug print for product 11473 investigation

---

## VM Migration: Old → New

### Old VM (dead)
- Host: `89.108.76.196`
- Died from OOM: multiple Python inference processes launched simultaneously,
  exhausted 2GB RAM + 4GB swap, kernel OOM-killed everything.

### New VM
- Host: `201.51.7.24` (msk-1-vm-tpez)
- User: `root`, password in `.codex/prod_vm.env`
- 16GB RAM, 154GB disk, 4GB swap (configured this session)
- Path: `/opt/demand-forecasting-model`
- Service user: `forecast` (git commands via `sudo -u forecast git -C /opt/...`)

### What Was Migrated

| Item | How |
|---|---|
| Repo | `git clone` from GitHub |
| `.venv` | Created fresh, `pip install -r requirements.txt` |
| `.env` | Written manually (credentials from old VM via web console) |
| `models/*.joblib, *.json` | Copied old→new via paramiko SFTP (2×21MB) |
| `data/processed/bakery_daily_sales*.csv` | Copied old→new via paramiko SFTP |
| `data/processed/bakery_hour_profile.csv` | Copied old→new via paramiko SFTP |
| `data/processed/bakery_weather_features.csv` | Copied old→new via paramiko SFTP |
| systemd service + timer | Copied from old VM, deployed to new |

### Key .env Settings on New VM

```
FORECAST_RECENT_CORRECTION_MODE=runner_city_prior_soft_weekpart
FORECAST_REFRESH_DATASETS=0   # dim_bakeries table broken in CH, keep 0
```

### Systemd Timer

```
forecast-production.timer — OnCalendar=*-*-* 03:30:00 UTC
```

Enabled and active. Next run: 2026-06-25 03:30 UTC.

### SSH Connection Note

`paramiko.SSHClient.connect()` fails on this server (key exchange error).
Use `paramiko.Transport` directly:

```python
t = paramiko.Transport(("201.51.7.24", 22))
t.connect(username="root", password="...")
```

---

## State After Session

- Production inference runs on new VM, timer active
- Fix deployed and verified: сметанник маковый now in forecast for 176/182 bakeries
- `FORECAST_REFRESH_DATASETS=0` — dataset refresh disabled (known CH issue with dim_bakeries)
- Old VM still exists but is OOM-crashed; can be accessed via hosting provider web console
