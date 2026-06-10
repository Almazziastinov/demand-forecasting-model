# Session Handoff - 2026-06-10 - City Prior SKU Rollout

## What was done

- Investigated the SKU allocation issues on the holdout / research data.
- Confirmed that the main runner underforecast cases are not explained by stockout / missed demand in the current data.
- Evaluated several allocation variants and selected:
  - `runner_city_prior_soft_weekpart`
- Added the selected city-prior guard to the production allocation path.
- Added tests for the new recent-correction mode.
- Pushed the code to `origin/master`.

## Current commit

```text
d33387c feat: add city-prior SKU allocation guard
```

## What changed in code

- `pipelines/forecast_publish/run_production_inference.py`
  - New default recent correction mode includes `runner_city_prior_soft_weekpart`.
- `src/experiments_v2/apply_bakery_profiles_clickhouse.py`
  - Added the city-prior guard logic for recent correction.
  - Added support for daily/weekpart recent shares in the ClickHouse allocation path.
- `tests/test_apply_bakery_profiles_clickhouse_recent.py`
  - Added regression coverage for the new mode.

## Important deployment note

- The user's production VM is not the VibeCode server.
- The VibeCode server is for the frontend only.
- Do not continue trying to deploy the forecast pipeline to VibeCode.
- The forecast pipeline must be updated only on the user's SSH-managed VM.

## What was verified locally

- `ruff` passed for the touched files.
- Allocation-layer tests passed.
- The branch is clean and pushed to `origin/master`.

## What remains to do on the SSH VM

1. Pull `master` on the SSH VM.
2. Update the VM-side forecast job to use the new code.
3. Run the production inference job on the SSH VM with:
   - `runner_city_prior_soft_weekpart`
4. Compare the candidate run with the currently active run.
5. Activate the candidate only if the comparison is acceptable.

## Caveats

- I started work on the VibeCode infra path, but that was not the correct target for the forecast VM. That work should be ignored for rollout purposes.
- The useful artifact for rollout is the git commit above plus the selected correction mode.

