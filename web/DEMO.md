# Artifact Demo

Separate demo app for the hybrid research stage.

## Run

```powershell
python web/demo_app.py
```

## What it shows

- SKU filtering by `best_r2` bucket
- Monthly charts from saved benchmark artifacts
- Comparison between a selected model and the global `66_cluster_features` baseline
- Same-bakery SKU comparison table

## Data source

- `src/experiments_v2/full_benchmark/best_by_sku.csv`
- `src/experiments_v2/full_benchmark/artifacts/*/predictions.csv`
