create table if not exists forecast_runs_embedded (
  run_id String,
  model_version String,
  profile_version String,
  source_kind String,
  horizon_start Date,
  horizon_end Date,
  generated_at DateTime64(3),
  status LowCardinality(String),
  notes Nullable(String),
  is_bias_adjusted Bool
)
engine = MergeTree
order by (run_id, generated_at);

create table if not exists bakery_forecast_day_embedded (
  run_id String,
  forecast_date Date,
  bakery_id Int64,
  bakery_name String,
  city Nullable(String),
  forecast_base Nullable(Float64),
  forecast_final Float64
)
engine = MergeTree
partition by toYYYYMM(forecast_date)
order by (run_id, forecast_date, bakery_id);

create table if not exists sku_forecast_day_embedded (
  run_id String,
  forecast_date Date,
  bakery_id Int64,
  product_id Int64,
  product_name Nullable(String),
  category_name Nullable(String),
  forecast_qty Float64
)
engine = MergeTree
partition by toYYYYMM(forecast_date)
order by (run_id, forecast_date, bakery_id, product_id);

create table if not exists sku_forecast_hour_embedded (
  run_id String,
  forecast_date Date,
  bakery_id Int64,
  product_id Int64,
  hour Int16,
  forecast_qty Float64
)
engine = MergeTree
partition by toYYYYMM(forecast_date)
order by (run_id, forecast_date, bakery_id, product_id, hour);

create table if not exists sku_hour_share_profile_smoothed_embedded (
  bakery_id Int64,
  bakery_name String,
  product_id Int64,
  product_name Nullable(String),
  category_name Nullable(String),
  dow Int16,
  hour Int16,
  n_days Int32,
  mean_sku_share_in_hour Float64,
  mean_sku_hour_sales Float64,
  median_sku_share_in_hour Float64,
  std_sku_share_in_hour Float64,
  mean_sku_share_in_hour_norm Float64
)
engine = MergeTree
order by (bakery_id, dow, hour, product_id);

create table if not exists sku_hour_uplift_multiplier_embedded (
  bakery_id Int64,
  dow Int16,
  hour Int16,
  sku_uplift_multiplier Float64,
  profile_version String,
  generated_at DateTime64(3)
)
engine = MergeTree
order by (profile_version, bakery_id, dow, hour);
