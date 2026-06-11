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

create table if not exists forecast_day_context_embedded (
  run_id String,
  forecast_date Date,
  city String,
  temp_mean Nullable(Float64),
  precipitation Nullable(Float64),
  rain Nullable(Float64),
  snowfall Nullable(Float64),
  windspeed_max Nullable(Float64),
  is_bad_weather Bool,
  weather_cat_code Int16,
  holiday_name Nullable(String),
  is_holiday Bool,
  is_pre_holiday Bool,
  is_post_holiday Bool,
  event_window_type LowCardinality(String),
  current_event_cluster LowCardinality(String),
  prev_event_cluster LowCardinality(String),
  next_event_cluster LowCardinality(String),
  days_since_prev_event Int16,
  days_to_next_event Int16
)
engine = MergeTree
partition by toYYYYMM(forecast_date)
order by (run_id, forecast_date, city);

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

create table if not exists bakery_forecast_day_snapshots (
  source_run_id String,
  forecast_origin_date Date,
  lead_days Int16,
  forecast_date Date,
  generated_at DateTime64(3),
  bakery_id Int64,
  bakery_name String,
  city Nullable(String),
  forecast_base Nullable(Float64),
  forecast_final Float64
)
engine = ReplacingMergeTree(generated_at)
partition by toYYYYMM(forecast_date)
order by (forecast_date, lead_days, bakery_id);

create table if not exists sku_forecast_day_snapshots (
  source_run_id String,
  forecast_origin_date Date,
  lead_days Int16,
  forecast_date Date,
  generated_at DateTime64(3),
  bakery_id Int64,
  product_id Int64,
  product_name Nullable(String),
  category_name Nullable(String),
  forecast_qty Float64
)
engine = ReplacingMergeTree(generated_at)
partition by toYYYYMM(forecast_date)
order by (forecast_date, lead_days, bakery_id, product_id);

create table if not exists sku_forecast_hour_snapshots (
  source_run_id String,
  forecast_origin_date Date,
  lead_days Int16,
  forecast_date Date,
  generated_at DateTime64(3),
  bakery_id Int64,
  product_id Int64,
  hour Int16,
  forecast_qty Float64
)
engine = ReplacingMergeTree(generated_at)
partition by toYYYYMM(forecast_date)
order by (forecast_date, lead_days, bakery_id, product_id, hour);

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

create table if not exists bitrix_user_bakery_access_embedded (
  bitrix_portal_id String,
  bitrix_user_id String,
  bitrix_email Nullable(String),
  bitrix_user_name Nullable(String),
  bitrix_work_position Nullable(String),
  partner_name Nullable(String),
  bakery_id Int64,
  bakery_name Nullable(String),
  access_role LowCardinality(String),
  match_method LowCardinality(String),
  source LowCardinality(String),
  updated_at DateTime64(3)
)
engine = ReplacingMergeTree(updated_at)
order by (bitrix_portal_id, bitrix_user_id, bakery_id, source);

create table if not exists bakery_month_revenue_embedded (
  month_start Date,
  bakery_id Int64,
  bakery_name Nullable(String),
  revenue Float64,
  revenue_bucket LowCardinality(String),
  source LowCardinality(String),
  generated_at DateTime64(3)
)
engine = ReplacingMergeTree(generated_at)
order by (month_start, bakery_id, source);
