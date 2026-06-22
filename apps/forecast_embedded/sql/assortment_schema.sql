create table if not exists assortment_city_products (
  city String,
  product_id String,
  product_name String,
  category_name String,
  is_required UInt8,
  is_top UInt8,
  top_rank Nullable(UInt16),
  source LowCardinality(String),
  source_priority UInt8,
  source_file String,
  source_scope String,
  valid_from Date,
  valid_to Nullable(Date),
  is_active UInt8,
  loaded_at DateTime64(3),
  comment String
)
engine = ReplacingMergeTree(loaded_at)
order by (city, product_id, valid_from, source);

create table if not exists assortment_source_audit (
  source LowCardinality(String),
  source_file String,
  source_scope String,
  city String,
  raw_product_name String,
  raw_category_name String,
  matched_product_id Nullable(String),
  matched_product_name Nullable(String),
  matched_category_name Nullable(String),
  match_status LowCardinality(String),
  issue String,
  is_required UInt8,
  is_top UInt8,
  top_rank Nullable(UInt16),
  loaded_at DateTime64(3)
)
engine = MergeTree
order by (source, city, raw_product_name, loaded_at);

create table if not exists bakeable_products (
  city LowCardinality(String),
  product_id String,
  product_name String,
  category_name String,
  is_bakeable UInt8,
  source LowCardinality(String),
  source_file String,
  valid_from Date,
  valid_to Nullable(Date),
  is_active UInt8,
  loaded_at DateTime64(3),
  comment String
)
engine = ReplacingMergeTree(loaded_at)
order by (city, product_id, valid_from);
