/*
Export from stg_check_lines (replaces fct_check_lines which was dropped).
Data range available: 2025-06-01 onwards.

Key differences vs fct_check_lines:
  - Added filter: is_deleted = 'Нет'  (soft-delete flag)
  - operation_type / is_deleted are new columns, not selected (not needed)

Output columns match the normalized English schema expected by
scripts/export_clickhouse_checks.py and build_bakery_daily_dataset.py:
  check_datetime, check_date, cash_event_type, quantity, price,
  line_amount, freshness, bakery_id, bakery_name, city,
  product_id, product_name, category_name
*/
SELECT
    fcl.check_datetime     AS check_datetime,
    fcl.check_date         AS check_date,
    fcl.cash_event_type    AS cash_event_type,
    fcl.quantity           AS quantity,
    fcl.price              AS price,
    fcl.line_amount        AS line_amount,
    fcl.freshness          AS freshness,
    fcl.bakery_id          AS bakery_id,
    db.bakery_name         AS bakery_name,
    db.city                AS city,
    fcl.product_id         AS product_id,
    dp.product_name        AS product_name,
    dp.category_name       AS category_name
FROM Svezhar.stg_check_lines AS fcl
ANY LEFT JOIN Svezhar.dim_bakeries AS db
    ON db.bakery_id = fcl.bakery_id
ANY LEFT JOIN Svezhar.dim_products AS dp
    ON dp.product_id = fcl.product_id
WHERE fcl.cash_event_type = 'Продажа'
  AND fcl.is_deleted = 'Нет'
  AND fcl.check_date BETWEEN toDate('{date_from}') AND toDate('{date_to}')
{limit_clause}
