/*
Return the normalized English-column raw schema for the bakery-driven pipeline.

Output columns:
  - check_datetime
  - check_date
  - cash_event_type
  - quantity
  - price
  - line_amount
  - freshness
  - bakery_id
  - bakery_name
  - city
  - product_id
  - product_name
  - category_name

The exporter will substitute:
  - {date_from}
  - {date_to}
  - {limit_clause}
*/
SELECT
    fcl.check_datetime AS check_datetime,
    fcl.check_date AS check_date,
    fcl.cash_event_type AS cash_event_type,
    fcl.quantity AS quantity,
    fcl.price AS price,
    fcl.line_amount AS line_amount,
    fcl.freshness AS freshness,
    fcl.bakery_id AS bakery_id,
    db.bakery_name AS bakery_name,
    db.city AS city,
    fcl.product_id AS product_id,
    dp.product_name AS product_name,
    dp.category_name AS category_name
FROM Svezhar.fct_check_lines AS fcl
ANY LEFT JOIN Svezhar.dim_bakeries AS db
    ON db.bakery_id = fcl.bakery_id
ANY LEFT JOIN Svezhar.dim_products AS dp
    ON dp.product_id = fcl.product_id
WHERE hex(fcl.cash_event_type) = 'D09FD180D0BED0B4D0B0D0B6D0B0'
  AND fcl.check_date BETWEEN toDate('{date_from}') AND toDate('{date_to}')
{limit_clause}
