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
    fcln.check_datetime AS check_datetime,
    fcln.check_date AS check_date,
    fcln.cash_event_type AS cash_event_type,
    fcln.quantity AS quantity,
    fcln.price AS price,
    fcln.line_amount AS line_amount,
    fcln.freshness AS freshness,
    fcln.bakery_id AS bakery_id,
    db.bakery_name AS bakery_name,
    db.city AS city,
    fcln.product_id AS product_id,
    dp.product_name AS product_name,
    dp.category_name AS category_name
FROM Svezhar.fct_check_lines_new AS fcln FINAL
JOIN Svezhar.dim_bakeries AS db
    ON db.bakery_id = fcln.bakery_id
JOIN Svezhar.dim_products AS dp
    ON dp.product_id = fcln.product_id
WHERE fcln.cash_event_type = 'Продажа'
  AND fcln.check_date BETWEEN toDate('{date_from}') AND toDate('{date_to}')
{limit_clause}
