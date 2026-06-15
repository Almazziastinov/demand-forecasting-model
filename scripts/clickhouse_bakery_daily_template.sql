/*
Return bakery-day aggregates for the production dataset refresh.

Output columns:
  - date
  - bakery_id
  - bakery_name
  - city
  - bakery_sales
  - line_amount_sum
  - priced_quantity
  - price_x_qty_sum

The exporter will substitute:
  - {date_from}
  - {date_to}
  - {limit_clause}
*/
SELECT
    sales.check_date AS date,
    sales.bakery_id AS bakery_id,
    any(db.bakery_name) AS bakery_name,
    any(db.city) AS city,
    sum(toFloat64(sales.quantity)) AS bakery_sales,
    sum(ifNull(toFloat64(sales.line_amount), 0.0)) AS line_amount_sum,
    sum(if(isNull(sales.price), 0.0, toFloat64(sales.quantity))) AS priced_quantity,
    sum(if(isNull(sales.price), 0.0, toFloat64(sales.price) * toFloat64(sales.quantity))) AS price_x_qty_sum
FROM (
    SELECT DISTINCT
        fcl.check_datetime,
        fcl.check_date,
        fcl.bakery_id,
        fcl.product_id,
        fcl.quantity,
        fcl.price,
        fcl.line_amount,
        fcl.cash_event_type
    FROM Svezhar.fct_check_lines AS fcl
    WHERE hex(fcl.cash_event_type) = 'D09FD180D0BED0B4D0B0D0B6D0B0'
      AND fcl.check_date BETWEEN toDate('{date_from}') AND toDate('{date_to}')
) AS sales
ANY LEFT JOIN Svezhar.dim_bakeries AS db
    ON db.bakery_id = sales.bakery_id
GROUP BY
    sales.check_date,
    sales.bakery_id
ORDER BY
    bakery_id,
    date
{limit_clause}
