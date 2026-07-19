/*
Pilot-only hourly check-line extract for stockout-demand research.

This keeps the same canonical column names as clickhouse_export_template_stg.sql
but filters to the current 11 pilot bakeries to avoid exporting the full
multi-gigabyte raw sales table.

Placeholders:
  - {date_from}
  - {date_to}
  - {limit_clause}
*/
SELECT
    check_datetime,
    check_date,
    cash_event_type,
    quantity,
    price,
    line_amount,
    freshness,
    toInt64OrZero(bakery_id_raw) AS bakery_id,
    bakery_name,
    city,
    toInt64OrZero(product_id_raw) AS product_id,
    product_name,
    category_name
FROM
(
    SELECT
        fcl.check_datetime     AS check_datetime,
        fcl.check_date         AS check_date,
        fcl.cash_event_type    AS cash_event_type,
        fcl.quantity           AS quantity,
        fcl.price              AS price,
        fcl.line_amount        AS line_amount,
        fcl.freshness          AS freshness,
        fcl.bakery_id          AS bakery_id_raw,
        db.bakery_name         AS bakery_name,
        db.city                AS city,
        fcl.product_id         AS product_id_raw,
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
      AND fcl.bakery_id IN (
        '000000016',
        '000000020',
        '000000021',
        '000000022',
        '000000028',
        '000000080',
        '000000089',
        '000000107',
        '000000221',
        '000000222',
        '000000257'
      )
)
{limit_clause}
