/*
Pilot-only daily sales/production/stock extract for stockout-demand research.

Placeholders:
  - {date_from}
  - {date_to}
  - {limit_clause}
*/
SELECT
    date,
    toInt64OrZero(bakery_id_raw) AS bakery_id,
    bakery_name,
    city,
    toInt64OrZero(product_id_raw) AS product_id,
    product_name,
    category_name,
    qty_sold,
    qty_produced,
    qty_received,
    qty_sent,
    stock_balance,
    last_sale_time,
    revenue
FROM
(
    SELECT
        dt AS date,
        bakery_id AS bakery_id_raw,
        bakery_name,
        city,
        product_id AS product_id_raw,
        product_name,
        category_name,
        qty_sold,
        qty_produced,
        qty_received,
        qty_sent,
        stock_balance,
        last_sale_time,
        revenue
    FROM Svezhar.mart_zero_sales_60d
    WHERE dt BETWEEN toDate('{date_from}') AND toDate('{date_to}')
      AND bakery_id IN (
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
