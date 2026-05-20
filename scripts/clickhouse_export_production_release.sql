/*
Canonical export for Svezhar.fct_production_release.

Placeholders:
  - {date_from}
  - {date_to}
  - {limit_clause}
*/
SELECT
    _UUID,
    release_id,
    line_id,
    release_date,
    bakery_id,
    product_id,
    quantity,
    baker_name,
    _updated_at
FROM Svezhar.fct_production_release
WHERE release_date BETWEEN toDate('{date_from}') AND toDate('{date_to}')
{limit_clause}
