/*
Canonical export for Svezhar.fct_moves.

Placeholders:
  - {date_from}
  - {date_to}
  - {limit_clause}
*/
SELECT
    move_id,
    move_date,
    product_id,
    sender_id,
    receiver_id,
    quantity,
    _updated_at
FROM Svezhar.fct_moves
WHERE move_date BETWEEN toDate('{date_from}') AND toDate('{date_to}')
{limit_clause}
