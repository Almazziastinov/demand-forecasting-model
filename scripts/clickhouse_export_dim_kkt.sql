/*
Canonical export for Svezhar.dim_kkt.

Placeholders:
  - {date_from}
  - {date_to}
  - {limit_clause}

This template does not need date filters, but placeholders are kept so it can
be used with the generic exporter consistently.
*/
SELECT
    kkt_id,
    kkt_name,
    kkt_number,
    organization_id,
    organization_name,
    bakery_id
FROM Svezhar.dim_kkt
{limit_clause}
