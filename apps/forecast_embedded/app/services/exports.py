from __future__ import annotations

import csv
import io

from app.auth import AuthContext
from app.services import bakery as bakery_service


def export_bakery_top_sku_csv(
    run_id: str,
    forecast_date: str,
    bakery_id: int,
    auth: AuthContext,
) -> str:
    rows = bakery_service.get_top_sku(
        run_id,
        forecast_date,
        bakery_id,
        auth,
        limit=10000,
    )
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["product_id", "product_name", "category_name", "forecast_qty"],
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()
