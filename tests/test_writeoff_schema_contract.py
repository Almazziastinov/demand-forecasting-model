from datetime import date

import pandas as pd

from pipelines.forecast_publish import weighted_weekday_production as forecast


class WriteoffSchemaClient:
    def query_df(self, query: str, parameters: dict | None = None) -> pd.DataFrame:
        if "fct_write_offs" in query:
            assert "GROUP BY document_number,line_id" in query
            assert "write_off_doc_num" not in query
            return pd.DataFrame(
                columns=["date", "bakery_id", "product_id", "written_off_qty"]
            )
        if "fct_check_lines" in query:
            return pd.DataFrame(
                columns=[
                    "date",
                    "bakery_id",
                    "product_id",
                    "observed_sales_qty",
                    "first_sale_time",
                    "last_sale_time",
                ]
            )
        if "fct_production_release" in query:
            return pd.DataFrame(
                columns=["date", "bakery_id", "product_id", "release_qty"]
            )
        if "fct_moves" in query:
            return pd.DataFrame(
                columns=[
                    "date",
                    "bakery_id",
                    "product_id",
                    "incoming_move_qty",
                    "outgoing_move_qty",
                ]
            )
        raise AssertionError(f"Unexpected query: {query}")


def test_legacy_writeoff_query_uses_current_document_number_schema() -> None:
    result = forecast._fetch_legacy_history_demand(
        WriteoffSchemaClient(), [date(2026, 11, 19)], [1]
    )

    assert result.empty


def test_regional_writeoff_query_uses_current_document_number_schema() -> None:
    result = forecast._fetch_regional_history_demand(
        WriteoffSchemaClient(),
        [date(2026, 11, 19)],
        {"alm": [100001], "irk": [], "nov": []},
    )

    assert result.empty
