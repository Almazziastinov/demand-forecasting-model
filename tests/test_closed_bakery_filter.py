from __future__ import annotations

import pandas as pd

from pipelines.forecast_publish.production_dataset_refresh import (
    exclude_closed_bakeries,
)
from src.experiments_v2.build_bakery_daily_dataset import (
    BAKERY_ID_COL,
    DATE_COL,
    TARGET_COL,
)


def test_excludes_only_bakeries_older_than_30_days() -> None:
    frame = pd.DataFrame(
        {
            DATE_COL: ["2026-07-30", "2026-07-31", "2026-08-30"],
            BAKERY_ID_COL: [101, 102, 103],
            TARGET_COL: [10.0, 10.0, 10.0],
        }
    )

    filtered, closed = exclude_closed_bakeries(
        frame,
        as_of_date="2026-08-30",
    )

    assert closed == [101]
    assert sorted(filtered[BAKERY_ID_COL].tolist()) == [102, 103]


def test_zero_rows_do_not_replace_last_positive_sale() -> None:
    frame = pd.DataFrame(
        {
            DATE_COL: ["2026-07-01", "2026-08-30", "2026-08-20"],
            BAKERY_ID_COL: [101, 101, 102],
            TARGET_COL: [5.0, 0.0, 3.0],
        }
    )

    filtered, closed = exclude_closed_bakeries(
        frame,
        as_of_date="2026-08-30",
    )

    assert closed == [101]
    assert filtered[BAKERY_ID_COL].unique().tolist() == [102]
