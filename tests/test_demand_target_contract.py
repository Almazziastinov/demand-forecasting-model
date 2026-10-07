from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.model_tournament.demand_target_contract import (
    validate_demand_target_contract,
)


def _target() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "demand_lower_bound": [0.0, 5.0, 10.0],
            "imputed_demand": [0.0, 2.0, 0.0],
            "demand_point_estimate": [0.0, 7.0, 10.0],
        }
    )


def test_local_reduction_is_allowed_when_total_demand_exceeds_sales() -> None:
    frame = _target()
    frame["outlier_reduction"] = [0.0, 0.0, 1.0]
    frame.loc[2, "demand_point_estimate"] = 9.0
    validate_demand_target_contract(
        frame, reduction_column="outlier_reduction", require_global_uplift=True
    )
    assert frame.loc[2, "demand_point_estimate"] < frame.loc[2, "demand_lower_bound"]
    assert frame["demand_point_estimate"].sum() > frame["demand_lower_bound"].sum()


def test_global_net_reduction_is_rejected() -> None:
    frame = _target()
    frame["outlier_reduction"] = [0.0, 0.0, 3.0]
    frame.loc[2, "demand_point_estimate"] = 7.0
    with pytest.raises(ValueError, match="Total restored demand must exceed"):
        validate_demand_target_contract(
            frame, reduction_column="outlier_reduction", require_global_uplift=True
        )


def test_target_rejects_negative_or_inconsistent_imputation() -> None:
    frame = _target()
    frame.loc[1, "imputed_demand"] = -1.0
    with pytest.raises(ValueError, match="Imputed demand cannot be negative"):
        validate_demand_target_contract(frame)
    frame = _target()
    frame.loc[1, "imputed_demand"] = 1.0
    with pytest.raises(ValueError, match="sales plus restoration minus reduction"):
        validate_demand_target_contract(frame)

    frame = _target()
    frame["outlier_reduction"] = [0.0, 0.0, -1.0]
    with pytest.raises(ValueError, match="Outlier reduction cannot be negative"):
        validate_demand_target_contract(frame, reduction_column="outlier_reduction")


def test_target_rejects_missing_and_nonfinite_values() -> None:
    frame = _target()
    frame.loc[0, "demand_point_estimate"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_demand_target_contract(frame)
