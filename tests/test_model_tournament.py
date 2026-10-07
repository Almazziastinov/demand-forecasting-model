from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.model_tournament.baselines import predict_baseline
from src.model_tournament.metrics import canonical_metrics
from src.model_tournament.runner import TournamentConfig, run_tournament, validate_panel
from src.model_tournament.trainable import train_and_predict


def _panel(days: int = 50) -> pd.DataFrame:
    rows = []
    for date in pd.date_range("2026-01-01", periods=days):
        for bakery_id in (1, 2):
            rows.append(
                {
                    "date": date,
                    "bakery_id": bakery_id,
                    "product_id": 100,
                    "city": "Kazan",
                    "category": "Bread",
                    "sold": float(date.dayofweek + bakery_id),
                    "demand": float(date.dayofweek + bakery_id + 1),
                }
            )
    return pd.DataFrame(rows)


def test_lag7_uses_exact_calendar_history() -> None:
    panel = _panel()
    prediction = predict_baseline(
        "lag7", panel, target_col="demand", key_cols=["bakery_id", "product_id"]
    )
    assert np.isnan(prediction.iloc[0])
    assert prediction.iloc[14] == panel.iloc[0]["demand"]


def test_weighted_weekday_uses_available_same_weekday_tail() -> None:
    panel = _panel(43)
    dropped_date = pd.Timestamp("2026-01-15")
    panel = panel[
        ~(
            panel["date"].eq(dropped_date)
            & panel["bakery_id"].eq(1)
            & panel["product_id"].eq(100)
        )
    ].reset_index(drop=True)
    prediction = predict_baseline(
        "weighted_weekday_formula_v1",
        panel,
        target_col="demand",
        key_cols=["bakery_id", "product_id"],
    )
    row = panel[
        panel["date"].eq(pd.Timestamp("2026-01-22"))
        & panel["bakery_id"].eq(1)
    ].index.item()
    # 1 and 8 January are the two available prior Thursdays.
    assert prediction.iloc[row] == pytest.approx(5.0)


def test_history_cutoff_blocks_future_facts_for_multiday_forecast() -> None:
    panel = _panel(50)
    panel.loc[panel["date"].eq(pd.Timestamp("2026-02-10")), "demand"] = 999.0
    target_row = panel[panel["date"].eq(pd.Timestamp("2026-02-17"))]
    target_index = target_row.index[0]
    unrestricted = predict_baseline(
        "weighted_weekday_formula_v1",
        panel,
        target_col="demand",
        key_cols=["bakery_id", "product_id"],
    )
    restricted = predict_baseline(
        "weighted_weekday_formula_v1",
        panel,
        target_col="demand",
        key_cols=["bakery_id", "product_id"],
        history_cutoff=pd.Timestamp("2026-02-03"),
    )
    assert unrestricted.iloc[target_index] > restricted.iloc[target_index]
    lag7_restricted = predict_baseline(
        "lag7",
        panel,
        target_col="demand",
        key_cols=["bakery_id", "product_id"],
        history_cutoff=pd.Timestamp("2026-02-03"),
    )
    assert np.isnan(lag7_restricted.iloc[target_index])


def test_weighted_weekday_ignores_history_older_than_eight_weeks() -> None:
    panel = _panel(80)
    target = pd.Timestamp("2026-03-20")
    pair = panel["bakery_id"].eq(1)
    same_dow = panel["date"].dt.dayofweek.eq(target.dayofweek)
    recent = panel["date"].between(target - pd.Timedelta(days=56), target)
    panel = panel[~(pair & same_dow & recent & panel["date"].ne(target))]
    panel = panel.reset_index(drop=True)
    prediction = predict_baseline(
        "weighted_weekday_formula_v1",
        panel,
        target_col="demand",
        key_cols=["bakery_id", "product_id"],
    )
    target_index = panel[
        panel["date"].eq(target) & panel["bakery_id"].eq(1)
    ].index.item()
    assert prediction.iloc[target_index] == 0.0


def test_canonical_metrics_report_zero_safe_mape_and_bias() -> None:
    frame = pd.DataFrame(
        {
            "actual": [0.0, 10.0, 20.0],
            "prediction": [2.0, 8.0, 25.0],
            "sales": [0.0, 8.0, 18.0],
        }
    )
    metrics = canonical_metrics(frame, sales_col="sales")
    assert metrics["zero_actual_rows"] == 1
    assert metrics["mape_rows"] == 2
    assert metrics["bias_qty"] == 5.0
    assert metrics["restored_lost_qty"] == 4.0
    assert metrics["false_positive_rows"] == 1
    assert metrics["false_positive_qty"] == 2.0


def test_validate_panel_rejects_duplicate_scope(tmp_path) -> None:
    panel = pd.concat([_panel(2), _panel(2).iloc[[0]]], ignore_index=True)
    path = tmp_path / "panel.csv"
    panel.to_csv(path, index=False)
    config = TournamentConfig(
        input_path=path,
        output_dir=tmp_path / "out",
        date_from="2026-01-01",
        date_to="2026-01-02",
        target_col="demand",
        models=("zero",),
    )
    with pytest.raises(ValueError, match="duplicate canonical keys"):
        validate_panel(panel, config)


def test_validate_panel_rejects_demand_below_sales(tmp_path) -> None:
    panel = _panel(2)
    panel.loc[0, "demand"] = panel.loc[0, "sold"] - 1.0
    config = TournamentConfig(
        input_path=tmp_path / "panel.csv",
        output_dir=tmp_path / "out",
        date_from="2026-01-01",
        date_to="2026-01-02",
        target_col="demand",
        sales_col="sold",
        models=("zero",),
    )
    with pytest.raises(ValueError, match="below observed sales"):
        validate_panel(panel, config)


def test_run_tournament_writes_common_scope_outputs(tmp_path) -> None:
    panel = _panel()
    path = tmp_path / "panel.parquet"
    panel.to_parquet(path, index=False)
    output = tmp_path / "out"
    config = TournamentConfig(
        input_path=path,
        output_dir=output,
        date_from="2026-02-05",
        date_to="2026-02-19",
        target_col="demand",
        sales_col="sold",
        models=("lag7", "same_weekday_mean2", "weighted_weekday_formula_v1"),
    )
    paths = run_tournament(config)
    leaderboard = pd.read_csv(paths["summary"])
    detail = pd.read_parquet(paths["detail"])
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))

    assert set(leaderboard["model"]) == {
        "lag7",
        "same_weekday_mean2",
        "weighted_weekday_formula_v1",
    }
    assert detail.groupby("model").size().nunique() == 1
    assert metadata["production_write"] is False
    assert (output / "breakdowns" / "by_city.csv").exists()
    assert paths["wins_losses"].exists()


def test_artifact_prediction_column_uses_same_scope(tmp_path) -> None:
    panel = _panel()
    panel["saved_forecast"] = panel["demand"] + 1.0
    path = tmp_path / "panel.parquet"
    panel.to_parquet(path, index=False)
    output = tmp_path / "out"
    config = TournamentConfig(
        input_path=path,
        output_dir=output,
        date_from="2026-02-05",
        date_to="2026-02-10",
        target_col="demand",
        models=("lag7",),
        prediction_columns=(("saved_model", "saved_forecast"),),
    )
    paths = run_tournament(config)
    leaderboard = pd.read_csv(paths["summary"])
    coverage = pd.read_csv(paths["coverage"])

    saved = leaderboard[leaderboard["model"].eq("saved_model")].iloc[0]
    assert saved["mae"] == 1.0
    assert set(coverage["model"]) == {"lag7", "saved_model"}
    assert (output / "breakdowns" / "by_category_name.csv").exists() is False


def test_pairwise_wins_use_explicit_reference(tmp_path) -> None:
    panel = _panel()
    panel["perfect"] = panel["demand"]
    path = tmp_path / "panel.parquet"
    panel.to_parquet(path, index=False)
    config = TournamentConfig(
        input_path=path,
        output_dir=tmp_path / "out",
        date_from="2026-02-05",
        date_to="2026-02-10",
        target_col="demand",
        models=("zero",),
        prediction_columns=(("perfect", "perfect"),),
        reference_model="zero",
    )
    paths = run_tournament(config)
    wins = pd.read_csv(paths["wins_losses"])
    perfect = wins[wins["model"].eq("perfect")].iloc[0]
    assert perfect["wins"] == perfect["rows"]
    assert perfect["mae_delta_vs_reference"] < 0


def test_artifact_negative_prediction_is_rejected(tmp_path) -> None:
    panel = _panel()
    panel["saved_forecast"] = panel["demand"]
    panel.loc[panel["date"].eq(pd.Timestamp("2026-02-05")), "saved_forecast"] = -1.0
    path = tmp_path / "panel.parquet"
    panel.to_parquet(path, index=False)
    config = TournamentConfig(
        input_path=path,
        output_dir=tmp_path / "out",
        date_from="2026-02-05",
        date_to="2026-02-10",
        target_col="demand",
        models=("zero",),
        prediction_columns=(("saved", "saved_forecast"),),
    )
    with pytest.raises(ValueError, match="negative predictions"):
        run_tournament(config)


def test_artifact_malformed_prediction_is_rejected(tmp_path) -> None:
    panel = _panel()
    panel["saved_forecast"] = panel["demand"].astype(str)
    panel.loc[panel["date"].eq(pd.Timestamp("2026-02-05")), "saved_forecast"] = "bad"
    path = tmp_path / "panel.csv"
    panel.to_csv(path, index=False)
    config = TournamentConfig(
        input_path=path,
        output_dir=tmp_path / "out",
        date_from="2026-02-05",
        date_to="2026-02-10",
        target_col="demand",
        models=("zero",),
        prediction_columns=(("saved", "saved_forecast"),),
    )
    with pytest.raises(ValueError, match="non-numeric artifact predictions"):
        run_tournament(config)


def test_corrected_asymmetric_model_uses_native_quantile_objective() -> None:
    panel = _panel(65)
    evaluation_mask = panel["date"].ge(pd.Timestamp("2026-02-20"))
    prediction, provenance = train_and_predict(
        "lgbm_quantile_p60",
        panel,
        target_col="sold",
        key_cols=["bakery_id", "product_id"],
        evaluation_mask=evaluation_mask,
        train_end=pd.Timestamp("2026-02-19"),
        n_estimators=20,
    )
    assert provenance["objective"] == "quantile"
    assert provenance["alpha"] == 0.6
    assert prediction.loc[evaluation_mask].notna().all()
    assert prediction.loc[~evaluation_mask].isna().all()


def test_trainable_model_cannot_train_on_evaluation_dates() -> None:
    panel = _panel(65)
    evaluation_mask = panel["date"].ge(pd.Timestamp("2026-02-20"))
    with pytest.raises(ValueError, match="train_end must precede"):
        train_and_predict(
            "lgbm_quantile_p60",
            panel,
            target_col="sold",
            key_cols=["bakery_id", "product_id"],
            evaluation_mask=evaluation_mask,
            train_end=pd.Timestamp("2026-02-20"),
            n_estimators=20,
        )
