"""Tests for src/config.py -- config loads and values are sane."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    FEATURES, TARGET, CATEGORICAL_COLS, MODEL_PARAMS,
    BASE_DIR, DATA_DIR, MODELS_DIR,
)


def test_features_not_empty():
    assert len(FEATURES) > 0


def test_target_is_string():
    assert isinstance(TARGET, str)
    assert len(TARGET) > 0


def test_categorical_cols_subset_of_features():
    for col in CATEGORICAL_COLS:
        assert col in FEATURES, f"{col} not in FEATURES"


def test_model_params_has_required_keys():
    required = ["n_estimators", "learning_rate", "num_leaves", "max_depth"]
    for key in required:
        assert key in MODEL_PARAMS, f"{key} missing from MODEL_PARAMS"


def test_paths_are_absolute():
    assert os.path.isabs(BASE_DIR)
    assert os.path.isabs(DATA_DIR)
    assert os.path.isabs(MODELS_DIR)
