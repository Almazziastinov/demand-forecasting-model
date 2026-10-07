from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.backtest_chronos2_lora_recent import checked_adapter
from scripts.run_chronos2_fixed_origin import MODEL_VARIANTS, _sha256


def test_checked_adapter_requires_verified_pre_origin_artifact(tmp_path) -> None:
    adapter = tmp_path / "adapter_model.safetensors"
    adapter.write_bytes(b"synthetic-adapter")
    _, repo, revision = MODEL_VARIANTS["small"]
    metadata = {
        "score_valid": True,
        "model_repo": repo,
        "model_revision": revision,
        "num_steps": 1000,
        "train_cutoff": "2026-07-05",
        "adapter_file": str(adapter),
        "adapter_sha256": _sha256(adapter),
    }
    meta_file = tmp_path / "metadata.json"
    meta_file.write_text(json.dumps(metadata), encoding="utf-8")
    assert checked_adapter(tmp_path, pd.Timestamp("2026-09-17"))[0] == adapter

    metadata["train_cutoff"] = "2026-09-17"
    meta_file.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="pre-origin"):
        checked_adapter(tmp_path, pd.Timestamp("2026-09-17"))

    metadata["train_cutoff"] = "2026-07-05"
    metadata["score_valid"] = False
    meta_file.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="verified"):
        checked_adapter(tmp_path, pd.Timestamp("2026-09-17"))
