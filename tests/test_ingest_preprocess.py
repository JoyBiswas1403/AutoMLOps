import os
import importlib
import pandas as pd
from pathlib import Path


def test_ingest_and_preprocess_tmp(tmp_path, monkeypatch):
    # Redirect ARTIFACTS_DIR to temp
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    # Reload utils to pick up new env
    utils = importlib.import_module("training.src.utils")
    importlib.reload(utils)

    data_ingestion = importlib.import_module("training.src.data_ingestion")
    importlib.reload(data_ingestion)
    preprocess = importlib.import_module("training.src.preprocess")
    importlib.reload(preprocess)

    paths = data_ingestion.generate_and_split()
    out = preprocess.fit_and_transform(paths["train"], paths["val"], paths["test"])

    # Files should exist
    assert Path(out["scaler_path"]).exists()
    assert Path(out["schema_path"]).exists()
    for k, v in out["processed"].items():
        assert Path(v).exists()
        df = pd.read_csv(v)
        assert "target" in df.columns
