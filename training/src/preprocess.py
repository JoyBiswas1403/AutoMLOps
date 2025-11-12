from __future__ import annotations
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib
from .utils import ARTIFACTS_DIR

PROCESSED_DIR = ARTIFACTS_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def fit_and_transform(train_csv: Path, val_csv: Path, test_csv: Path):
    train = pd.read_csv(train_csv)
    val = pd.read_csv(val_csv)
    test = pd.read_csv(test_csv)

    feature_cols = [c for c in train.columns if c != "target"]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols])
    X_val = scaler.transform(val[feature_cols])
    X_test = scaler.transform(test[feature_cols])

    # Save processed datasets
    pd.DataFrame(X_train, columns=feature_cols).assign(target=train["target"]).to_csv(PROCESSED_DIR / "train.csv", index=False)
    pd.DataFrame(X_val, columns=feature_cols).assign(target=val["target"]).to_csv(PROCESSED_DIR / "val.csv", index=False)
    pd.DataFrame(X_test, columns=feature_cols).assign(target=test["target"]).to_csv(PROCESSED_DIR / "test.csv", index=False)

    # Save scaler for serving
    scaler_path = ARTIFACTS_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)

    schema_path = ARTIFACTS_DIR / "schema.json"
    schema = {"feature_order": feature_cols}
    schema_path.write_text(__import__("json").dumps(schema, indent=2))

    return {
        "scaler_path": str(scaler_path),
        "schema_path": str(schema_path),
        "processed": {
            "train": str(PROCESSED_DIR / "train.csv"),
            "val": str(PROCESSED_DIR / "val.csv"),
            "test": str(PROCESSED_DIR / "test.csv"),
        },
    }


if __name__ == "__main__":
    raise SystemExit("Run via training/src/train.py")
