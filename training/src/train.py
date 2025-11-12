from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score
from .utils import load_config, ARTIFACTS_DIR, MODELS_DIR, next_model_version
from . import data_ingestion
from . import preprocess
from pipelines.notify import send as notify


def build_model(input_dim: int, hidden_units: list[int], dropout: float, lr: float) -> tf.keras.Model:
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,)))
    for h in hidden_units:
        model.add(tf.keras.layers.Dense(h, activation="relu"))
        if dropout > 0:
            model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def train_and_deploy():
    cfg = load_config()
    exp = cfg["experiment"]
    tr = cfg["training"]

    # 1) Ingestion
    paths = data_ingestion.generate_and_split()

    # 2) Preprocess
    proc = preprocess.fit_and_transform(paths["train"], paths["val"], paths["test"])

    # 3) Load processed data
    train = pd.read_csv(proc["processed"]["train"])
    val = pd.read_csv(proc["processed"]["val"])
    test = pd.read_csv(proc["processed"]["test"])

    feature_cols = [c for c in train.columns if c != "target"]
    X_train, y_train = train[feature_cols].values.astype(np.float32), train["target"].values.astype(np.int32)
    X_val, y_val = val[feature_cols].values.astype(np.float32), val["target"].values.astype(np.int32)
    X_test, y_test = test[feature_cols].values.astype(np.float32), test["target"].values.astype(np.int32)

    # 4) Model
    mlflow.set_experiment(exp["name"])
    mlflow.tensorflow.autolog()
    with mlflow.start_run(run_name="keras_tabular") as run:
        model = build_model(
            input_dim=len(feature_cols),
            hidden_units=tr["hidden_units"],
            dropout=tr["dropout"],
            lr=tr["learning_rate"],
        )
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=tr["epochs"],
            batch_size=tr["batch_size"],
            verbose=2,
        )

        # 5) Evaluate
        y_prob = model.predict(X_test, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)
        auc = float(roc_auc_score(y_test, y_prob))
        acc = float(accuracy_score(y_test, y_pred))
        mlflow.log_metrics({"test_auc": auc, "test_acc": acc})

        # 6) Persist SavedModel for TF Serving (canary path)
        canary_base = MODELS_DIR / f"{exp['model_name']}_canary"
        version = next_model_version(canary_base)
        export_path = canary_base / str(version)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        # Export Keras 3 SavedModel for TF Serving
        try:
            model.export(str(export_path))
        except Exception:
            # Fallback for older TF/Keras
            tf.saved_model.save(model, str(export_path))
        mlflow.log_param("serving_canary_version", version)

        # 7) Log model to MLflow Model Registry and artifacts for API
        mlflow.tensorflow.log_model(
            model,
            artifact_path="model",
            registered_model_name=exp["model_name"],
        )
        mlflow.log_artifact(proc["scaler_path"], artifact_path="artifacts")
        mlflow.log_artifact(proc["schema_path"], artifact_path="artifacts")

        # Transition latest to Staging
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            rm = client.get_registered_model(exp["model_name"])
            latest = max(rm.latest_versions, key=lambda v: int(v.version))
            client.transition_model_version_stage(
                name=exp["model_name"], version=latest.version, stage="Staging", archive_existing_versions=False
            )
        except Exception:
            pass

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    result = {
        "saved_model_canary": str(export_path),
        "scaler": proc["scaler_path"],
        "schema": proc["schema_path"],
        "metrics": {"test_auc": auc, "test_acc": acc},
    }
    print(result)
    try:
        notify("Training completed", "Model trained and logged", result)
    except Exception:
        pass


if __name__ == "__main__":
    train_and_deploy()
