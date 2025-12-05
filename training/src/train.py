# =============================================================================
# AutoMLOps - End-to-End MLOps Pipeline
# Copyright (c) 2025 Joy Biswas
# Licensed under MIT License
# https://github.com/JoyBiswas1403/AutoMLOps
# =============================================================================
"""Main training pipeline with MLflow integration and TensorFlow Serving export."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score

from . import data_ingestion, preprocess
from .model_card import generate_model_card
from .utils import ARTIFACTS_DIR, MODELS_DIR, load_config, next_model_version

# Import notification module
try:
    from pipelines.notify import send as notify
except ImportError:
    def notify(*args: Any, **kwargs: Any) -> None:
        """Fallback notify function when pipelines module is not available."""
        pass


def build_model(
    input_dim: int,
    hidden_units: list[int],
    dropout: float,
    lr: float,
) -> tf.keras.Model:
    """Build a feedforward neural network for binary classification.

    Args:
        input_dim: Number of input features.
        hidden_units: List of units for each hidden layer.
        dropout: Dropout rate (0 to disable).
        lr: Learning rate for Adam optimizer.

    Returns:
        Compiled Keras model ready for training.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,)))

    for units in hidden_units:
        model.add(tf.keras.layers.Dense(units, activation="relu"))
        if dropout > 0:
            model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def export_model_for_serving(
    model: tf.keras.Model,
    export_path: Path,
) -> None:
    """Export model in SavedModel format for TensorFlow Serving.

    Args:
        model: Trained Keras model.
        export_path: Directory path for the SavedModel.
    """
    export_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Keras 3 export method
        model.export(str(export_path))
    except (AttributeError, TypeError):
        # Fallback for older TF/Keras versions
        tf.saved_model.save(model, str(export_path))


def train_and_deploy() -> dict[str, Any]:
    """Execute the full training and deployment pipeline.

    This function:
    1. Generates/loads and preprocesses data
    2. Trains a neural network model
    3. Evaluates on test set
    4. Exports for TensorFlow Serving (canary)
    5. Logs to MLflow registry
    6. Generates model card

    Returns:
        Dictionary containing paths and metrics from training.
    """
    cfg = load_config()
    exp = cfg["experiment"]
    tr = cfg["training"]
    data_cfg = cfg.get("data", {})

    # 1) Data Ingestion
    data_source = data_cfg.get("source", "synthetic")
    print(f"\n{'='*60}")
    print(f"AutoMLOps Training Pipeline - {data_source.upper()} data")
    print(f"{'='*60}\n")

    paths = data_ingestion.generate_and_split()

    # 2) Preprocessing
    proc = preprocess.fit_and_transform(paths["train"], paths["val"], paths["test"])
    n_features = proc.get("n_features", 20)

    # 3) Load processed data
    train = pd.read_csv(proc["processed"]["train"])
    val = pd.read_csv(proc["processed"]["val"])
    test = pd.read_csv(proc["processed"]["test"])

    # 4) Validate data
    print("\nValidating data...")
    try:
        from .validation import validate_training_data

        validation_result = validate_training_data(train, val, test)
        if validation_result["all_valid"]:
            print("✅ Data validation passed")
        else:
            print("⚠️ Data validation has warnings/errors")
        validation_report = validation_result.get("report_path")
    except Exception as e:
        print(f"Warning: Data validation failed: {e}")
        validation_report = None

    feature_cols = [c for c in train.columns if c not in ("target", "row_id")]
    X_train = train[feature_cols].values.astype(np.float32)
    y_train = train["target"].values.astype(np.int32)
    X_val = val[feature_cols].values.astype(np.float32)
    y_val = val["target"].values.astype(np.int32)
    X_test = test[feature_cols].values.astype(np.float32)
    y_test = test["target"].values.astype(np.int32)

    print(f"\nTraining with {len(feature_cols)} features")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Positive rate: {y_train.mean():.2%}")

    # 4) MLflow tracking
    mlflow.set_experiment(exp["name"])
    mlflow.tensorflow.autolog()

    with mlflow.start_run(run_name="keras_fraud_detection") as run:
        # Log data source
        mlflow.set_tag("data_source", data_source)
        mlflow.log_param("n_features", n_features)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("positive_rate", float(y_train.mean()))

        # Build and train model
        model = build_model(
            input_dim=len(feature_cols),
            hidden_units=tr["hidden_units"],
            dropout=tr["dropout"],
            lr=tr["learning_rate"],
        )

        print(f"\nModel architecture: {tr['hidden_units']}")
        print(f"Training for {tr['epochs']} epochs...\n")

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

        print(f"\n{'='*60}")
        print(f"Test Results: AUC={auc:.4f}, Accuracy={acc:.4f}")
        print(f"{'='*60}\n")

        # 6) Export for TF Serving (canary path)
        canary_base = MODELS_DIR / f"{exp['model_name']}_canary"
        version = next_model_version(canary_base)
        export_path = canary_base / str(version)
        export_model_for_serving(model, export_path)
        mlflow.log_param("serving_canary_version", version)

        # 7) Log to MLflow Model Registry
        mlflow.tensorflow.log_model(
            model,
            artifact_path="model",
            registered_model_name=exp["model_name"],
        )

        # Log preprocessing artifacts
        mlflow.log_artifact(proc["scaler_path"], artifact_path="artifacts")
        mlflow.log_artifact(proc["schema_path"], artifact_path="artifacts")

        # Log validation report
        if validation_report and Path(validation_report).exists():
            mlflow.log_artifact(validation_report, artifact_path="validation")

        # 8) Generate explainability report
        print("\nGenerating explainability report...")
        try:
            from .explainability import generate_explainability_report

            explain_result = generate_explainability_report(
                model=model,
                X_train=X_train,
                X_test=X_test,
                feature_names=feature_cols,
            )

            # Log SHAP plots to MLflow
            for plot_name, plot_path in explain_result.get("plots", {}).items():
                if plot_path and Path(plot_path).exists():
                    mlflow.log_artifact(str(plot_path), artifact_path="explainability")

            top_features = explain_result.get("top_features", [])
            if top_features:
                mlflow.log_param("top_features", ", ".join(top_features[:5]))
        except Exception as e:
            print(f"Warning: Explainability report failed: {e}")
            top_features = []

        # Generate and log model card
        card = generate_model_card(
            metrics={"test_auc": auc, "test_acc": acc},
            params=tr,
            data_source=data_source,
            n_features=n_features,
            train_samples=len(X_train),
            top_features=top_features,
        )
        mlflow.log_artifact(str(card), artifact_path="artifacts")

        # Transition to Staging
        _transition_to_staging(exp["model_name"])

    # Ensure artifacts directory exists
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    result = {
        "saved_model_canary": str(export_path),
        "scaler": proc["scaler_path"],
        "schema": proc["schema_path"],
        "metrics": {"test_auc": auc, "test_acc": acc},
        "data_source": data_source,
        "n_features": n_features,
    }

    print(result)
    _send_notification("Training completed", f"Model trained on {data_source} data", result)

    return result


def _transition_to_staging(model_name: str) -> None:
    """Transition the latest model version to Staging stage.

    Args:
        model_name: Name of the registered model.
    """
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        rm = client.get_registered_model(model_name)
        latest = max(rm.latest_versions, key=lambda v: int(v.version))
        client.transition_model_version_stage(
            name=model_name,
            version=latest.version,
            stage="Staging",
            archive_existing_versions=False,
        )
        print(f"Model v{latest.version} transitioned to Staging")
    except Exception as e:
        print(f"Warning: Could not transition to staging: {e}")


def _send_notification(title: str, message: str, data: dict[str, Any]) -> None:
    """Send a notification about training completion.

    Args:
        title: Notification title.
        message: Notification message.
        data: Additional data to include.
    """
    try:
        notify(title, message, data)
    except Exception as e:
        print(f"Warning: Notification failed: {e}")


if __name__ == "__main__":
    train_and_deploy()
