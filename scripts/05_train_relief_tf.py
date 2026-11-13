#!/usr/bin/env python
"""Step 5 - TensorFlow / Keras model: P(complaint closes with monetary relief).

Run this in the TensorFlow environment (requirements-tf.txt). Inputs are the
narrative embeddings from step 3 plus one-hot structured fields (see
triage/features.py). The output probability feeds the LangGraph risk node.

  python scripts/05_train_relief_tf.py

MLflow records: params, per-epoch metrics (Keras callback), test PR-AUC / ROC-AUC,
precision & recall at the configured threshold, the feature spec, and the
model (tensorflow flavor) as a new registered version.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from triage.config import CFG, path  # noqa: E402
from triage.data_io import load_split  # noqa: E402
from triage.features import FeatureSpec  # noqa: E402
from triage.taxonomy import MONETARY_RELIEF  # noqa: E402
from triage.tracking import setup_mlflow  # noqa: E402


class MlflowEpochLogger(tf.keras.callbacks.Callback):
    """Log every Keras epoch metric (loss, pr_auc, val_pr_auc, ...) to the active MLflow run."""

    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch)


def build_model(input_dim: int, hidden: list[int], dropout: float) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    x = inputs
    for i, units in enumerate(hidden):
        x = tf.keras.layers.Dense(units, activation="relu", name=f"dense_{i}")(x)
        x = tf.keras.layers.Dropout(dropout, name=f"dropout_{i}")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="p_relief")(x)
    model = tf.keras.Model(inputs, outputs, name="relief_predictor")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc"), tf.keras.metrics.AUC(name="roc_auc")],
    )
    return model


def main() -> None:
    r = CFG["relief"]
    processed_dir = path("processed_dir")
    train_df, test_df = load_split(processed_dir, "train"), load_split(processed_dir, "test")
    emb_train, emb_test = np.load(processed_dir / "emb_train.npy"), np.load(processed_dir / "emb_test.npy")

    spec = FeatureSpec.fit(train_df, embedding_dim=emb_train.shape[1])
    spec_path = processed_dir / "feature_spec.json"
    spec.save(spec_path)
    X_train, X_test = spec.transform(train_df, emb_train), spec.transform(test_df, emb_test)
    y_train = (train_df["company_response"] == MONETARY_RELIEF).to_numpy().astype(np.float32)
    y_test = (test_df["company_response"] == MONETARY_RELIEF).to_numpy().astype(np.float32)
    print(f"features: {X_train.shape[1]} dims | positives: train {y_train.mean():.3f}, test {y_test.mean():.3f}")

    weights = compute_class_weight("balanced", classes=np.array([0.0, 1.0]), y=y_train)
    class_weight = {0: float(weights[0]), 1: float(weights[1])}

    setup_mlflow()
    with mlflow.start_run(run_name="relief-tf-mlp") as run:
        mlflow.set_tags({"model_type": "relief", "framework": "tensorflow+keras"})
        mlflow.log_params({"hidden": str(r["hidden"]), "dropout": r["dropout"], "epochs": r["epochs"],
                           "batch_size": r["batch_size"], "class_weight_pos": class_weight[1],
                           "feature_dim": int(X_train.shape[1]), "threshold": r["threshold"]})

        model = build_model(X_train.shape[1], r["hidden"], r["dropout"])
        model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=r["epochs"],
            batch_size=r["batch_size"],
            class_weight=class_weight,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=5, restore_best_weights=True),
                MlflowEpochLogger(),
            ],
            verbose=2,
        )

        p = model.predict(X_test, verbose=0).reshape(-1)
        thr = float(r["threshold"])
        test_metrics = {
            "test_pr_auc": float(average_precision_score(y_test, p)),
            "test_roc_auc": float(roc_auc_score(y_test, p)) if len(np.unique(y_test)) > 1 else float("nan"),
            "test_precision_at_thr": float(precision_score(y_test, p >= thr, zero_division=0)),
            "test_recall_at_thr": float(recall_score(y_test, p >= thr, zero_division=0)),
            "test_flag_rate_at_thr": float((p >= thr).mean()),
        }
        mlflow.log_metrics(test_metrics)
        mlflow.log_artifact(str(spec_path))
        print(json.dumps(test_metrics, indent=2))

        signature = mlflow.models.infer_signature(X_test[:5], model.predict(X_test[:5], verbose=0))
        info = mlflow.tensorflow.log_model(
            model, name="model",
            registered_model_name=r["registered_name"],
            signature=signature,
            input_example=X_test[:2],
        )
        print(f"registered '{r['registered_name']}' version {info.registered_model_version} (run {run.info.run_id})")


if __name__ == "__main__":
    main()
