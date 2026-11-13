"""Thin MLflow helpers shared by training, evaluation and serving.

Keep this importable from both environments (mlflow only, no torch/tf).
"""
from __future__ import annotations

import os

import mlflow
from mlflow import MlflowClient

from .config import CFG

CHAMPION = "champion"


def setup_mlflow(experiment: str | None = None) -> None:
    uri = os.environ.get("MLFLOW_TRACKING_URI", CFG["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment or CFG["mlflow"]["experiment"])


def model_uri(registered_name: str) -> str:
    """models:/<name>@champion if the alias exists, otherwise the newest version."""
    client = MlflowClient()
    try:
        client.get_model_version_by_alias(registered_name, CHAMPION)
        return f"models:/{registered_name}@{CHAMPION}"
    except Exception:
        versions = client.search_model_versions(f"name='{registered_name}'")
        if not versions:
            raise RuntimeError(f"no registered model named '{registered_name}' - train it first")
        latest = max(versions, key=lambda v: int(v.version))
        return f"models:/{registered_name}/{latest.version}"


def set_champion(registered_name: str, version: str) -> None:
    MlflowClient().set_registered_model_alias(registered_name, CHAMPION, version)
    print(f"alias '{CHAMPION}' -> {registered_name} v{version}")


def versions_with_metrics(registered_name: str, metric_keys: list[str]) -> list[dict]:
    """Every registered version with the selected metrics of its training run."""
    client = MlflowClient()
    rows = []
    for v in client.search_model_versions(f"name='{registered_name}'"):
        metrics = client.get_run(v.run_id).data.metrics if v.run_id else {}
        rows.append({"version": v.version, "run_id": v.run_id, **{k: metrics.get(k) for k in metric_keys}})
    return sorted(rows, key=lambda r: int(r["version"]))
