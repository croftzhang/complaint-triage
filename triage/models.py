"""Clients for the two trained models, used by the LangGraph nodes.

Backends (config.yaml -> serving):
  classifier_backend: local  -> mlflow.pyfunc.load_model("models:/<name>@champion") in-process
                      http   -> POST to `mlflow models serve` (port 5001)
  relief_backend:     disabled -> node returns None (before the TF model is served)
                      http     -> POST to `mlflow models serve` run from the TF env (port 5002)
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import requests

from .config import CFG
from .tracking import model_uri, setup_mlflow


def _post(url: str, payload: dict) -> dict:
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


class ClassifierClient:
    def __init__(self) -> None:
        s = CFG["serving"]
        self.backend, self.url = s["classifier_backend"], s["classifier_url"]
        self._model = None

    def _load(self):
        if self._model is None:
            import mlflow

            setup_mlflow()
            uri = model_uri(CFG["classifier"]["registered_name"])
            print(f"[classifier] loading {uri}")
            self._model = mlflow.pyfunc.load_model(uri)
        return self._model

    @staticmethod
    def _parse(pred) -> tuple[str, float]:
        """Normalise the transformers text-classification output ({label, score})."""
        if hasattr(pred, "iloc"):          # pandas DataFrame
            row = pred.iloc[0]
            return str(row["label"]), float(row["score"])
        first = pred[0] if isinstance(pred, list) else pred
        if isinstance(first, list):        # [[{label, score}, ...]] when return_all_scores
            first = max(first, key=lambda d: d["score"])
        return str(first["label"]), float(first["score"])

    def predict(self, text: str) -> tuple[str, float]:
        if self.backend == "http":
            return self._parse(_post(self.url, {"inputs": [text]})["predictions"])
        return self._parse(self._load().predict([text]))


class ReliefClient:
    def __init__(self) -> None:
        s = CFG["serving"]
        self.backend, self.url = s["relief_backend"], s["relief_url"]

    def predict(self, features: np.ndarray) -> float | None:
        if self.backend == "disabled":
            return None
        out = _post(self.url, {"inputs": [np.asarray(features, dtype=np.float32).tolist()]})["predictions"]
        val = out[0]
        while isinstance(val, (list, tuple)):
            val = val[0]
        if isinstance(val, dict):
            val = next(iter(val.values()))
        return float(val)


@lru_cache(maxsize=1)
def classifier() -> ClassifierClient:
    return ClassifierClient()


@lru_cache(maxsize=1)
def relief() -> ReliefClient:
    return ReliefClient()
