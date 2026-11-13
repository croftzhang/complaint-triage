"""Feature engineering shared by TF training (TF env) and inference (main env).

Only numpy / pandas here: this module must import cleanly in both environments.

Feature vector = [ narrative embedding | one-hot(product_group) | one-hot(submitted_via)
                   | one-hot(tags) | log1p(narrative length) / 10 ]
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORICAL_COLUMNS = ["product_group", "submitted_via", "tags"]
UNKNOWN = "__unknown__"


def normalise_tags(value) -> str:
    """CFPB tags are None / 'Older American' / 'Servicemember' / 'Older American, Servicemember'."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == "":
        return "None"
    return str(value).strip()


@dataclass
class FeatureSpec:
    embedding_dim: int
    vocab: dict[str, list[str]] = field(default_factory=dict)  # column -> ordered categories

    # ---------- fit / persist ----------
    @classmethod
    def fit(cls, df: pd.DataFrame, embedding_dim: int) -> "FeatureSpec":
        vocab = {}
        for col in CATEGORICAL_COLUMNS:
            values = df[col].map(normalise_tags) if col == "tags" else df[col].fillna(UNKNOWN).astype(str)
            vocab[col] = sorted(values.unique().tolist())
        return cls(embedding_dim=embedding_dim, vocab=vocab)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"embedding_dim": self.embedding_dim, "vocab": self.vocab}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "FeatureSpec":
        d = json.loads(Path(path).read_text())
        return cls(embedding_dim=d["embedding_dim"], vocab=d["vocab"])

    # ---------- transform ----------
    @property
    def dim(self) -> int:
        return self.embedding_dim + sum(len(v) for v in self.vocab.values()) + 1

    def _onehot(self, col: str, value) -> np.ndarray:
        cats = self.vocab[col]
        vec = np.zeros(len(cats), dtype=np.float32)
        value = normalise_tags(value) if col == "tags" else (UNKNOWN if value is None else str(value))
        if value in cats:
            vec[cats.index(value)] = 1.0
        return vec

    def transform_one(self, embedding: np.ndarray, product_group: str, submitted_via: str, tags, narrative: str) -> np.ndarray:
        parts = [
            np.asarray(embedding, dtype=np.float32).reshape(-1),
            self._onehot("product_group", product_group),
            self._onehot("submitted_via", submitted_via),
            self._onehot("tags", tags),
            np.array([np.log1p(len(narrative or "")) / 10.0], dtype=np.float32),
        ]
        vec = np.concatenate(parts)
        assert vec.shape[0] == self.dim, f"feature dim mismatch {vec.shape[0]} != {self.dim}"
        return vec

    def transform(self, df: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray:
        assert len(df) == len(embeddings), "rows and embeddings must align"
        rows = [
            self.transform_one(embeddings[i], r.product_group, r.submitted_via, r.tags, r.narrative)
            for i, r in enumerate(df.itertuples(index=False))
        ]
        return np.vstack(rows)
