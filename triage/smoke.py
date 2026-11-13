"""Build tiny, randomly initialised HuggingFace models for the offline smoke test.

These models learn nothing useful; they only let every script run end-to-end
on CPU in a couple of minutes without downloading pretrained weights.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def _build_vocab(texts: list[str], max_size: int = 4000) -> list[str]:
    counts: dict[str, int] = {}
    for t in texts:
        for w in re.findall(r"[a-z0-9]+|[^\sa-z0-9]", t.lower()):
            counts[w] = counts.get(w, 0) + 1
    words = sorted(counts, key=counts.get, reverse=True)[: max_size - 5]
    return ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words


def build_tiny_encoder(texts: list[str], out_dir: Path, dim: int = 64) -> Path:
    """Tiny DistilBERT base model + WordPiece-style tokenizer saved to out_dir."""
    from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizerFast

    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_file = out_dir / "vocab.txt"
    vocab_file.write_text("\n".join(_build_vocab(texts)), encoding="utf-8")
    tok = DistilBertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=True)
    tok.save_pretrained(out_dir)

    cfg = DistilBertConfig(
        vocab_size=len(tok), dim=dim, n_layers=2, n_heads=2, hidden_dim=dim * 2,
        max_position_embeddings=128,
    )
    DistilBertModel(cfg).save_pretrained(out_dir)
    return out_dir


def build_tiny_embedder(encoder_dir: Path, out_dir: Path) -> Path:
    """Wrap the tiny encoder as a sentence-transformers model (mean pooling)."""
    from sentence_transformers import SentenceTransformer, models

    word = models.Transformer(str(encoder_dir), max_seq_length=64)
    pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="mean")
    SentenceTransformer(modules=[word, pool]).save(str(out_dir))
    return out_dir


def ensure_tiny_models(cfg: dict, train_df: pd.DataFrame) -> None:
    """Create the models named in config.smoke.yaml if they do not exist yet."""
    from .config import REPO_ROOT

    enc = REPO_ROOT / cfg["classifier"]["base_model"]
    emb = REPO_ROOT / cfg["embedding"]["model"]
    if not (enc / "config.json").exists():
        build_tiny_encoder(train_df["narrative"].tolist(), enc)
        print(f"built tiny encoder -> {enc}")
    if not (emb / "modules.json").exists():
        build_tiny_embedder(enc, emb)
        print(f"built tiny embedder -> {emb}")
