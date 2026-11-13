#!/usr/bin/env python
"""Step 3 - Embeddings with a HuggingFace sentence-transformers model (GPU if available).

  1. Build the FAISS index over Regulation E chunks (used by the LangGraph retrieve node).
  2. Embed every complaint narrative in train/test and save .npy files
     (used as input features by the TensorFlow relief model in step 5).

  python scripts/03_embed_and_index.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from triage.config import CFG, path  # noqa: E402
from triage.data_io import load_split  # noqa: E402
from triage.reg_e import load_chunks  # noqa: E402


def main() -> None:
    processed_dir = path("processed_dir")
    train, test = load_split(processed_dir, "train"), load_split(processed_dir, "test")

    if CFG.get("smoke", {}).get("tiny_models"):
        from triage.smoke import ensure_tiny_models
        ensure_tiny_models(CFG, train)

    # Imported after the tiny models exist (get_embedder is cached).
    from triage.retrieval import build_index, embed_texts, get_embedder

    t0 = time.time()
    print(f"embedding model: {CFG['embedding']['model']} on {get_embedder().model_kwargs.get('device')}")

    chunks = load_chunks(processed_dir / "reg_e_chunks.jsonl")
    build_index(chunks)
    print(f"FAISS index: {len(chunks)} Reg E chunks -> {CFG['paths']['index_dir']}")

    for name, df in (("train", train), ("test", test)):
        emb = embed_texts(df["narrative"].tolist(), show_progress=True)
        np.save(processed_dir / f"emb_{name}.npy", emb)
        print(f"{name}: {emb.shape} -> emb_{name}.npy")

    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
