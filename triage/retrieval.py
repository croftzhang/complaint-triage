"""Embeddings (HuggingFace sentence-transformers via LangChain) and the FAISS
retriever over Regulation E chunks.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .config import CFG, path, resolve
from .reg_e import Chunk


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_embedder() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=resolve(CFG["embedding"]["model"]),
        model_kwargs={"device": _device()},
        encode_kwargs={"normalize_embeddings": True, "batch_size": CFG["embedding"]["batch_size"]},
    )


def embed_texts(texts: list[str], show_progress: bool = False) -> np.ndarray:
    emb = get_embedder()
    emb.show_progress = show_progress
    return np.asarray(emb.embed_documents(texts), dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    return np.asarray(get_embedder().embed_query(text), dtype=np.float32)


def chunks_to_documents(chunks: list[Chunk]) -> list[Document]:
    return [
        Document(page_content=c.text, metadata={"id": c.id, "section": c.section, "heading": c.heading, "chunk_idx": c.chunk_idx})
        for c in chunks
    ]


def build_index(chunks: list[Chunk]) -> FAISS:
    store = FAISS.from_documents(chunks_to_documents(chunks), get_embedder())
    store.save_local(str(path("index_dir")))
    return store


@lru_cache(maxsize=1)
def load_index() -> FAISS:
    index_dir = Path(CFG["paths"]["index_dir"])
    if not (index_dir / "index.faiss").exists():
        raise FileNotFoundError(f"no FAISS index in {index_dir}; run scripts/03_embed_and_index.py first")
    return FAISS.load_local(str(index_dir), get_embedder(), allow_dangerous_deserialization=True)


def retrieve(query: str, k: int | None = None) -> list[Document]:
    retriever = load_index().as_retriever(search_kwargs={"k": k or CFG["retrieval"]["k"]})
    return retriever.invoke(query)
