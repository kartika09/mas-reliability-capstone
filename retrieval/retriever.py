"""
Turns the SciFact-style dataset (dataset.load_dataset(...)["corpus"]) into
something Evidence/Critical can actually search: an in-memory embedding
index over every abstract, plus a retrieve(claim) function that returns the
top-k most relevant passages for a claim.

No vector database needed at this scale (a few thousand abstracts fit
easily in memory) -- just numpy + sentence-transformers.

CONTRACT:
  build_index(corpus) -> Index
  retrieve(index, claim, k=5) -> list[dict]   (each dict is a corpus passage,
                                                with a "_score" field added)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None  # lazy-loaded, so importing this module doesn't need the model or network


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


@dataclass
class Index:
    passages: list[dict]     # corpus rows, same order as `embeddings`
    embeddings: np.ndarray   # shape (n_passages, dim), L2-normalized


def build_index(corpus: list[dict]) -> Index:
    """corpus: the ["corpus"] list from dataset.load_dataset(...) -- each
    item needs at least "passage_id" and "text"."""
    model = _get_model()
    texts = [p["text"] for p in corpus]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return Index(passages=corpus, embeddings=np.asarray(embeddings))


def retrieve(index: Index, claim: str, k: int = 5) -> list[dict]:
    """Return the top-k passages most similar to `claim`, each with a
    similarity score attached under "_score" (1.0 = identical, -1.0 = opposite)."""
    model = _get_model()
    query_vec = model.encode([claim], normalize_embeddings=True)[0]
    scores = index.embeddings @ query_vec   # cosine similarity (both sides already normalized)
    top_idx = np.argsort(-scores)[:k]
    results = []
    for i in top_idx:
        passage = dict(index.passages[i])
        passage["_score"] = float(scores[i])
        results.append(passage)
    return results