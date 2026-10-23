"""Text embeddings and hybrid (semantic + lexical) nearest-neighbour matching.

We use a small static embedding model (model2vec "potion") because it runs on CPU in seconds and was
good enough in earlier Project Iceberg work. The matcher blends cosine similarity on embeddings with
TF-IDF cosine similarity, which guards against the two classic failure modes: embeddings that match
on vibe but miss the key noun, and TF-IDF that misses paraphrases.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import CACHE, EMBED_MODEL

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from model2vec import StaticModel

        _MODEL = StaticModel.from_pretrained(EMBED_MODEL)
    return _MODEL


def _key(texts: list[str]) -> str:
    h = hashlib.sha1()
    h.update(EMBED_MODEL.encode())
    for t in texts:
        h.update(t.encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()[:20]


def encode(texts: list[str], cache: bool = True) -> np.ndarray:
    """L2-normalised embeddings, cached on disk by content hash."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path: Path = CACHE / f"emb_{_key(texts)}.npy"
    if cache and path.exists():
        return np.load(path)
    emb = _model().encode(list(texts), show_progress_bar=False).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.clip(norms, 1e-8, None)
    if cache:
        np.save(path, emb)
    return emb


class HybridMatcher:
    """Index a corpus once, then query it with top-k hybrid similarity.

    score = alpha * cosine(embedding) + (1 - alpha) * cosine(tfidf)
    """

    def __init__(self, corpus: list[str], alpha: float = 0.6, ngram=(1, 2), min_df: int = 1):
        self.corpus = list(corpus)
        self.alpha = alpha
        self.emb = encode(self.corpus)
        self.vec = TfidfVectorizer(ngram_range=ngram, min_df=min_df, sublinear_tf=True, stop_words="english")
        self.tfidf = self.vec.fit_transform(self.corpus)  # rows are L2-normalised by default

    def query(self, queries: list[str], k: int = 5, chunk: int = 2000, candidate_mask=None):
        """Return (indices [n,k], scores [n,k]). candidate_mask: optional callable(i) -> boolean array over
        the corpus restricting candidates for query i (e.g. only tasks of mapped occupations)."""
        queries = list(queries)
        q_emb = encode(queries)
        q_tfidf = self.vec.transform(queries)
        n = len(queries)
        idx = np.zeros((n, k), dtype=np.int64)
        sc = np.zeros((n, k), dtype=np.float32)
        for s in range(0, n, chunk):
            e = s + min(chunk, n - s)
            sim = self.alpha * (q_emb[s:e] @ self.emb.T) + (1 - self.alpha) * (q_tfidf[s:e] @ self.tfidf.T).toarray()
            if candidate_mask is not None:
                for i in range(s, e):
                    m = candidate_mask(i)
                    if m is not None:
                        sim[i - s, ~m] = -1.0
            top = np.argpartition(-sim, kth=min(k, sim.shape[1] - 1), axis=1)[:, :k]
            top_scores = np.take_along_axis(sim, top, axis=1)
            order = np.argsort(-top_scores, axis=1)
            idx[s:e] = np.take_along_axis(top, order, axis=1)
            sc[s:e] = np.take_along_axis(top_scores, order, axis=1)
        return idx, sc
