"""Optional pairwise cross-encoder scoring after candidate selection."""
import numpy as np
from src.retrieval import _validate_top_k


def rerank(model, query: str, candidates: list[dict], top_k: int) -> list[dict]:
    _validate_top_k(top_k)
    if not candidates or not query.strip():
        return []
    pairs = [(query, hit["text"]) for hit in candidates]
    scores = np.concatenate([batch for batch, _ in model.batches(pairs)])
    if scores.shape != (len(pairs), 1) or not np.isfinite(scores).all():
        raise ValueError("Invalid cross-encoder logits")
    hits = [hit | {"initial_rank": i, "fusion_score": hit["score"], "score": float(score[0])}
            for i, (hit, score) in enumerate(zip(candidates, scores), 1)]
    return sorted(hits, key=lambda hit: (-hit["score"], hit["id"]))[:top_k]


class Reranked:
    def __init__(self, initial, model, depth: int = 20):
        _validate_top_k(depth)
        self.initial, self.model, self.depth = initial, model, depth

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        return rerank(self.model, query, self.initial.search(query, self.depth), top_k)
