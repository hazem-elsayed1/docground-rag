"""Reciprocal rank fusion of independently ranked chunk lists."""
from src.retrieval import _validate_top_k


def fuse(rankings: dict[str, list[dict]], top_k: int, constant: int = 60) -> list[dict]:
    _validate_top_k(top_k)
    _validate_top_k(constant)
    candidates = {}
    for method, ranking in rankings.items():
        seen = set()
        for hit in ranking:
            key = hit["id"]
            if key in seen:
                continue
            seen.add(key)
            rank = len(seen)
            result = candidates.setdefault(key, hit | {"score": 0.0, "components": {}})
            result["components"][method] = {"rank": rank, "score": hit["score"]}
            result["score"] += 1 / (constant + rank)
    return sorted(candidates.values(), key=lambda hit: (-hit["score"], hit["id"]))[:top_k]


class Hybrid:
    def __init__(self, sparse, dense, depth: int = 20):
        _validate_top_k(depth)
        self.sparse, self.dense, self.depth = sparse, dense, depth

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        return fuse({"bm25": self.sparse.search(query, self.depth),
                     "dense": self.dense.search(query, self.depth)}, top_k)
