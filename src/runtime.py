"""Application retrieval entry point for the pinned documentation corpus."""
from hashlib import sha256
from pathlib import Path
from src.dense import Dense
from src.hybrid import Hybrid
from src.neural import Transformer
from src.rerank import Reranked
from src.retrieval import BM25

CHUNKS = Path("data/processed/corpus-v1/chunks.json")
VECTORS = Path("models/corpus-v1.npz")
CHUNK_HASH = "cee1cbdfc77457a77a6a1741bd72937f8095c511f0b4c4208616367e7dcfa882"
METHODS = ("hybrid", "bm25", "dense", "reranked")


def load_corpus() -> BM25:
    payload = CHUNKS.read_bytes()
    if sha256(payload).hexdigest() != CHUNK_HASH:
        raise ValueError("Unexpected chunk artifact")
    return BM25.from_bytes(payload)


def retriever(method: str):
    if method not in METHODS:
        raise ValueError("Unknown retrieval method")
    corpus = load_corpus()
    if method == "bm25":
        return corpus
    dense = Dense(corpus, Transformer("encoder"), VECTORS)
    if method == "dense":
        return dense
    hybrid = Hybrid(corpus, dense)
    if method == "hybrid":
        return hybrid
    return Reranked(hybrid, Transformer("reranker"))
