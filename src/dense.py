"""Exact cosine retrieval and a corpus-bound embedding cache."""
import json
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from src.neural import MODELS, Transformer
from src.retrieval import BM25, _validate_top_k


def normalized(matrix: np.ndarray, rows: int) -> np.ndarray:
    if matrix.shape != (rows, 384) or rows < 1 or not np.isfinite(matrix).all():
        raise ValueError("Invalid embedding shape or values")
    lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
    if (lengths <= 0).any() or not np.isfinite(lengths).all():
        raise ValueError("Invalid embedding length")
    return matrix / lengths


def encode(model: Transformer, texts: list[str]) -> np.ndarray:
    if not texts or any(not text.strip() for text in texts):
        raise ValueError("Expected nonempty texts")
    pooled = []
    for tokens, mask in model.batches(texts):
        weights = mask[:, :, None].astype(np.float32)
        pooled.append((tokens * weights).sum(axis=1) / weights.sum(axis=1))
    return normalized(np.concatenate(pooled), len(texts))


def configuration(corpus: BM25) -> dict:
    return {"corpus_revision": corpus.corpus_revision, "chunk_sha256": corpus.artifact_sha256,
            "model": MODELS["encoder"][0], "revision": MODELS["encoder"][1],
            "weights_sha256": MODELS["encoder"][2], "tokenizer_sha256": MODELS["encoder"][3],
            "dimensions": 384, "max_tokens": 256, "pooling": "masked-mean", "normalization": "L2",
            "input": "chunk text in artifact order", "batch_size": 8, "backend": "onnx-cpu-fp32",
            "numpy": np.__version__, "onnxruntime": __import__("onnxruntime").__version__,
            "tokenizers": __import__("tokenizers").__version__}


def build_embeddings(corpus: BM25, model: Transformer, path: Path) -> None:
    vectors = encode(model, [chunk["text"] for chunk in corpus.chunks])
    metadata = configuration(corpus) | {"vectors_sha256": sha256(vectors.tobytes()).hexdigest()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        np.savez(temp, vectors=vectors, metadata=json.dumps(metadata, sort_keys=True))
    Path(temp.name).replace(path)


class Dense:
    def __init__(self, corpus: BM25, model: Transformer, path: Path):
        self.corpus, self.model = corpus, model
        with np.load(path, allow_pickle=False) as artifact:
            self.vectors = artifact["vectors"]
            metadata = json.loads(str(artifact["metadata"]))
        expected = configuration(corpus) | {"vectors_sha256": sha256(self.vectors.tobytes()).hexdigest()}
        if metadata != expected or self.vectors.dtype != np.float32:
            raise ValueError("Embedding cache identity mismatch")
        if not np.allclose(normalized(self.vectors, len(corpus.chunks)), self.vectors, atol=1e-6):
            raise ValueError("Expected normalized corpus vectors")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        _validate_top_k(top_k)
        if not query.strip():
            return []
        return self.rank_vector(encode(self.model, [query])[0], top_k)

    def rank_vector(self, vector: np.ndarray, top_k: int) -> list[dict]:
        _validate_top_k(top_k)
        vector = normalized(np.asarray(vector)[None, :], 1)[0]
        scores = self.vectors @ vector
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self.corpus.chunks[i]["id"]))
        return [self.corpus.chunks[i] | {"score": float(scores[i])} for i in order[:top_k]]
