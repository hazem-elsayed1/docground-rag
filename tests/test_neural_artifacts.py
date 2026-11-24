from types import SimpleNamespace
import json
import numpy as np
import pytest
from src.neural import Transformer
from src.dense import Dense, build_embeddings
from src.rerank import rerank


def test_unavailable_or_corrupt_model_is_rejected_before_inference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        Transformer("reranker")
    path = tmp_path / "models/reranker/tokenizer.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        Transformer("reranker")


def test_cache_rejects_engine_revision_and_changed_vector_bytes(tmp_path):
    class Encoder:
        def batches(self, texts):
            yield np.ones((1, 1, 384), dtype=np.float32), np.ones((1, 1))
    corpus = SimpleNamespace(chunks=[{"text": "sample"}], corpus_revision="corpus", artifact_sha256="hash")
    path = tmp_path / "cache.npz"
    build_embeddings(corpus, Encoder(), path)
    with np.load(path) as data:
        vectors = data["vectors"]
        meta = json.loads(str(data["metadata"]))
    for values, metadata in [(vectors, meta | {"revision": "other"}), (-vectors, meta)]:
        np.savez(path, vectors=values, metadata=json.dumps(metadata))
        with pytest.raises(ValueError, match="identity"):
            Dense(corpus, Encoder(), path)


def test_later_reranker_batch_failure_does_not_return_partial_answers():
    class Broken:
        def batches(self, pairs):
            yield np.array([[1.]]), None
            raise RuntimeError("backend unavailable")
    candidates = [{"id": i, "text": "some words", "score": 1} for i in ("x", "y")]
    with pytest.raises(RuntimeError, match="backend unavailable"):
        rerank(Broken(), "q", candidates, 2)
