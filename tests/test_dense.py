from types import SimpleNamespace

import numpy as np
import pytest

from src.dense import Dense, build_embeddings, encode, normalized


class Encoder:
    def batches(self, texts):
        yield np.ones((len(texts), 2, 384), dtype=np.float32), np.ones((len(texts), 2))


@pytest.mark.parametrize("vectors", [np.zeros((1, 384)), np.empty((0, 384)), np.ones((1, 383)),
                                     np.full((1, 384), np.nan), np.full((1, 384), np.inf)])
def test_invalid_embeddings(vectors):
    with pytest.raises(ValueError):
        normalized(vectors, 1)


def test_cache_and_ties(tmp_path):
    corpus = SimpleNamespace(corpus_revision="one", artifact_sha256="abc",
                             chunks=[{"id": k, "text": "café"} for k in ("z", "a")])
    path = tmp_path / "index.npz"
    model = Encoder()
    with pytest.raises(FileNotFoundError):
        Dense(corpus, model, path)
    build_embeddings(corpus, model, path)
    dense = Dense(corpus, model, path)
    assert dense.search(" ") == []
    assert [r["id"] for r in dense.search("新")] == ["a", "z"]
    corpus.artifact_sha256 = "changed"
    with pytest.raises(ValueError, match="identity"):
        Dense(corpus, model, path)
    with pytest.raises(ValueError):
        encode(model, [])
