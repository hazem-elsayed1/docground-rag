from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from experiments.minilm_windows import token_windows, window_vectors
from src.dense import encode
from src.neural import Transformer


@pytest.fixture
def tokenizer():
    vocab = {s: i for i, s in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]"] + [f"t{i}" for i in range(700)])}
    token = Tokenizer(WordPiece(vocab, unk_token="[UNK]"))
    token.pre_tokenizer = Whitespace()
    token.post_processor = TemplateProcessing(single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])])
    token.enable_truncation(256)
    token.enable_padding()
    return token


@pytest.mark.parametrize(("length", "starts"), [(1, [0]), (254, [0]), (255, [0, 222]), (640, [0, 222, 444])])
def test_overflows_preserve_original_tokens_with_exact_coverage_and_special_tokens(tokenizer, length, starts):
    text = " ".join(f"t{i}" for i in range(length))
    original = Tokenizer.from_str(tokenizer.to_str())
    original.no_truncation()
    full_ids = original.encode(text, add_special_tokens=False).ids
    windows, = token_windows(tokenizer, [text], 256, 32)
    assert [row.ids[1:-1] for row in windows] == [full_ids[start:start + 254] for start in starts]
    assert set(range(length)) == {i for start in starts for i in range(start, min(start + 254, length))}
    assert all(len(row.ids) <= 256 and row.ids[0] == 2 and row.ids[-1] == 3 for row in windows)
    repeated, = token_windows(tokenizer, [text], 256, 32)
    assert [row.ids for row in repeated] == [row.ids for row in windows]
    assert tokenizer.truncation["stride"] == 0


class TinySession:
    """Deterministic stand-in for ONNX inference, with the real tokenizer and pooling."""

    def get_inputs(self):
        return [SimpleNamespace(name=key) for key in ("input_ids", "attention_mask", "token_type_ids")]

    def run(self, _, feed):
        ids = feed["input_ids"]
        return [np.eye(384, dtype=np.float32)[ids % 384]]


def test_window_builder_preserves_row_order_id_and_prefix_without_extra_metadata(tokenizer):
    backend = Transformer.__new__(Transformer)
    backend.tokenizer, backend.session = tokenizer, TinySession()
    long = " ".join(f"t{i}" for i in range(300))
    corpus = SimpleNamespace(chunks=[
        {"id": "z", "text": "t610 t2", "heading_path": ["t699"]},
        {"id": "m", "text": long, "heading_path": ["t698"]},
        {"id": "a", "text": "t9 t10", "heading_path": ["t697"]},
    ])
    config = {"max_tokens": 256, "overlap_content_tokens": 32}
    before = encode(backend, [chunk["text"] for chunk in corpus.chunks])
    first, counts, bytes_used = window_vectors(corpus, backend, config)
    assert counts == [1, 2, 1] and bytes_used == 4 * 384 * 4
    assert first.shape == (3, 384) and first.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(first[[0, 2]], before[[0, 2]], rtol=1e-6)
    assert first[1, tokenizer.token_to_id("t299")] > 0 and before[1, tokenizer.token_to_id("t299")] == 0
    assert [chunk["id"] for chunk in corpus.chunks] == ["z", "m", "a"]
    np.testing.assert_array_equal(first, window_vectors(corpus, backend, config)[0])
    np.testing.assert_array_equal(before, encode(backend, [chunk["text"] for chunk in corpus.chunks]))
