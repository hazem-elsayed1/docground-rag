import hashlib
import json
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from scripts.audit_embedding_truncation import measure
from scripts.compare_retrieval import config_digest
from src.hybrid import fuse


def test_config_digest_is_reproducible_from_visible_json_object():
    canonical = '{"a":"é","z":[1,2]}'.encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    assert config_digest({"z": (1, 2), "a": "é"}) == digest
    assert config_digest(json.loads(canonical)) == digest
    with pytest.raises(ValueError):
        config_digest({"nonfinite": float("nan")})


def test_rrf_cutoff_uses_chunk_id_for_equal_scores():
    a, z = ({"id": name, "score": 1.0} for name in ("a", "z"))
    assert [row["id"] for row in fuse({"lexical": [z], "dense": [a]}, 1)] == ["a"]
    assert [row["id"] for row in fuse({"dense": [a], "lexical": [z]}, 2)] == ["a", "z"]


def test_encoder_audit_detects_omitted_evidence_using_wordpiece_offsets():
    vocab = {token: i for i, token in enumerate(("[UNK]", "[CLS]", "[SEP]", "ab", "##cd", "word"))}
    tokenizer = Tokenizer(WordPiece(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])])
    tokenizer.enable_truncation(4)
    chunk = {"text": "abcd word", "id": "a", "document_id": "v", "section_id": "s", "start": 10, "end": 19}
    kept = {"document_id": "v", "section_id": "s", "start": 10, "end": 14}
    removed = kept | {"start": 15, "end": 19}
    row, = measure([chunk], [{"id": "q", "evidence_units": [[kept], [removed]]}], tokenizer)
    assert (row["token_count"], row["retained_token_count"], row["retained_text_end"]) == (5, 4, 14)
    assert [item["evidence_tokens_omitted"] for item in row["development_span_overlaps"]] == [False, True]
