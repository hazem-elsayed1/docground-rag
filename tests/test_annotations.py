import json
import subprocess
import sys

import pytest

from src.annotations import EvidenceDocument


@pytest.fixture
def document():
    return {
        "schema_version": 1,
        "source": {"url": "https://example.org/guide", "sha256": "a" * 64},
        "sections": [
            {
                "id": "usage", "heading_path": ["Guide", "Usage"], "anchors": ["example"],
                "blocks": [
                    {"type": "paragraph", "text": "\U0001f40d fit fit"},
                    {"type": "details", "anchor": "panel", "summary": "Example",
                     "blocks": [{"type": "code", "text": "if ready:\r\n    fit()"}]},
                ],
            },
            {
                "id": "other", "heading_path": ["Guide", "Other"], "anchors": [],
                "blocks": [{"type": "paragraph", "text": "fit"}],
            },
        ],
    }


def test_literal_search_preserves_unicode_offsets_and_returns_each_occurrence(document):
    corpus = EvidenceDocument(document)
    matches = list(corpus.find("fit"))
    assert [(hit["section_id"], hit["start"], hit["end"]) for hit in matches] == [
        ("usage", 2, 5), ("usage", 6, 9), ("usage", 35, 38), ("other", 42, 45),
    ]
    assert list(corpus.find("fit", "example")) == matches[:3]
    assert list(corpus.find("Fit")) == []


def test_cross_field_quote_preserves_separators_and_code_but_cannot_cross_sections(document):
    corpus = EvidenceDocument(document)
    quote = "fit\n\nExample\n\nif ready:\r\n    fit()"
    [span] = corpus.find(quote, "usage")
    assert span == {
        "section_id": "usage", "start_block_path": "/blocks/0/text",
        "end_block_path": "/blocks/1/blocks/0/text", "start": 6, "end": 40,
    }
    assert corpus.text[span["start"]:span["end"]] == quote
    assert list(corpus.find("fit()\n\nfit")) == []
    assert list(corpus.find("\n\nExample")) == []


def run_cli(document, *args):
    return subprocess.run(
        [sys.executable, "-m", "src.annotations", str(document), *map(str, args)],
        capture_output=True, encoding="utf-8",
    )


@pytest.fixture
def draft_files(document, tmp_path):
    document_path = tmp_path / "processed.json"
    document_path.write_text(json.dumps(document), encoding="utf-8")
    corpus = EvidenceDocument(document)
    first, second, _, other = corpus.find("fit")
    [code] = corpus.find("if ready:\r\n    fit()")
    draft = {
        "annotation_schema_version": 1, "annotation_version": "synthetic-1",
        "corpus_revision": corpus.revision, "split": "development",
        "questions": [
            {"id": "fixture-1", "question": "Synthetic fixture", "answerable": True,
             "evidence_units": [[first, second, other], [code]]},
            {"id": "fixture-2", "question": "Synthetic unanswerable fixture", "answerable": False,
             "evidence_units": []},
        ],
    }
    draft_path = tmp_path / "draft.json"
    draft_path.write_text(json.dumps(draft), encoding="utf-8")
    return document_path, draft_path, tmp_path / "reviewed.json"


def test_cli_browses_fields_reviews_alternatives_and_exports_without_overwriting(draft_files):
    document, draft, output = draft_files
    browse = run_cli(document, "browse", "--section", "example")
    assert browse.returncode == 0, browse.stderr
    assert "Guide > Usage" in browse.stdout
    assert '/blocks/1/blocks/0/text [20, 40) (field anchor ' in browse.stdout
    assert '"if ready:\\r\\n    fit()"' in browse.stdout
    found = run_cli(document, "find", "fit", "--section", "other")
    assert found.returncode == 0, found.stderr
    assert '"start": 42, "end": 45' in found.stdout
    review = run_cli(document, "review", draft)
    assert review.returncode == 0, review.stderr
    assert "Required unit 1, alternative 3" in review.stdout
    assert "Required unit 2, alternative 1" in review.stdout
    assert '"if ready:\\r\\n    fit()"' in review.stdout
    assert "fixture-2 (unanswerable)" in review.stdout
    assert not output.exists()
    export = run_cli(document, "export", draft, output)
    assert export.returncode == 0, export.stderr
    assert json.loads(output.read_bytes()) == json.loads(draft.read_bytes())
    original = output.read_bytes()
    retry = run_cli(document, "export", draft, output)
    assert retry.returncode == 1
    assert output.read_bytes() == original


@pytest.mark.parametrize(("key", "value", "message"), [
    ("corpus_revision", "different-document", "corpus_revision"),
    ("start", 9, "does not match start_block_path"),
    ("end", 42, "does not match end_block_path"),
    ("end_block_path", "/blocks/99/text", "does not match end_block_path"),
])
def test_export_rejects_invalid_draft_without_writing(draft_files, key, value, message):
    document, draft_path, output = draft_files
    draft = json.loads(draft_path.read_bytes())
    record = draft if key == "corpus_revision" else draft["questions"][0]["evidence_units"][1][0]
    record[key] = value
    draft_path.write_text(json.dumps(draft), encoding="utf-8")
    result = run_cli(document, "export", draft_path, output)
    assert result.returncode == 1
    assert message in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(("quote", "section", "message"), [
    (" ", None, "non-whitespace"),
    ("fit", "unknown", "Unknown section"),
])
def test_find_rejects_invalid_selection(document, quote, section, message):
    with pytest.raises(ValueError, match=message):
        list(EvidenceDocument(document).find(quote, section))
