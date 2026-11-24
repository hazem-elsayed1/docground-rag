import json
import subprocess
import sys

import pytest

from src.chunking import chunk_document, evidence_coverage
from src.ingestion import ingest_document


@pytest.fixture
def document(tmp_path):
    source = tmp_path / "guide.html"
    source.write_bytes(
        ("<article><h1 id='intro'>Guide</h1><p>é🙂</p><p>abc</p>"
         "<h2 id='usage'>Usage</h2><details id='demo'><summary>Run</summary>"
         "<pre>if x:\r\n    go()\r\n</pre></details>"
         "<ul><li>one</li></ul><table><caption>Scores</caption><tr><td>two</td></tr></table>"
         "<h2>Index</h2></article>").encode("utf-8")
    )
    return ingest_document(source, source_url="https://example.org/guide")


def test_cli_reproduces_identical_bytes_and_ids(document, tmp_path):
    source = tmp_path / "guide.json"
    source.write_text(json.dumps(document), encoding="utf-8")
    output = tmp_path / "guide.chunks.json"
    command = [sys.executable, "-m", "src.chunking", str(source), str(output)]
    subprocess.run(command, check=True)
    first = output.read_bytes()
    subprocess.run(command, check=True)
    assert output.read_bytes() == first
    artifact = json.loads(first)
    assert artifact == chunk_document(document)
    assert artifact["max_chars"] == 1200
    assert len({chunk["id"] for chunk in artifact["chunks"]}) == len(artifact["chunks"])


def test_nested_fields_round_trip_to_source_and_canonical_offsets(document):
    chunks = chunk_document(document, 1200)["chunks"]
    intro, usage = chunks
    source = document["source"]
    code = "if x:\r\n    go()\r\n"
    canonical = "\n\n".join(["é🙂", "abc", "Run", code, "one", "Scores", "two"])
    assert intro["text"] == "é🙂\n\nabc"
    assert usage["anchor"] == "usage"
    assert usage["heading_path"] == ["Guide", "Usage"]
    assert [field["anchor"] for field in usage["fields"]] == ["demo", "demo", "usage", "usage", "usage"]
    for chunk, section in zip(chunks, document["sections"]):
        assert chunk["section_id"] == section["id"]
        assert chunk["source_sha256"] == source["sha256"]
        assert chunk["source_url"] == source["url"]
        assert chunk["text"] == canonical[chunk["start"]:chunk["end"]]
        for field in chunk["fields"]:
            original = section
            for key in field["path"].strip("/").split("/"):
                original = original[int(key)] if isinstance(original, list) else original[key]
            assert original == canonical[field["start"]:field["end"]]


def test_exact_limit_separator_and_oversized_fields(document):
    exact = chunk_document(document, 7)["chunks"]
    smaller = chunk_document(document, 6)["chunks"]
    assert exact[0]["text"] == "é🙂\n\nabc"
    assert [c["text"] for c in smaller[:2]] == ["é🙂", "abc"]
    assert [(c["start"], c["end"]) for c in smaller[:2]] == [(0, 2), (4, 7)]
    assert [c["text"] for c in smaller if len(c["text"]) > 6] == ["if x:\r\n    go()\r\n"]
    assert [f for c in exact for f in c["fields"]] == [f for c in smaller for f in c["fields"]]
    exact_code = next(c for c in exact if c["text"].startswith("if x:"))
    smaller_code = next(c for c in smaller if c["text"].startswith("if x:"))
    assert exact_code["id"] == smaller_code["id"]


def test_evidence_requires_full_containment_and_all_required_units(document):
    wide = chunk_document(document)["chunks"]
    small = chunk_document(document, 3)["chunks"]
    first, second = wide
    first_unit = {k: first[k] for k in ("document_id", "section_id", "start", "end")}
    panel = second["fields"][1]
    second_unit = {
        "document_id": second["document_id"], "section_id": second["section_id"],
        "start": panel["start"] + 2, "end": panel["end"],
    }
    required = [first_unit, second_unit]
    assert evidence_coverage(required, wide) == [True, True]
    assert evidence_coverage(required, [first, first]) == [True, False]
    assert evidence_coverage(required, small) == [False, True]
    assert evidence_coverage([first_unit], []) == [False]
    misses = [
        first_unit | {"document_id": "other"},
        first_unit | {"section_id": second["section_id"]},
        first_unit | {"end": first["end"] + 1},
        first_unit | {"start": 2, "end": 4},  # Only the inserted separator.
    ]
    assert evidence_coverage(misses, wide) == [False] * len(misses)
    with pytest.raises(ValueError, match="range"):
        evidence_coverage([first_unit | {"start": first_unit["end"]}], wide)
    with pytest.raises(ValueError, match="evidence unit"):
        evidence_coverage([], wide)


@pytest.mark.parametrize(("change", "message"), [
    ({"sections": []}, "no text"),
    ({"schema_version": 2}, "version"),
    ({"sections": [None]}, "id"),
    ({"source": {"url": "https://example.org", "sha256": "unknown"}}, "SHA-256"),
])
def test_invalid_processed_documents_fail(document, change, message):
    with pytest.raises(ValueError, match=message):
        chunk_document(document | change)


def test_duplicate_sections_fail(document):
    section = document["sections"][1]
    document["sections"].append(section)
    with pytest.raises(ValueError, match="duplicate section"):
        chunk_document(document)


def test_unknown_nested_blocks_fail(document):
    section = document["sections"][1]
    section["blocks"][0]["blocks"][0]["type"] = "unknown"
    with pytest.raises(ValueError, match="Unknown block type.*blocks/0/blocks/0"):
        chunk_document(document)


def test_nonpositive_limit_fails(document):
    with pytest.raises(ValueError, match="positive integer"):
        chunk_document(document, 0)


def test_malformed_json_keeps_previous_output(tmp_path):
    source = tmp_path / "guide.json"
    output = tmp_path / "chunks.json"
    source.write_bytes(b'{"sections":')
    output.write_bytes(b"previous artifact")
    result = subprocess.run(
        [sys.executable, "-m", "src.chunking", str(source), str(output)], capture_output=True
    )
    assert result.returncode == 1
    assert b"chunking:" in result.stderr
    assert output.read_bytes() == b"previous artifact"
