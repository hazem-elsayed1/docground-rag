from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.ingestion import ingest_document


SOURCE_URL = "https://example.org/docs/example.html"


def test_article_extracts_heading_hierarchy_and_section_anchors(tmp_path: Path):
    source = tmp_path / "article.html"
    source.write_text(
        """
        <main>
          <article>
            <nav><script>Sidebar script</script><p>Sidebar</p></nav>
            <section id="root"><h1>Data Leakage</h1>
              <p>Fit <code>preprocessing</code> on training data only.</p>
              <section id="sub"><span id="alias"></span>
                <h3 id="heading-id">Café<a class="headerlink" href="#sub">#</a></h3>
                <ul><li>Outer <strong>text</strong><ol><li>Inner</li></ol></li></ul>
              </section>
              <p>Parent summary.</p>
            </section>
          </article>
        </main>
        """,
        encoding="utf-8",
    )

    root, child = ingest_document(source, source_url=SOURCE_URL)["sections"]

    assert root["anchors"] == ["root"]
    assert root["blocks"] == [
        {"type": "paragraph", "text": "Fit preprocessing on training data only."},
        {"type": "paragraph", "text": "Parent summary."},
    ]
    assert child["heading_path"] == ["Data Leakage", "Café"]
    assert child["level"] == 3
    assert child["parent_id"] == root["id"]
    assert child["anchors"] == ["sub", "heading-id", "alias"]
    assert child["blocks"] == [{
        "type": "list",
        "ordered": False,
        "items": [[
            {"type": "paragraph", "text": "Outer text"},
            {"type": "list", "ordered": True, "items": [[{"type": "paragraph", "text": "Inner"}]]},
        ]],
    }]


def test_equal_rank_nested_section_returns_trailing_prose_to_parent(tmp_path: Path):
    source = tmp_path / "nested.html"
    source.write_text(
        """
        <article><section id="outer">
          <h2>Outer</h2>
          <section id="inner"><h2>Inner</h2><p>Inner body.</p></section>
          <p>Outer summary.</p>
        </section></article>
        """,
        encoding="utf-8",
    )

    outer, inner = ingest_document(source, source_url=SOURCE_URL)["sections"]

    assert inner["parent_id"] == outer["id"]
    assert inner["heading_path"] == ["Outer", "Inner"]
    assert inner["blocks"] == [{"type": "paragraph", "text": "Inner body."}]
    assert outer["blocks"] == [{"type": "paragraph", "text": "Outer summary."}]


def test_section_anchor_belongs_to_its_own_heading_after_nested_section(tmp_path: Path):
    source = tmp_path / "late_heading.html"
    source.write_text(
        """
        <article><section id="outer">
          <section id="inner"><h2>Inner</h2></section>
          <h1>Outer</h1>
        </section></article>
        """,
        encoding="utf-8",
    )

    inner, outer = ingest_document(source, source_url=SOURCE_URL)["sections"]

    assert inner["anchors"] == ["inner"]
    assert outer["anchors"] == ["outer"]


def test_article_header_keeps_heading_anchor_and_paragraph_boundaries(tmp_path: Path):
    source = tmp_path / "header.html"
    source.write_text(
        "<article><header><hgroup><h1 id='title'>Title</h1><p>Intro.</p></hgroup></header>"
        "<p>Body.</p></article>",
        encoding="utf-8",
    )

    section, = ingest_document(source, source_url=SOURCE_URL)["sections"]

    assert section["heading"] == "Title"
    assert section["anchors"] == ["title"]
    assert section["blocks"] == [
        {"type": "paragraph", "text": "Intro."},
        {"type": "paragraph", "text": "Body."},
    ]


def test_highlighted_pre_keeps_indentation_and_line_endings(tmp_path: Path):
    source = tmp_path / "code.html"
    source.write_bytes(
        b"<article><h1>Example</h1><details id='cloning'><summary>Cloning</summary>"
        b"<div class='highlight-python'><pre>"
        b"<span>def</span> double(value):\r\n    <span>return</span> value &lt;&lt; 1\r\n\r\n"
        b"</pre></div></details></article>"
    )

    details, = ingest_document(source, source_url=SOURCE_URL)["sections"][0]["blocks"]

    assert details == {
        "type": "details",
        "summary": "Cloning",
        "anchor": "cloning",
        "blocks": [{"type": "code", "text": "def double(value):\r\n    return value << 1\r\n\r\n"}],
    }


def test_table_preserves_caption_rows_headers_and_spans(tmp_path: Path):
    source = tmp_path / "table.html"
    source.write_text(
        """
        <article><h1>Results</h1>
          <table><caption>Cross-validation</caption>
            <thead><tr><th colspan="2">Model and score</th></tr></thead>
            <tbody>
              <tr><th scope="row" rowspan="2">Linear</th><td>0.80</td></tr>
              <tr><td><pre>score = 0.79\n</pre></td></tr>
            </tbody>
          </table>
        </article>
        """,
        encoding="utf-8",
        newline="\n",
    )

    table, = ingest_document(source, source_url=SOURCE_URL)["sections"][0]["blocks"]

    assert table["type"] == "table"
    assert table["caption"] == "Cross-validation"
    header, first, second = table["rows"]
    assert header == {
        "group": "thead",
        "cells": [{
            "header": True,
            "colspan": "2",
            "blocks": [{"type": "paragraph", "text": "Model and score"}],
        }],
    }
    assert first["group"] == second["group"] == "tbody"
    assert first["cells"] == [
        {"header": True, "scope": "row", "rowspan": "2", "blocks": [{"type": "paragraph", "text": "Linear"}]},
        {"header": False, "blocks": [{"type": "paragraph", "text": "0.80"}]},
    ]
    assert second["cells"] == [{"header": False, "blocks": [{"type": "code", "text": "score = 0.79\n"}]}]


def test_paragraph_edits_leave_anchored_and_repeated_unanchored_ids_stable(tmp_path: Path):
    source = tmp_path / "ids.html"
    html = (
        "<main><h1 id='root'>Title</h1><h2>Example</h2><p>Before</p><h2>Example</h2>"
        "<h1 id='other'>Other</h1><h2>Example</h2></main>"
    )
    source.write_text(html, encoding="utf-8")
    before = ingest_document(source, source_url=SOURCE_URL)
    assert ingest_document(source, source_url=SOURCE_URL) == before

    source.write_text(html.replace("Before", "After"), encoding="utf-8")
    after = ingest_document(source, source_url=SOURCE_URL)

    ids = [section["id"] for section in before["sections"]]
    assert len(set(ids)) == 5
    assert [section["parent_id"] for section in before["sections"]] == [None, ids[0], ids[0], None, ids[3]]
    assert [section["id"] for section in after["sections"]] == ids
    assert before["source"]["sha256"] != after["source"]["sha256"]


@pytest.mark.parametrize(
    ("raw", "expected_error", "message"),
    [
        (None, FileNotFoundError, "sample.html"),
        (b" \n", ValueError, "HTML input is empty"),
        (b"<article>\xff</article>", ValueError, "not valid UTF-8"),
        (b"<article><p>\x00</p></article>", ValueError, "NUL character"),
        (b"<html><body><nav>Menu</nav></body></html>", ValueError, "No main documentation content"),
        (b"<main><p> </p></main>", ValueError, "extracted document is empty"),
    ],
)
def test_invalid_input_fails_clearly(tmp_path: Path, raw, expected_error, message):
    source = tmp_path / "sample.html"
    if raw is not None:
        source.write_bytes(raw)

    with pytest.raises(expected_error, match=message):
        ingest_document(source, source_url=SOURCE_URL)


def test_cli_writes_identical_utf8_artifacts_and_preserves_source(tmp_path: Path):
    source = tmp_path / "example.html"
    output = tmp_path / "processed" / "example.json"
    title = "Café — scikit-learn 1.9.1 documentation"
    raw = (
        f'<title>{title}</title><link rel="canonical" href="{SOURCE_URL}">'
        "<article><h1>Café</h1><p>Train first.</p></article>"
    ).encode("utf-8")
    source.write_bytes(raw)
    digest = sha256(raw).hexdigest()
    command = [sys.executable, "-m", "src.ingestion", str(source), str(output), "--expected-sha256"]

    subprocess.run([*command, digest.upper()], check=True, capture_output=True)
    first = output.read_bytes()
    subprocess.run([*command, digest], check=True, capture_output=True)
    assert output.read_bytes() == first
    assert "Café".encode("utf-8") in first
    assert source.read_bytes() == raw

    document = json.loads(first)
    assert document["schema_version"] == 1
    assert document["source"] == {
        "filename": "example.html",
        "url": SOURCE_URL,
        "title": title,
        "documentation_version": "1.9.1",
        "sha256": digest,
    }
    assert document["processing"]["parser"] == "html.parser"
    assert document["processing"]["encoding"] == "UTF-8"

    mismatch = subprocess.run([*command, "0" * 64], check=False, capture_output=True)
    assert mismatch.returncode == 1
    assert b"SHA-256 mismatch" in mismatch.stderr
    assert output.read_bytes() == first


def test_cli_rejects_using_source_as_output(tmp_path: Path):
    source = tmp_path / "snapshot.json"
    raw = b"<article><h1>Original</h1></article>"
    source.write_bytes(raw)

    attempt = subprocess.run(
        [sys.executable, "-m", "src.ingestion", str(source), str(source), "--source-url", SOURCE_URL],
        check=False,
        capture_output=True,
    )

    assert attempt.returncode == 1
    assert b"separate .json file" in attempt.stderr
    assert source.read_bytes() == raw
