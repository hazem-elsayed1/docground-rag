"""Synthetic regression tests for exact citation validation on realistic chunk shapes.

The shapes mirror the supplied contexts of the two historical development cases: multi-field chunks,
multi-line code with quotes, backslashes and non-ASCII text, list items, and headings that are sent to
the model as metadata but are not part of the quotable body. Every citation goes through answer(), so
the provider's JSON text is decoded exactly as in production.
"""
import json
from unittest.mock import Mock

import pytest

from src.generation import InvalidCitation, answer

PROSE = "Use the step name, two underscores and the parameter name."
CODE = 'pipe = Pipeline([("reduce_dim", PCA())])\n>>> pipe.set_params(reduce_dim__n_components=2)\n    "naïve\\path"'
ITEM = "Parameters are addressed as <estimator>__<parameter>."
REPEAT = "    return self"
FIELD_TEXTS = (PROSE, CODE, ITEM, REPEAT, REPEAT)
START = 5000


def _chunk():
    fields, text, offset = [], "", START
    for index, value in enumerate(FIELD_TEXTS):
        if text:
            text += "\n\n"
            offset += 2
        fields.append({"path": f"/blocks/{index}/text", "start": offset, "end": offset + len(value),
                       "anchor": "nested-parameters" if index == 1 else None})
        text += value
        offset += len(value)
    return {"id": "chunk-a", "document_id": "revision-a", "manifest_id": "compose", "section_id": "section-a",
            "heading_path": ["Pipelines", "Access to nested parameters"], "text": text, "start": START,
            "end": START + len(text), "anchor": "pipeline", "snapshot_url": "https://example.org/1.9/compose.html",
            "fields": fields}


OTHER = {"id": "chunk-b", "document_id": "revision-a", "manifest_id": "compose", "section_id": "section-b",
         "heading_path": ["Pipelines"], "text": "Only this chunk mentions memory caching.", "start": 9000,
         "end": 9040, "anchor": None, "snapshot_url": "https://example.org/1.9/compose.html",
         "fields": [{"path": "/blocks/0/text", "start": 9000, "end": 9040, "anchor": None}]}


def _cite(quote, chunk_id="chunk-a", document_id="revision-a"):
    claims = [{"text": "Synthetic claim.", "kind": "explicit",
               "citations": [{"chunk_id": chunk_id, "document_id": document_id, "quote": quote}]}]
    provider = Mock(generate=Mock(return_value=(json.dumps({"status": "answered", "claims": claims}), {})))
    return answer("How are nested parameters set?", [_chunk(), OTHER], provider)["claims"][0]["citations"][0]


@pytest.mark.parametrize(("quote", "field_index", "local_offset"), [
    pytest.param(PROSE, 0, 0, id="whole_prose_field"),
    pytest.param("two underscores", 0, PROSE.index("two underscores"), id="prose_substring"),
    pytest.param(CODE, 1, 0, id="whole_multiline_code_field"),
    pytest.param("pipe.set_params(reduce_dim__n_components=2)\n    \"naïve\\path\"", 1,
                 CODE.index("pipe.set_params"), id="code_with_newline_indent_quote_backslash_non_ascii"),
    pytest.param("<estimator>__<parameter>", 2, ITEM.index("<estimator>"), id="list_item_markup_characters"),
])
def test_exact_quote_resolves_to_its_field_and_document_offsets(quote, field_index, local_offset):
    chunk = _chunk()
    field = chunk["fields"][field_index]
    citation = _cite(quote)
    assert (citation["field"], citation["start"], citation["end"]) == (
        field["path"], field["start"] + local_offset, field["start"] + local_offset + len(quote))
    assert chunk["text"][citation["start"] - START:citation["end"] - START] == quote
    assert citation["url"].endswith("#" + (field["anchor"] or chunk["anchor"]))
    assert citation["quote_valid"] and citation["provenance_valid"]


@pytest.mark.parametrize(("quote", "message"), [
    pytest.param(PROSE + "\n\n" + CODE[:10], "located", id="crosses_a_field_separator"),
    pytest.param("Access to nested parameters", "located", id="heading_metadata_is_not_quotable"),
    pytest.param("pipe.set_params(reduce_dim__n_components=2) \"naïve\\path\"", "located",
                 id="code_whitespace_collapsed"),
    pytest.param(CODE.replace("\n", "\\n"), "located", id="literal_backslash_n_instead_of_newline"),
    pytest.param(CODE.replace("ï", "i"), "located", id="ascii_folded"),
    pytest.param(CODE.replace("ï", "ï"), "located", id="decomposed_unicode_is_not_normalized"),
    pytest.param("Use the step name and the parameter name, joined by two underscores.", "located",
                 id="paraphrase"),
    pytest.param("Only this chunk mentions memory caching.", "located", id="quote_from_another_supplied_chunk"),
    pytest.param(REPEAT, "ambiguous", id="repeated_field_text"),
])
def test_inexact_misplaced_or_ambiguous_quotes_are_rejected(quote, message):
    if message == "located" and quote in _chunk()["text"] and any(
            quote in value for value in FIELD_TEXTS):
        pytest.fail("fixture error: the quote must not occur inside a single field")
    with pytest.raises(InvalidCitation, match=message):
        _cite(quote)


def test_prefix_of_a_field_is_an_exact_substring_and_resolves():
    # Dropping trailing punctuation keeps an exact substring, which is valid evidence of location only.
    assert _cite(PROSE[:-1])["start"] == START


@pytest.mark.parametrize(("chunk_id", "document_id"), [("chunk-x", "revision-a"), ("chunk-a", "revision-b")])
def test_unsupplied_chunk_or_wrong_revision_is_outside_the_evidence(chunk_id, document_id):
    with pytest.raises(InvalidCitation, match="outside the supplied evidence"):
        _cite(PROSE, chunk_id, document_id)
