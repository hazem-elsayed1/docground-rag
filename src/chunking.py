"""Pack text from one DocGround schema-v1 document into citable chunks."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from src.ingestion import _write_json


DEFAULT_MAX_CHARS = 1200
SEPARATOR = "\n\n"


def _digest(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _field(obj, key, kind, path):
    if not isinstance(obj, dict) or not isinstance(obj.get(key), kind):
        raise ValueError(f"Missing or invalid {path}/{key}")
    return obj[key]


def _range(record, path):
    start = _field(record, "start", int, path)
    end = _field(record, "end", int, path)
    if type(start) is not int or type(end) is not int or not 0 <= start < end:
        raise ValueError(f"Invalid character range at {path}")
    return start, end


def _leaves(blocks, path, anchor):
    """Visit text fields in schema order. Paths are relative to the section."""
    for index, block in enumerate(blocks):
        base = f"{path}/{index}"
        kind = _field(block, "type", str, base)
        if kind in ("paragraph", "code"):
            text = _field(block, "text", str, base)
            if not text.strip():
                raise ValueError(f"Empty text at {base}/text")
            yield f"{base}/text", text, anchor
        elif kind == "list":
            _field(block, "ordered", bool, base)
            for i, item in enumerate(_field(block, "items", list, base)):
                if not isinstance(item, list):
                    raise ValueError(f"Invalid list item at {base}/items/{i}")
                yield from _leaves(item, f"{base}/items/{i}", anchor)
        elif kind == "details":
            panel = _field(block, "anchor", (str, type(None)), base) or anchor
            summary = _field(block, "summary", (str, type(None)), base)
            if summary:
                yield f"{base}/summary", summary, panel
            yield from _leaves(_field(block, "blocks", list, base), f"{base}/blocks", panel)
        elif kind == "table":
            caption = _field(block, "caption", (str, type(None)), base)
            if caption:
                yield f"{base}/caption", caption, anchor
            for r, row in enumerate(_field(block, "rows", list, base)):
                row_path = f"{base}/rows/{r}"
                for c, cell in enumerate(_field(row, "cells", list, row_path)):
                    cell_path = f"{row_path}/cells/{c}"
                    yield from _leaves(_field(cell, "blocks", list, cell_path), f"{cell_path}/blocks", anchor)
        else:
            raise ValueError(f"Unknown block type {kind!r} at {base}")


def chunk_document(document: dict, max_chars: int = DEFAULT_MAX_CHARS) -> dict:
    if type(max_chars) is not int or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    version = _field(document, "schema_version", int, "")
    if type(version) is not int or version != 1:
        raise ValueError("Expected processed schema version 1")
    source = _field(document, "source", dict, "")
    url = _field(source, "url", str, "/source")
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or "#" in url or re.search(r"\s", url):
        raise ValueError("Invalid source URL")
    raw_hash = _field(source, "sha256", str, "/source")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", raw_hash):
        raise ValueError("Invalid source SHA-256")

    document_id = _digest(document)
    chunks = []
    seen_sections = set()
    offset = 0
    for index, section in enumerate(_field(document, "sections", list, "")):
        path = f"/sections/{index}"
        section_id = _field(section, "id", str, path)
        if not section_id.strip() or section_id in seen_sections:
            raise ValueError(f"Empty or duplicate section ID at {path}")
        seen_sections.add(section_id)
        headings = _field(section, "heading_path", list, path)
        anchors = _field(section, "anchors", list, path)
        if not all(isinstance(s, str) and s.strip() for s in headings + anchors):
            raise ValueError(f"Invalid heading path or anchor at {path}")
        anchor = anchors[0] if anchors else None
        current = None
        for field, text, field_anchor in _leaves(_field(section, "blocks", list, path), "/blocks", anchor):
            start = offset
            end = start + len(text)
            if current is None or end - current["start"] > max_chars:
                current = {
                    "document_id": document_id, "source_sha256": raw_hash, "source_url": url,
                    "section_id": section_id, "heading_path": headings, "anchor": anchor,
                    "start": start, "text": text, "fields": [],
                }
                chunks.append(current)
            else:
                current["text"] += SEPARATOR + text
            current["end"] = end
            current["fields"].append({"path": field, "start": start, "end": end, "anchor": field_anchor})
            offset = end + len(SEPARATOR)
    if not chunks:
        raise ValueError("Processed document has no text blocks")
    for chunk in chunks:
        chunk["id"] = _digest(["chunk-v1", document_id, chunk["section_id"], chunk["start"], chunk["end"]])
    return {"chunk_schema_version": 1, "max_chars": max_chars, "chunks": chunks}


def evidence_coverage(evidence_units: list[dict], chunks: list[dict]) -> list[bool]:
    """One flag per required unit. A single supplied chunk must contain the full span."""
    if not evidence_units:
        raise ValueError("At least one evidence unit is required")
    covered = []
    for i, unit in enumerate(evidence_units):
        path = f"/evidence/{i}"
        document_id = _field(unit, "document_id", str, path)
        section_id = _field(unit, "section_id", str, path)
        start = _field(unit, "start", int, path)
        end = _field(unit, "end", int, path)
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise ValueError(f"Invalid evidence range at {path}")
        covered.append(any(
            chunk["document_id"] == document_id and chunk["section_id"] == section_id
            and chunk["start"] <= start < end <= chunk["end"]
            and any(f["start"] <= start < f["end"] for f in chunk["fields"])
            and any(f["start"] < end <= f["end"] for f in chunk["fields"])
            for chunk in chunks
        ))
    return covered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    args = parser.parse_args()
    try:
        if args.output.suffix.lower() != ".json" or args.document.resolve() == args.output.resolve():
            raise ValueError("Output must be a separate .json file")
        document = json.loads(args.document.read_text(encoding="utf-8"))
        _write_json(chunk_document(document, args.max_chars), args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"chunking: {error}\n")


if __name__ == "__main__":
    main()
