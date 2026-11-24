"""Rank chunks from one document or a corpus with a deterministic BM25 index."""

import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from math import log1p
from pathlib import Path
import re
from urllib.parse import urlsplit

from src.chunking import SEPARATOR, _digest, _field, _range
from src.corpus import corpus_sources


DEFAULT_TOP_K = 5
K1 = 1.2
B = 0.75
WORD_RUN = r"\w+"
CASE_BOUNDARY = r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
FIELD_PATH = r"/blocks/\d+(?:(?:/items/\d+|/rows/\d+/cells/\d+/blocks|/blocks)/\d+)*/(?:text|summary|caption)"
TOKENIZER = {
    "version": 1,
    "word_pattern": WORD_RUN,
    "case_boundary_pattern": CASE_BOUNDARY,
    "component_delimiter": "_",
    "normalization": "casefold",
    "keep_original_compounds": True,
    "deduplicate_query_terms": True,
}


def tokenize(text: str) -> list[str]:
    tokens = []
    for word in re.findall(WORD_RUN, text):
        parts = [part for piece in word.split("_") for part in re.split(CASE_BOUNDARY, piece) if part]
        if parts:
            tokens.append(word.casefold())
            if parts != [word]:
                tokens.extend(part.casefold() for part in parts)
    return tokens


def _validate_top_k(top_k: int) -> None:
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")


def _validate_chunks(artifact):
    version = _field(artifact, "chunk_schema_version", int, "")
    if type(version) is not int or version != 1:
        raise ValueError("Expected chunk schema version 1")
    max_chars = _field(artifact, "max_chars", int, "")
    if type(max_chars) is not int or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    chunks = _field(artifact, "chunks", list, "")
    if not chunks:
        raise ValueError("Chunk artifact is empty")

    source = None
    next_start = 0
    seen_fields = set()
    section_metadata = {}
    for i, chunk in enumerate(chunks):
        path = f"/chunks/{i}"
        document_id = _field(chunk, "document_id", str, path)
        raw_hash = _field(chunk, "source_sha256", str, path)
        if not re.fullmatch(r"[0-9a-f]{64}", document_id) or not re.fullmatch(r"[0-9a-fA-F]{64}", raw_hash):
            raise ValueError(f"Invalid SHA-256 at {path}")
        url = _field(chunk, "source_url", str, path)
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or "#" in url or re.search(r"\s", url):
            raise ValueError(f"Invalid source URL at {path}")
        chunk_source = (document_id, raw_hash, url)
        if source is None:
            source = chunk_source
        elif chunk_source != source:
            raise ValueError(f"Inconsistent source provenance at {path}")

        section_id = _field(chunk, "section_id", str, path)
        headings = _field(chunk, "heading_path", list, path)
        anchor = _field(chunk, "anchor", (str, type(None)), path)
        if not section_id.strip() or not all(isinstance(s, str) and s.strip() for s in headings):
            raise ValueError(f"Invalid section provenance at {path}")
        if section_metadata.setdefault(section_id, (headings, anchor)) != (headings, anchor):
            raise ValueError(f"Inconsistent section provenance at {path}")

        start, end = _range(chunk, path)
        text = _field(chunk, "text", str, path)
        fields = _field(chunk, "fields", list, path)
        if start != next_start or end - start != len(text) or not fields:
            raise ValueError(f"Inconsistent chunk text or range at {path}")
        if len(text) > max_chars and len(fields) != 1:
            raise ValueError(f"Oversized chunk has multiple fields at {path}")
        expected_id = _digest(["chunk-v1", document_id, section_id, start, end])
        if _field(chunk, "id", str, path) != expected_id:
            raise ValueError(f"Invalid chunk ID at {path}")

        previous_end = start
        for j, field in enumerate(fields):
            field_location = f"{path}/fields/{j}"
            field_path = _field(field, "path", str, field_location)
            field_start, field_end = _range(field, field_location)
            _field(field, "anchor", (str, type(None)), field_location)
            if not re.fullmatch(FIELD_PATH, field_path) or (section_id, field_path) in seen_fields:
                raise ValueError(f"Invalid or duplicate field path at {field_location}")
            seen_fields.add((section_id, field_path))
            separator = SEPARATOR if j else ""
            if (field_start != previous_end + len(separator) or field_end > end
                    or text[previous_end - start:field_start - start] != separator):
                raise ValueError(f"Invalid field boundaries at {field_location}")
            previous_end = field_end
        if previous_end != end:
            raise ValueError(f"Fields do not cover chunk text at {path}")
        next_start = end + len(SEPARATOR)
    return chunks


class BM25:
    def __init__(self, artifact_path: Path):
        self._load(artifact_path.read_bytes())

    @classmethod
    def from_bytes(cls, payload: bytes) -> "BM25":
        """Build an index from bytes the caller has already verified."""
        index = cls.__new__(cls)
        index._load(payload)
        return index

    def _load(self, payload: bytes) -> None:
        artifact = json.loads(payload.decode("utf-8"))
        self.artifact_sha256 = sha256(payload).hexdigest()
        self.sources = None
        if isinstance(artifact, dict) and "corpus_schema_version" in artifact:
            self.sources, self.corpus_revision = corpus_sources(artifact)
            self.chunks = []
            limits = set()
            for source, entry in zip(self.sources, artifact["documents"]):
                nested = _field(entry, "chunk_artifact", dict, "/documents")
                chunks = _validate_chunks(nested)
                first = chunks[0]
                if (first["document_id"], first["source_sha256"]) != (source["document_id"], source["sha256"]):
                    raise ValueError("Nested chunk provenance disagrees with corpus entry")
                self.chunks.extend(chunk | {"manifest_id": source["id"], "snapshot_url": source["url"]}
                                   for chunk in chunks)
                limits.add(nested["max_chars"])
            if len(limits) != 1:
                raise ValueError("Inconsistent chunk size limits in corpus")
            self.max_chars = limits.pop()
        else:
            self.chunks = _validate_chunks(artifact)
            self.corpus_revision = self.chunks[0]["document_id"]
            self.source_sha256 = self.chunks[0]["source_sha256"]
            self.max_chars = artifact["max_chars"]

        self._counts = [Counter(tokenize(chunk["text"])) for chunk in self.chunks]
        lengths = [counts.total() for counts in self._counts]
        n = len(self.chunks)
        average_length = sum(lengths) / n or 1.0
        self._norm = [K1 * (1 - B + B * length / average_length) for length in lengths]
        df = Counter(term for counts in self._counts for term in counts)
        self._idf = {term: log1p((n - count + 0.5) / (count + 0.5)) for term, count in df.items()}

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        _validate_top_k(top_k)
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        terms = sorted(set(tokenize(query)) & self._idf.keys())
        if not terms:
            return []
        # shortcut: scan every chunk for this chapter, use postings if the corpus grows.
        matches = []
        for chunk, counts, norm in zip(self.chunks, self._counts, self._norm):
            score = sum(
                self._idf[term] * counts[term] * (K1 + 1) / (counts[term] + norm)
                for term in terms if term in counts
            )
            if score > 0:
                matches.append((score, chunk))
        matches.sort(key=lambda match: (-match[0], match[1]["id"]))
        return [deepcopy(chunk) | {"score": score} for score, chunk in matches[:top_k]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunks", type=Path)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()
    try:
        results = BM25(args.chunks).search(args.query, args.top_k)
    except (OSError, ValueError) as error:
        parser.exit(1, f"retrieval: {error}\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
