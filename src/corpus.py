"""Fetch and build a pinned scikit-learn corpus in manifest order."""

import argparse
import json
import ntpath
from pathlib import Path
import re

from scripts.fetch_common_pitfalls import fetch_snapshot
from src.chunking import _digest, _field, chunk_document
from src.ingestion import _write_json, ingest_document


DEFAULT_MANIFEST = Path("corpus/scikit-learn-1.9.1-v1.json")


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict) -> None:
    schema = _field(manifest, "manifest_schema_version", int, "")
    if type(schema) is not int or schema != 1:
        raise ValueError("Expected manifest schema version 1")
    if not _field(manifest, "corpus_version", str, "").strip():
        raise ValueError("corpus_version must be nonempty")
    release = _field(manifest, "documentation_version", str, "")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", release):
        raise ValueError("Expected a major.minor.patch documentation version")
    attribution = _field(manifest, "attribution", dict, "")
    for key in ("project", "copyright", "license"):
        if not _field(attribution, key, str, "/attribution").strip():
            raise ValueError(f"Missing attribution {key}")
    license_url = _field(attribution, "license_url", str, "/attribution")
    if license_url != f"https://github.com/scikit-learn/scikit-learn/blob/{release}/COPYING":
        raise ValueError("Expected the release's scikit-learn license URL")

    documents = _field(manifest, "documents", list, "")
    if not documents:
        raise ValueError("Manifest must contain at least one document")
    _validate_documents(documents, release.rsplit(".", 1)[0])


def corpus_sources(wrapper: dict) -> tuple[list[dict], str]:
    """Check the embedded manifest and bind its ordered entries to document revisions."""
    version = _field(wrapper, "corpus_schema_version", int, "")
    if type(version) is not int or version != 1:
        raise ValueError("Expected corpus schema version 1")
    entries = _field(wrapper, "documents", list, "")
    sources = []
    revisions = set()
    for entry in entries:
        source = {key: _field(entry, key, str, "/documents")
                  for key in ("id", "url", "sha256", "document_id")}
        revision = source["document_id"]
        if not re.fullmatch(r"[0-9a-f]{64}", revision) or revision in revisions:
            raise ValueError("Invalid or duplicate document revision")
        revisions.add(revision)
        sources.append(source)
    manifest = {key: wrapper.get(key) for key in ("corpus_version", "documentation_version", "attribution")}
    manifest["manifest_schema_version"] = 1
    manifest["documents"] = [{k: s[k] for k in ("id", "url", "sha256")} for s in sources]
    validate_manifest(manifest)
    digest = _digest(manifest)
    if digest != wrapper.get("manifest_sha256"):
        raise ValueError("Corpus manifest SHA-256 mismatch")
    return sources, _digest(["corpus-v1", digest, [s["document_id"] for s in sources]])


def _validate_documents(documents, version_series):
    seen_ids = set()
    seen_urls = set()
    pattern = rf"https://scikit-learn\.org/{re.escape(version_series)}/(?:[a-z0-9_]+/)*[a-z0-9_]+\.html"
    for number, entry in enumerate(documents):
        location = f"/documents/{number}"
        name = _field(entry, "id", str, location)
        url = _field(entry, "url", str, location)
        digest = _field(entry, "sha256", str, location)
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name) or ntpath.isreserved(name):
            raise ValueError(f"Invalid document ID at {location}")
        if not re.fullmatch(pattern, url):
            raise ValueError(f"Expected official versioned HTTPS source URL at {location}")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid source SHA-256 at {location}")
        if name in seen_ids or url in seen_urls:
            raise ValueError(f"Duplicate document ID or source URL at {location}")
        seen_ids.add(name)
        seen_urls.add(url)


def _raw_path(raw_dir, entry):
    return raw_dir / f"{entry['id']}.html"


def fetch_corpus(manifest: dict, raw_dir: Path) -> None:
    for entry in manifest["documents"]:
        status = fetch_snapshot(_raw_path(raw_dir, entry), url=entry["url"], expected_sha256=entry["sha256"])
        print(f"{entry['id']}: {status}")


def _add_unique(seen, identifiers, kind):
    for identifier in identifiers:
        if identifier in seen:
            raise ValueError(f"Duplicate {kind} across corpus: {identifier}")
        seen.add(identifier)


def build_corpus(manifest: dict, raw_dir: Path) -> tuple[dict, dict]:
    """Wrap unchanged single-document outputs without sharing text coordinates."""
    provenance = {
        "corpus_schema_version": 1,
        "corpus_version": manifest["corpus_version"],
        "documentation_version": manifest["documentation_version"],
        "manifest_sha256": _digest(manifest),
        "attribution": manifest["attribution"],
    }
    processed = provenance | {"documents": []}
    chunked = provenance | {"documents": []}
    revisions, sections, chunks = set(), set(), set()
    for entry in manifest["documents"]:
        document = ingest_document(_raw_path(raw_dir, entry), expected_sha256=entry["sha256"])
        if document["source"]["documentation_version"] != manifest["documentation_version"]:
            raise ValueError(f"Documentation version mismatch for {entry['id']}")
        artifact = chunk_document(document)
        revision = artifact["chunks"][0]["document_id"]
        _add_unique(revisions, [revision], "document revision")
        _add_unique(sections, (section["id"] for section in document["sections"]), "section ID")
        _add_unique(chunks, (chunk["id"] for chunk in artifact["chunks"]), "chunk ID")
        source = entry | {"document_id": revision}
        processed["documents"].append(source | {"document": document})
        chunked["documents"].append(source | {"chunk_artifact": artifact})
    return processed, chunked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "build"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/corpus-v1"))
    args = parser.parse_args()
    try:
        manifest = load_manifest(args.manifest)
        if args.command == "fetch":
            fetch_corpus(manifest, args.raw_dir)
        else:
            documents_path = args.output_dir / "documents.json"
            chunks_path = args.output_dir / "chunks.json"
            targets = {documents_path.resolve(), chunks_path.resolve()}
            if len(targets) != 2 or args.manifest.resolve() in targets:
                raise ValueError("Corpus outputs must be separate from each other and the manifest")
            if any(_raw_path(args.raw_dir, entry).resolve() in targets for entry in manifest["documents"]):
                raise ValueError("Corpus outputs must be separate from raw snapshots")
            processed, chunks = build_corpus(manifest, args.raw_dir)
            _write_json(processed, documents_path)
            _write_json(chunks, chunks_path)
    except (OSError, ValueError) as error:
        parser.exit(1, f"corpus: {error}\n")


if __name__ == "__main__":
    main()
