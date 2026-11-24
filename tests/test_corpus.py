import json
from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import benchmark_bm25_development as benchmark
from scripts import fetch_common_pitfalls
from src.chunking import evidence_coverage
from src.corpus import fetch_corpus, load_manifest
from src.evaluation import evaluate
from src.retrieval import BM25


@pytest.fixture
def corpus_inputs(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    manifest = {
        "manifest_schema_version": 1,
        "corpus_version": "fixture-v1",
        "documentation_version": "1.9.1",
        "attribution": {
            "project": "scikit-learn", "copyright": "The scikit-learn developers",
            "license": "BSD-3-Clause",
            "license_url": "https://github.com/scikit-learn/scikit-learn/blob/1.9.1/COPYING",
        },
        "documents": [],
    }
    for name in ("zeta", "alpha"):
        url = f"https://scikit-learn.org/1.9/modules/{name}.html"
        payload = (
            f'<title>Fixture \u2014 scikit-learn 1.9.1 documentation</title>'
            f'<link rel="canonical" href="{url}">'
            '<article><section id="shared"><h1>Usage</h1><pre>if ready:\n    fit()\n</pre></section></article>'
        ).encode()
        (raw_dir / f"{name}.html").write_bytes(payload)
        manifest["documents"].append({"id": name, "url": url, "sha256": sha256(payload).hexdigest()})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, raw_dir


@pytest.mark.parametrize(("key", "value", "message"), [
    ("manifest_schema_version", True, "schema version"),
    ("documentation_version", "stable", "documentation version"),
    ("documents", [], "at least one"),
    ("license_url", "https://example.com/license", "license URL"),
    ("id", "../escape", "document ID"),
    ("id", "con", "document ID"),
    ("id", "zeta", "Duplicate"),
    ("url", "https://scikit-learn.org/stable/modules/alpha.html", "versioned HTTPS"),
    ("url", "https://scikit-learn.org/1.9/modules/zeta.html", "Duplicate"),
    ("sha256", "not-a-hash", "SHA-256"),
])
def test_invalid_manifest_is_rejected(corpus_inputs, key, value, message):
    manifest_path, _ = corpus_inputs
    manifest = json.loads(manifest_path.read_bytes())
    if key in manifest:
        manifest[key] = value
    elif key == "license_url":
        manifest["attribution"][key] = value
    else:
        manifest["documents"][1][key] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_manifest(manifest_path)


def run_cli(action, manifest_path, raw_dir, output_dir):
    return subprocess.run(
        [sys.executable, "-m", "src.corpus", action, "--manifest", str(manifest_path),
         "--raw-dir", str(raw_dir), "--output-dir", str(output_dir)],
        capture_output=True, encoding="utf-8",
    )


def test_build_writes_reproducible_documents_and_chunks_in_manifest_order(corpus_inputs, tmp_path):
    manifest_path, raw_dir = corpus_inputs
    first, second = (tmp_path / name for name in ("first", "second"))
    for directory in (first, second):
        result = run_cli("build", manifest_path, raw_dir, directory)
        assert result.returncode == 0, result.stderr
    for file in ("documents.json", "chunks.json"):
        assert (first / file).read_bytes() == (second / file).read_bytes()

    processed = json.loads((first / "documents.json").read_bytes())
    chunked = json.loads((first / "chunks.json").read_bytes())
    manifest = load_manifest(manifest_path)
    assert [entry["id"] for entry in processed["documents"]] == ["zeta", "alpha"]
    assert [entry["id"] for entry in chunked["documents"]] == ["zeta", "alpha"]
    assert processed["attribution"] == chunked["attribution"] == manifest["attribution"]
    assert processed["manifest_sha256"] == chunked["manifest_sha256"]
    chunks = []
    for entry, doc, group in zip(manifest["documents"], processed["documents"], chunked["documents"], strict=True):
        [chunk] = group["chunk_artifact"]["chunks"]
        assert doc["url"] == group["url"] == chunk["source_url"] == entry["url"]
        assert doc["document_id"] == group["document_id"] == chunk["document_id"]
        assert doc["document"]["source"]["sha256"] == group["sha256"] == chunk["source_sha256"] == entry["sha256"]
        assert chunk["text"] == "if ready:\n    fit()\n"
        assert chunk["start"] == 0
        assert chunk["fields"][0]["path"] == "/blocks/0/text"
        assert chunk["anchor"] == "shared"
        chunks.append(chunk)
    assert all(chunks[0][key] != chunks[1][key] for key in ("id", "document_id", "section_id"))
    span = {key: chunks[0][key] for key in ("document_id", "section_id", "start", "end")}
    assert evidence_coverage([span], [chunks[0]]) == [True]
    assert evidence_coverage([span], [chunks[1]]) == [False]
    assert evidence_coverage([span | {"section_id": chunks[1]["section_id"]}], chunks) == [False]


def test_manifest_fetch_uses_pins_without_overwriting(corpus_inputs, tmp_path, monkeypatch):
    manifest_path, raw_dir = corpus_inputs
    manifest = load_manifest(manifest_path)
    payloads = {doc["url"]: (raw_dir / f"{doc['id']}.html").read_bytes() for doc in manifest["documents"]}
    downloaded = tmp_path / "new-raw"
    monkeypatch.setattr(fetch_common_pitfalls._opener, "open", lambda url, timeout: BytesIO(payloads[url]))
    fetch_corpus(manifest, downloaded)
    for doc in manifest["documents"]:
        assert (downloaded / f"{doc['id']}.html").read_bytes() == payloads[doc["url"]]

    existing = downloaded / "zeta.html"
    existing.write_bytes(b"locally existing different file")
    with pytest.raises(ValueError, match="expected SHA-256"):
        fetch_corpus(manifest, downloaded)
    assert existing.read_bytes() == b"locally existing different file"


@pytest.mark.parametrize(("failure", "message"), [
    ("missing", "No such file"),
    ("hash", "SHA-256 mismatch"),
    ("version", "Documentation version mismatch"),
    ("section collision", "Duplicate section ID"),
])
def test_invalid_source_does_not_publish_corpus(corpus_inputs, tmp_path, failure, message):
    manifest_path, raw_dir = corpus_inputs
    path = raw_dir / "alpha.html"
    if failure == "missing":
        path.unlink()
    elif failure == "hash":
        path.write_bytes(b"changed")
    else:
        old = path.read_bytes()
        changed = (old.replace(b"1.9.1 documentation", b"1.9.2 documentation") if failure == "version"
                   else old.replace(b"/alpha.html", b"/zeta.html"))
        assert changed != old
        path.write_bytes(changed)
        manifest = json.loads(manifest_path.read_bytes())
        manifest["documents"][1]["sha256"] = sha256(changed).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    output_dir = tmp_path / "output"
    result = run_cli("build", manifest_path, raw_dir, output_dir)
    assert result.returncode == 1
    assert message in result.stderr
    assert not output_dir.exists()


@pytest.fixture
def multi(tmp_path, corpus_inputs):
    manifest, raw = corpus_inputs
    out = tmp_path / "wrapped"
    result = run_cli("build", manifest, raw, out)
    assert result.returncode == 0, result.stderr
    document_path = out / "documents.json"
    found = subprocess.run(
        [sys.executable, "-m", "src.annotations", str(document_path), "find", "fit()"],
        capture_output=True, encoding="utf-8", check=True,
    ).stdout
    revision = found.splitlines()[0].split()[-1]
    spans = [json.loads(line) for line in found.splitlines() if line.strip().startswith("{")]
    assert [span["manifest_id"] for span in spans] == ["zeta", "alpha"]
    assert spans[0]["start"] == spans[1]["start"]
    index = BM25(out / "chunks.json")
    assert index.corpus_revision == revision
    by_id = {span["manifest_id"]: span for span in spans}
    # Gold points at the second of two equal-score chunks, exposing cross-document leakage.
    hits = index.search("fit")
    first, second = [by_id[hit["manifest_id"]] for hit in hits]
    draft = {
        "annotation_schema_version": 2, "annotation_version": "synthetic", "split": "development",
        "corpus_revision": revision, "questions": [
            {"id": "one", "question": "fit", "answerable": True, "evidence_units": [[second]]},
            {"id": "both", "question": "fit", "answerable": True, "evidence_units": [[first], [second]]},
            {"id": "either", "question": "fit", "answerable": True, "evidence_units": [[first, second]]},
            {"id": "unknown", "question": "absentword", "answerable": False, "evidence_units": []},
        ],
    }
    return out, index, draft


def test_duplicate_passages_keep_distinct_relevance_and_coverage(multi):
    out, index, draft = multi
    hits = index.search("fit", 2)
    assert hits[0]["score"] == hits[1]["score"] > 0
    assert [hit["id"] for hit in hits] == sorted(hit["id"] for hit in hits)
    assert BM25(out / "chunks.json").search("fit", 2) == hits
    for hit in hits:
        source = next(source for source in index.sources if source["id"] == hit["manifest_id"])
        assert (hit["document_id"], hit["source_sha256"], hit["snapshot_url"]) == (
            source["document_id"], source["sha256"], source["url"],
        )
    report = evaluate(index, draft, 1)
    one, both, either, unknown = report["questions"]
    assert one["precision_at_k"] == one["recall_at_k"] == one["reciprocal_rank_at_k"] == 0
    assert one["evidence_units_covered"] == [False]
    assert both["evidence_units_covered"] == [True, False]
    assert both["recall_at_k"] == 0.5
    assert either["evidence_units_covered"] == [True]
    assert "precision_at_k" not in unknown
    larger = evaluate(index, draft, 2)
    assert larger["summary"]["complete_evidence_coverage_at_k"] == 1
    assert larger["questions"][0]["reciprocal_rank_at_k"] == 0.5


def test_benchmark_unit_recall_counts_questions_and_alternatives_once(multi):
    _, index, draft = multi
    questions = draft["questions"]
    questions.append(questions[0] | {"id": "lexical-miss", "question": "absentword"})
    containment = benchmark.containing_chunks(questions, index.chunks)
    report = benchmark.measure(index, draft, containment)
    top_one = report["cutoffs"]["1"]
    rows = {row["question_id"]: row for row in top_one["questions"]}

    assert top_one["summary"]["macro_unit_recall_at_k"] == 0.375
    assert top_one["summary"]["covered_units"] == 2
    assert top_one["summary"]["required_units"] == 5
    assert rows["one"]["ranking_missed_units"] == [1]
    assert rows["both"]["unit_recall_at_k"] == 0.5
    assert rows["either"]["unit_recall_at_k"] == 1
    assert rows["either"]["evidence_units_covered"] == [True]
    assert len(containment["either"][0]) == 2
    assert "unit_recall_at_k" not in rows["unknown"]
    assert top_one["categories"]["unanswerable"]["result_counts"] == {"unknown": 0}
    assert top_one["categories"]["cross_document"]["macro_unit_recall_at_k"] == 0.5
    assert benchmark.unit_diagnostics(index, questions, containment)["lexical-miss"][0]["best_bm25_rank"] is None


@pytest.fixture
def benchmark_inputs(multi, corpus_inputs, tmp_path):
    out, _, draft = multi
    manifest, _ = corpus_inputs
    for question in draft["questions"]:
        if question["answerable"]:
            question["question"] = "if ready fit"
    pins = {}
    for relative, payload in {
        benchmark.ANNOTATIONS: json.dumps(draft).encode(),
        benchmark.CHUNKS: (out / "chunks.json").read_bytes(),
        benchmark.MANIFEST: manifest.read_bytes(),
    }.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        pins[relative] = sha256(payload).hexdigest()
    return tmp_path.resolve(), pins


def test_pinned_benchmark_repeats_rankings_and_metrics(benchmark_inputs, monkeypatch):
    root, pins = benchmark_inputs
    for relative, digest in pins.items():
        monkeypatch.setitem(benchmark.INPUT_SHA256, relative, digest)
    first = benchmark.benchmark(root)
    second = benchmark.benchmark(root)
    del first["retrieval_latency_ms"], second["retrieval_latency_ms"]
    assert json.dumps(first) == json.dumps(second)
    assert list(first["cutoffs"]) == ["1", "3", "5", "10"]
    assert first["cutoffs"]["10"]["summary"]["macro_unit_recall_at_k"] == 1


def test_pinned_benchmark_is_stable_across_hash_seeds(benchmark_inputs):
    root, pins = benchmark_inputs
    fixture_pins = json.dumps({path.as_posix(): digest for path, digest in pins.items()})
    script = """
import json
from pathlib import Path
import sys

from scripts import benchmark_bm25_development as benchmark
from src.retrieval import BM25

root = Path(sys.argv[1])
benchmark.INPUT_SHA256.update({Path(path): digest for path, digest in json.loads(sys.argv[2]).items()})
report = benchmark.benchmark(root)
annotations, _, chunk_bytes = benchmark.load_inputs(root)
index = BM25.from_bytes(chunk_bytes)
report["ranked_scores"] = {
    question["id"]: [(hit["id"], hit["score"]) for hit in index.search(question["question"], 10)]
    for question in annotations["questions"]
}
print(json.dumps(report, allow_nan=False))
"""
    reports = []
    for seed in ("11", "97"):
        run = subprocess.run(
            [sys.executable, "-c", script, str(root), fixture_pins],
            cwd=Path(__file__).resolve().parents[1],
            env=os.environ | {"PYTHONHASHSEED": seed},
            capture_output=True, encoding="utf-8",
        )
        assert run.returncode == 0, f"PYTHONHASHSEED={seed}: {run.stderr}"
        report = json.loads(run.stdout)
        del report["retrieval_latency_ms"]
        reports.append(report)
    assert reports[0] == reports[1], "Benchmark differs between PYTHONHASHSEED=11 and 97"
    scores = reports[0]["ranked_scores"]["one"]
    first_question = reports[0]["cutoffs"]["10"]["questions"][0]
    assert len(scores) == 2 and scores[0][1] == scores[1][1] > 0
    assert [chunk_id for chunk_id, _ in scores] == first_question["retrieved_chunk_ids"]


def test_corpus_cli_browse_review_export_and_evaluate(multi):
    out, index, draft = multi
    def cli(module, *args):
        return subprocess.check_output([sys.executable, "-m", module, *map(str, args)], encoding="utf-8")
    doc = out / "documents.json"
    selection = cli("src.annotations", doc, "browse", "--document", "alpha", "--section", "shared")
    assert "Document alpha" in selection and "Document zeta" not in selection
    assert "/blocks/0/text" in selection
    draft_path = out / "draft.json"
    draft_path.write_text(json.dumps(draft), encoding="utf-8")
    reviewed = cli("src.annotations", doc, "review", draft_path)
    assert reviewed.count('"fit()"') == 5
    exported = out / "exported.json"
    cli("src.annotations", doc, "export", draft_path, exported)
    assert json.loads(exported.read_bytes()) == draft
    chunks = out / "chunks.json"
    assert json.loads(cli("src.retrieval", chunks, "fit")) == index.search("fit")
    report = json.loads(cli("src.evaluation", chunks, exported, "--top-k", "2"))
    assert report["summary"]["answerable_questions"] == 3
    assert report["sources"] == index.sources


@pytest.mark.parametrize("corruption", [
    "missing manifest_id", "missing document_id", "wrong manifest", "wrong revision",
    "wrong section", "bad endpoint", "old schema", "wrong corpus",
])
def test_invalid_multi_document_references_fail(multi, corruption):
    _, index, draft = multi
    span = draft["questions"][0]["evidence_units"][0][0]
    other = draft["questions"][1]["evidence_units"][0][0]
    if corruption.startswith("missing "):
        del span[corruption.split()[1]]
    elif corruption == "wrong manifest":
        span["manifest_id"] = other["manifest_id"]
    elif corruption == "wrong revision":
        span["document_id"] = other["document_id"]
    elif corruption == "wrong section":
        span["section_id"] = other["section_id"]
    elif corruption == "bad endpoint":
        span["end"] += 100
    elif corruption == "old schema":
        draft["annotation_schema_version"] = 1
    else:
        draft["corpus_revision"] = span["document_id"]
    with pytest.raises(ValueError):
        evaluate(index, draft)


@pytest.mark.parametrize("corruption", ["swap payloads", "duplicate revision", "wrong manifest hash"])
def test_chunk_wrapper_must_bind_payload_to_manifest_entry(multi, corruption):
    out, _, _ = multi
    path = out / "chunks.json"
    wrapper = json.loads(path.read_bytes())
    left, right = wrapper["documents"]
    if corruption == "swap payloads":
        left["chunk_artifact"], right["chunk_artifact"] = right["chunk_artifact"], left["chunk_artifact"]
    elif corruption == "duplicate revision":
        left["document_id"] = right["document_id"]
    else:
        wrapper["manifest_sha256"] = "0" * 64
    path.write_text(json.dumps(wrapper), encoding="utf-8")
    with pytest.raises(ValueError):
        BM25(path)


def test_historical_pilot_contract_is_preserved_for_path_and_verified_bytes(multi):
    out, combined, _ = multi
    entry = json.loads((out / "chunks.json").read_bytes())["documents"][0]
    single_path = out / "single.json"
    single_path.write_text(json.dumps(entry["chunk_artifact"]), encoding="utf-8")
    single = BM25(single_path)
    [hit] = single.search("fit")
    assert "manifest_id" not in hit and "snapshot_url" not in hit
    assert single.corpus_revision == hit["document_id"] != combined.corpus_revision
    field = hit["fields"][0]
    draft = {
        "annotation_schema_version": 1, "annotation_version": "fixture", "split": "development",
        "corpus_revision": single.corpus_revision, "questions": [{
            "id": "legacy", "question": "fit", "answerable": True,
            "evidence_units": [[{"section_id": hit["section_id"], "start": field["start"], "end": field["end"],
                                 "start_block_path": field["path"], "end_block_path": field["path"]}]],
        }],
    }
    report = evaluate(single, draft)
    assert report["source_sha256"] == entry["sha256"] and "sources" not in report
    assert report["questions"][0]["evidence_units_covered"] == [True]
    verified = BM25.from_bytes(single_path.read_bytes())
    assert verified.search("fit") == [hit]
    verified_report = evaluate(verified, draft)
    for evaluation in (report, verified_report):
        del evaluation["summary"]["mean_retrieval_latency_ms"]
        del evaluation["questions"][0]["retrieval_latency_ms"]
    assert verified_report == report
    assert "macro_unit_recall_at_k" not in report["summary"]
    with pytest.raises(ValueError, match="schema version 2"):
        evaluate(combined, draft)


def test_document_wrapper_rejects_substituted_content(multi):
    out, _, _ = multi
    path = out / "documents.json"
    wrapper = json.loads(path.read_bytes())
    left, right = wrapper["documents"]
    left["document"] = right["document"]
    path.write_text(json.dumps(wrapper), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "src.annotations", str(path), "browse"], capture_output=True, encoding="utf-8",
    )
    assert result.returncode == 1
    assert "provenance" in result.stderr
