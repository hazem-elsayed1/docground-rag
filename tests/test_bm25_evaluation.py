from hashlib import sha256
import json
from math import log1p
import subprocess
import sys

import pytest

from scripts import benchmark_bm25_development as benchmark
from src.chunking import chunk_document
from src.evaluation import evaluate
from src.retrieval import BM25, tokenize


@pytest.fixture
def synthetic_document():
    return {
        "schema_version": 1,
        "source": {"url": "https://example.org/guide", "sha256": "a" * 64},
        "sections": [
            {
                "id": "primary", "anchors": ["guide"], "heading_path": ["Guide"],
                "blocks": [{"type": "paragraph", "text": text}
                           for text in ("noise noise noise", "alpha beta", "gamma beta")],
            },
            {
                "id": "alternate", "anchors": [], "heading_path": ["Guide", "Alternative"],
                "blocks": [{"type": "paragraph", "text": "alpha beta"}],
            },
        ],
    }


@pytest.fixture
def chunk_path(tmp_path, synthetic_document):
    path = tmp_path / "chunks.json"
    path.write_text(json.dumps(chunk_document(synthetic_document, max_chars=1)), encoding="utf-8")
    return path


def evidence_span(first, last=None):
    last = first if last is None else last
    return {
        "section_id": first["section_id"],
        "start_block_path": first["fields"][0]["path"],
        "end_block_path": last["fields"][-1]["path"],
        "start": first["start"],
        "end": last["end"],
    }


def questions_for(index, *questions):
    return {
        "annotation_schema_version": 1, "annotation_version": "synthetic-1",
        "corpus_revision": index.corpus_revision, "split": "development", "questions": list(questions),
    }


@pytest.mark.parametrize(("text", "expected"), [
    ("train_test_split", ["train_test_split", "train", "test", "split"]),
    ("StandardScaler", ["standardscaler", "standard", "scaler"]),
    ("Pipeline", ["pipeline"]),
    ("sklearn.HTTPServer().CAF\u00c9", ["sklearn", "httpserver", "http", "server", "caf\u00e9"]),
])
def test_tokenizer_preserves_and_expands_technical_identifiers(text, expected):
    assert tokenize(text) == expected


def test_bm25_ranks_rare_terms_and_breaks_ties_by_id_with_provenance(chunk_path):
    index = BM25(chunk_path)
    _, alpha, gamma, alternate = index.chunks
    ranked = index.search("GAMMA alpha alpha", 3)
    assert [hit["id"] for hit in ranked] == [gamma["id"], *sorted([alpha["id"], alternate["id"]])]
    idf = log1p(3.5 / 1.5)
    gamma_norm = 1.2 * (0.25 + 0.75 * 2 / 2.25)
    noise_norm = 1.2 * (0.25 + 0.75 * 3 / 2.25)
    assert ranked[0] == gamma | {"score": pytest.approx(idf * 2.2 / (1 + gamma_norm))}
    assert index.search("noise")[0]["score"] == pytest.approx(idf * 3 * 2.2 / (3 + noise_norm))
    assert ranked[1]["score"] == ranked[2]["score"]
    assert index.search("alpha gamma", 2) == ranked[:2]
    assert BM25(chunk_path).search("gamma alpha", 3) == ranked


@pytest.mark.parametrize("query", ["", " __ ... ", "unknownvocabulary"])
def test_empty_or_unknown_query_returns_no_chunks(chunk_path, query):
    assert BM25(chunk_path).search(query) == []


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_k_is_rejected(chunk_path, top_k):
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        BM25(chunk_path).search("alpha", top_k)


@pytest.mark.parametrize(("corruption", "message"), [
    ("json", "Expecting"), ("version", "schema version"), ("empty", "artifact is empty"),
    ("id", "Invalid chunk ID"), ("source", "Inconsistent source provenance"),
    ("field", "Invalid field boundaries"),
])
def test_bad_chunk_artifacts_are_rejected_before_indexing(chunk_path, corruption, message):
    artifact = json.loads(chunk_path.read_bytes())
    if corruption == "version":
        artifact["chunk_schema_version"] = True
    elif corruption == "empty":
        artifact["chunks"] = []
    elif corruption == "id":
        artifact["chunks"][1]["id"] = artifact["chunks"][0]["id"]
    elif corruption == "source":
        artifact["chunks"][1]["source_sha256"] = "b" * 64
    elif corruption == "field":
        artifact["chunks"][0]["fields"][0]["start"] += 1
    chunk_path.write_text("{" if corruption == "json" else json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        BM25(chunk_path)


def test_macro_metrics_and_multi_unit_alternatives(chunk_path):
    index = BM25(chunk_path)
    noise, alpha, gamma, alternate = index.chunks
    first_tie, second_tie = sorted([alpha, alternate], key=lambda chunk: chunk["id"])
    units = [[evidence_span(second_tie), evidence_span(first_tie)], [evidence_span(gamma)]]
    annotations = questions_for(
        index,
        {"id": "synthetic-1", "question": "noise alpha", "answerable": True, "evidence_units": units},
        {"id": "synthetic-2", "question": "gamma alpha", "answerable": True, "evidence_units": units},
        {"id": "synthetic-3", "question": "unknownvocabulary", "answerable": False, "evidence_units": []},
    )
    report = evaluate(index, annotations, top_k=2)
    first, second, unanswerable = report["questions"]
    assert first["retrieved_chunk_ids"] == [noise["id"], first_tie["id"]]
    assert second["retrieved_chunk_ids"] == [gamma["id"], first_tie["id"]]
    assert first["evidence_units_covered"] == [True, False]
    assert second["evidence_units_covered"] == [True, True]
    assert unanswerable["retrieved_chunk_ids"] == []
    assert "precision_at_k" not in unanswerable
    assert report["summary"] == {
        "answerable_questions": 2, "unanswerable_questions": 1, "unanswerable_no_results": 1,
        "precision_at_k": 0.75, "recall_at_k": 0.5, "mrr_at_k": 0.75, "complete_evidence_coverage_at_k": 0.5,
        "mean_retrieval_latency_ms": pytest.approx(sum(row["retrieval_latency_ms"] for row in report["questions"]) / 3),
    }
    assert all(row["retrieval_latency_ms"] >= 0 for row in report["questions"])
    assert report["chunk_artifact_sha256"] == sha256(chunk_path.read_bytes()).hexdigest()
    assert report["corpus_revision"] == index.chunks[0]["document_id"]
    assert report["source_sha256"] == index.chunks[0]["source_sha256"]
    assert report["tokenizer"]["normalization"] == "casefold"
    assert report["bm25"]["k1"] == 1.2 and report["bm25"]["b"] == 0.75
    assert report["split"] == "development" and report["evaluation_protocol_version"] == 1


def test_same_span_gets_complete_coverage_only_when_contiguous_in_one_chunk(chunk_path, synthetic_document):
    index = BM25(chunk_path)
    _, alpha, gamma, alternate = index.chunks
    annotations = questions_for(index, {
        "id": "synthetic-cross-chunk", "question": "beta", "answerable": True,
        "evidence_units": [[evidence_span(alpha, gamma) | {"start": alpha["start"] + 1, "end": gamma["end"] - 1}]],
    })
    row = evaluate(index, annotations, top_k=6)["questions"][0]
    assert set(row["retrieved_chunk_ids"]) == {alpha["id"], gamma["id"], alternate["id"]}
    assert row["precision_at_k"] == pytest.approx(2 / 6)
    assert row["recall_at_k"] == 1.0
    assert row["complete_evidence_coverage_at_k"] == 0.0

    chunk_path.write_text(json.dumps(chunk_document(synthetic_document)), encoding="utf-8")
    merged = evaluate(BM25(chunk_path), annotations, top_k=6)["questions"][0]
    assert merged["complete_evidence_coverage_at_k"] == 1.0


def test_benchmark_ceiling_excludes_span_split_across_chunks(chunk_path):
    index = BM25(chunk_path)
    _, alpha, gamma, alternate = index.chunks
    document = {"document_id": index.corpus_revision}
    question = {
        "id": "split-unit",
        "question": "beta",
        "answerable": True,
        "evidence_units": [[evidence_span(alpha, gamma) | document], [evidence_span(alternate) | document]],
    }
    containment = benchmark.containing_chunks([question], index.chunks)
    assert benchmark.structural_ceiling(containment) == {
        "structurally_containable_units": 1,
        "structurally_uncontainable_units": 1,
        "maximum_macro_unit_recall": 0.5,
        "maximum_complete_evidence_coverage": 0,
    }
    report = benchmark.measure(index, questions_for(index, question), containment)
    row = report["cutoffs"]["3"]["questions"][0]
    assert row["missing_evidence_units"] == row["structurally_uncontainable_units"] == [1]
    assert row["ranking_missed_units"] == []


def test_all_unanswerable_split_has_no_quality_averages(chunk_path):
    index = BM25(chunk_path)
    annotations = questions_for(index, {
        "id": "synthetic-unanswerable", "question": "gamma", "answerable": False, "evidence_units": [],
    }) | {"split": "frozen-test"}
    summary = evaluate(index, annotations)["summary"]
    assert summary["unanswerable_questions"] == 1
    assert summary["unanswerable_no_results"] == 0
    assert all(summary[key] is None for key in (
        "precision_at_k", "recall_at_k", "mrr_at_k", "complete_evidence_coverage_at_k",
    ))


def test_answerable_query_without_hits_scores_zero(chunk_path):
    index = BM25(chunk_path)
    annotations = questions_for(index, {
        "id": "synthetic-miss", "question": "unknownvocabulary", "answerable": True,
        "evidence_units": [[evidence_span(index.chunks[0])]],
    })
    summary = evaluate(index, annotations)["summary"]
    assert all(summary[key] == 0 for key in (
        "precision_at_k", "recall_at_k", "mrr_at_k", "complete_evidence_coverage_at_k",
    ))


@pytest.mark.parametrize(("change", "message"), [
    ({"revision": "other"}, "corpus_revision"),
    ({"split": "both"}, "split must be"),
    ({"answerable": False}, "only for answerable"),
    ({"evidence_units": [[]]}, "alternative span"),
    ({"start_block_path": "/blocks/99/text"}, "does not match start_block_path"),
    ({"start": True}, "Invalid character range"),
])
def test_annotation_validation_rejects_invalid_coordinates_and_metadata(chunk_path, change, message):
    index = BM25(chunk_path)
    coordinates = {key: value for key, value in change.items() if key in ("start_block_path", "start")}
    question = {
        "id": "synthetic-invalid", "question": "noise", "answerable": change.get("answerable", True),
        "evidence_units": change.get("evidence_units", [[evidence_span(index.chunks[0]) | coordinates]]),
    }
    annotations = questions_for(index, question)
    annotations["split"] = change.get("split", annotations["split"])
    annotations["corpus_revision"] = change.get("revision", annotations["corpus_revision"])
    with pytest.raises(ValueError, match=message):
        evaluate(index, annotations)


def test_cli_returns_json_for_retrieval_and_evaluation(chunk_path, tmp_path):
    annotations_path = tmp_path / "synthetic.json"
    index = BM25(chunk_path)
    annotations_path.write_text(json.dumps(questions_for(index, {
        "id": "synthetic-cli", "question": "gamma", "answerable": True,
        "evidence_units": [[evidence_span(index.chunks[2])]],
    })), encoding="utf-8")
    retrieval = subprocess.run(
        [sys.executable, "-m", "src.retrieval", str(chunk_path), "gamma", "--top-k", "1"],
        capture_output=True, check=True,
    )
    evaluation = subprocess.run(
        [sys.executable, "-m", "src.evaluation", str(chunk_path), str(annotations_path), "--top-k", "1"],
        capture_output=True, check=True,
    )
    assert json.loads(retrieval.stdout)[0]["id"] == json.loads(evaluation.stdout)["questions"][0]["retrieved_chunk_ids"][0]
