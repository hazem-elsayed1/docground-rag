"""Print the pinned multi-document BM25 development benchmark as JSON."""

import argparse
from hashlib import sha256
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
from statistics import mean
import subprocess

from src.chunking import _digest, evidence_coverage
from src.evaluation import METRICS, PROTOCOL_VERSION, evaluate
from src.retrieval import B, BM25, K1, TOKENIZER


ANNOTATIONS = Path("annotations/development/multi-document-v1.json")
CHUNKS = Path("data/processed/corpus-v1/chunks.json")
MANIFEST = Path("corpus/scikit-learn-1.9.1-v1.json")
INPUT_SHA256 = {
    ANNOTATIONS: "9f7131bfeedf80cd82cb0f956f97cd7b9a5b28871843060c446fed80ff3510a2",
    CHUNKS: "cee1cbdfc77457a77a6a1741bd72937f8095c511f0b4c4208616367e7dcfa882",
    MANIFEST: "805962c9e6fcd5b077fae77664de5cfa51955563ee09de97fa3bde2c6a8fd04c",
}
CUTOFFS = (1, 3, 5, 10)
# The reviewed development draft lists these IDs under Direct factual.
DIRECT_FACTUAL_IDS = {
    "pipeline-nested-parameter-key", "estimator-repr-text-mode",
    "leave-one-out-split-count", "cross-validate-return-indices",
    "confusion-matrix-axis-convention", "top-k-accuracy-multilabel-support",
}


def pinned_input(root: Path, relative: Path) -> bytes:
    expected = INPUT_SHA256.get(relative)
    if expected is None:
        raise ValueError("Only the fixed benchmark input paths are permitted")
    path = root / relative
    if path.resolve() != path:
        raise ValueError(f"Expected a file at the fixed path {relative.as_posix()}")
    payload = path.read_bytes()
    actual = sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {relative.as_posix()}: expected {expected}, got {actual}")
    return payload


def load_inputs(root: Path) -> tuple[dict, dict, bytes]:
    inputs = {path: pinned_input(root, path) for path in INPUT_SHA256}
    annotations = json.loads(inputs[ANNOTATIONS])
    if annotations.get("split") != "development":
        raise ValueError("This benchmark requires the development split")
    wrapper = json.loads(inputs[CHUNKS])
    if wrapper["manifest_sha256"] != _digest(json.loads(inputs[MANIFEST])):
        raise ValueError("Chunk artifact does not match the pinned manifest")
    return annotations, wrapper, inputs[CHUNKS]


def question_category(question: dict) -> str:
    if not question["answerable"]:
        return "unanswerable"
    if question["id"] in DIRECT_FACTUAL_IDS:
        return "direct_factual"
    units = question["evidence_units"]
    if len(units) == 1:
        return "conceptual"
    unit_documents = [{span["document_id"] for span in alternatives} for alternatives in units]
    if set.intersection(*unit_documents):
        return "multi_evidence_single_document"
    return "cross_document"


def containing_chunks(questions: list[dict], chunks: list[dict]) -> dict:
    by_question = {}
    for question in questions:
        if question["answerable"]:
            by_question[question["id"]] = [
                [chunk["id"] for chunk in chunks if any(evidence_coverage(alternatives, [chunk]))]
                for alternatives in question["evidence_units"]
            ]
    return by_question


def unit_recall(covered: list[bool]) -> float:
    return sum(covered) / len(covered)


def structural_ceiling(containment: dict) -> dict:
    flags = [[bool(chunk_ids) for chunk_ids in units] for units in containment.values()]
    return {
        "structurally_containable_units": sum(map(sum, flags)),
        "structurally_uncontainable_units": sum(len(units) - sum(units) for units in flags),
        "maximum_macro_unit_recall": mean(map(unit_recall, flags)),
        "maximum_complete_evidence_coverage": mean(all(units) for units in flags),
    }


def unit_diagnostics(index: BM25, questions: list[dict], containment: dict) -> dict:
    diagnostics = {}
    for question in questions:
        question_id = question["id"]
        if question_id not in containment:
            continue
        hits = index.search(question["question"], len(index.chunks))
        ranks = {hit["id"]: rank for rank, hit in enumerate(hits, 1)}
        diagnostics[question_id] = [
            {
                "unit": unit,
                "containing_chunk_ids": chunk_ids,
                "best_bm25_rank": min((ranks[chunk_id] for chunk_id in chunk_ids if chunk_id in ranks), default=None),
            }
            for unit, chunk_ids in enumerate(containment[question_id], 1)
        ]
    return diagnostics


def summarize_units(rows: list[dict]) -> dict:
    return {
        "macro_unit_recall_at_k": mean(row["unit_recall_at_k"] for row in rows),
        "covered_units": sum(sum(row["evidence_units_covered"]) for row in rows),
        "required_units": sum(len(row["evidence_units_covered"]) for row in rows),
        "questions_with_complete_evidence": sum(bool(row["complete_evidence_coverage_at_k"]) for row in rows),
        "ranking_missed_units": sum(len(row["ranking_missed_units"]) for row in rows),
    }


def summarize_unanswerable(rows: list[dict]) -> dict:
    counts = {row["question_id"]: len(row["retrieved_chunk_ids"]) for row in rows}
    return {"questions": len(rows), "no_results": sum(count == 0 for count in counts.values()),
            "result_counts": counts}


def category_breakdown(rows: list[dict]) -> dict:
    groups = {}
    for row in rows:
        groups.setdefault(row["category"], []).append(row)
    categories = {}
    for category, group in groups.items():
        if category == "unanswerable":
            categories[category] = summarize_unanswerable(group)
            continue
        averages = {metric: mean(row[metric] for row in group) for metric in METRICS}
        averages["mrr_at_k"] = averages.pop("reciprocal_rank_at_k")
        categories[category] = {"questions": len(group)} | averages | summarize_units(group)
    return categories


def unit_metrics(flags: list[bool], containment: list[list[str]]) -> dict:
    ranking_misses = []
    structural_misses = []
    for unit, (covered, chunk_ids) in enumerate(zip(flags, containment, strict=True), 1):
        if covered:
            continue
        if chunk_ids:
            ranking_misses.append(unit)
        else:
            structural_misses.append(unit)
    return {
        "unit_recall_at_k": unit_recall(flags),
        "missing_evidence_units": sorted(ranking_misses + structural_misses),
        "ranking_missed_units": ranking_misses,
        "structurally_uncontainable_units": structural_misses,
    }


def record_cutoff(evaluation: dict, categories: dict, containment: dict) -> tuple[dict, dict]:
    rows, per_question_latency = [], {}
    for row in evaluation["questions"]:
        question_id = row["question_id"]
        per_question_latency[question_id] = row["retrieval_latency_ms"]
        record = {key: score for key, score in row.items() if key != "retrieval_latency_ms"}
        record["category"] = categories[question_id]
        if row["answerable"]:
            record.update(unit_metrics(row["evidence_units_covered"], containment[question_id]))
        rows.append(record)
    answerable_rows = [row for row in rows if row["answerable"]]
    summary = {key: score for key, score in evaluation["summary"].items()
               if key not in ("mean_retrieval_latency_ms", "unanswerable_no_results", "unanswerable_questions")}
    cutoff = {
        "summary": summary | summarize_units(answerable_rows),
        "categories": category_breakdown(rows),
        "questions": rows,
    }
    return cutoff, {"mean": evaluation["summary"]["mean_retrieval_latency_ms"], "per_question": per_question_latency}


def measure(index: BM25, annotations: dict, containment: dict) -> dict:
    categories = {question["id"]: question_category(question) for question in annotations["questions"]}
    cutoffs, timings = {}, {}
    for top_k in CUTOFFS:
        evaluation = evaluate(index, annotations, top_k)
        cutoffs[str(top_k)], timings[str(top_k)] = record_cutoff(evaluation, categories, containment)
    return {
        "evaluation_protocol_version": evaluation["evaluation_protocol_version"],
        "annotation_schema_version": evaluation["annotation_schema_version"],
        "annotation_version": evaluation["annotation_version"],
        "split": evaluation["split"],
        "corpus_revision": evaluation["corpus_revision"],
        "chunk_count": len(index.chunks),
        "chunk_max_chars": evaluation["chunk_max_chars"],
        "bm25": evaluation["bm25"],
        "tokenizer": evaluation["tokenizer"],
        "cutoffs": cutoffs,
        "retrieval_latency_ms": timings,
    }


def benchmark(root: Path) -> dict:
    if (K1, B, TOKENIZER["version"], PROTOCOL_VERSION) != (1.2, 0.75, 1, 1):
        raise ValueError("BM25, tokenizer or evaluation protocol differs from the pinned baseline")
    annotations, wrapper, chunk_bytes = load_inputs(root)
    chunks = [chunk for entry in wrapper["documents"] for chunk in entry["chunk_artifact"]["chunks"]]
    containment = containing_chunks(annotations["questions"], chunks)
    ceiling = structural_ceiling(containment)
    index = BM25.from_bytes(chunk_bytes)
    return {
        "input_sha256": {path.as_posix(): digest for path, digest in INPUT_SHA256.items()},
        "structural_coverage": ceiling,
        **measure(index, annotations, containment),
        "evidence_diagnostics": unit_diagnostics(index, annotations["questions"], containment),
    }


def provenance(root: Path) -> dict:
    changes = subprocess.check_output(
        ["git", "--no-optional-locks", "status", "--porcelain=v1", "--untracked-files=all", "--", "src", "scripts"],
        cwd=root, text=True,
    )
    if changes:
        raise ValueError("Commit the benchmark source and scripts before recording a benchmark")
    git_commit, git_tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD", "HEAD^{tree}"], cwd=root, text=True,
    ).splitlines()
    dependencies = {dist.metadata["Name"]: dist.version for dist in distributions()}
    return {
        "benchmark_schema_version": 1,
        "git_commit": git_commit,
        "git_tree": git_tree,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": dict(sorted(dependencies.items(), key=lambda package: package[0].casefold())),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        report = provenance(root) | benchmark(root)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"benchmark: {error}\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
