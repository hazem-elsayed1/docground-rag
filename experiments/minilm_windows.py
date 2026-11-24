"""Development-only comparison of single and overlapping document windows."""
import argparse
from copy import copy
from hashlib import sha256
import json
from pathlib import Path
import os
import platform
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from tokenizers import Tokenizer

from scripts import benchmark_bm25_development as baseline
from scripts.compare_retrieval import config_digest, timed
from src.dense import Dense, build_embeddings, configuration, encode, normalized
from src.evaluation import evaluate_rankings
from src.hybrid import Hybrid
from src.neural import TOKEN_LIMITS, Transformer
from src.retrieval import BM25
from src.runtime import VECTORS

SPEC_COMMIT = "48752414dbfe1c229a967b81a7fa19b059f30844"
SPEC = Path("experiments/minilm-windows-v1.md")
CONFIG = Path("experiments/minilm-windows-v1.json")
ARCHIVE = Path("reports/retrieval-comparison-v2/report.json")
AUDIT = Path("reports/retrieval-comparison-v2/embedding-truncation.json")
METHODS = ("dense", "hybrid")


def token_windows(tokenizer, texts, max_tokens, overlap):
    """Use the pinned post-processor for each native tokenizer overflow window."""
    overflow = Tokenizer.from_str(tokenizer.to_str())
    overflow.no_padding()
    overflow.enable_truncation(max_tokens, stride=overlap)
    return [[first, *first.overflowing] for first in overflow.encode_batch(texts)]


class PreparedWindows:
    """Supply exact encodings to the existing Transformer batch path by row index."""

    def __init__(self, windows, padding):
        self.windows = windows
        self.padding = {key: padding[key] for key in ("direction", "pad_id", "pad_type_id", "pad_token")}

    def encode_batch(self, indices):
        batch = [self.windows[int(index)] for index in indices]
        length = max(len(row.ids) for row in batch)
        for row in batch:
            row.pad(length, **self.padding)
        return batch


def window_vectors(corpus, model, config):
    groups = token_windows(model.tokenizer, [chunk["text"] for chunk in corpus.chunks],
                           config["max_tokens"], config["overlap_content_tokens"])
    counts = [len(group) for group in groups]
    windows = [item for group in groups for item in group]
    prepared = copy(model)
    prepared.tokenizer = PreparedWindows(windows, model.tokenizer.padding)
    vectors = encode(prepared, [str(index) for index in range(len(windows))])
    offsets = np.cumsum([0, *counts])
    means = np.stack([vectors[start:end].mean(axis=0)
                      for start, end in zip(offsets[:-1], offsets[1:], strict=True)])
    return normalized(means, len(groups)), counts, vectors.nbytes


def trial_artifact(corpus, model, path, config):
    vectors, counts, transient_bytes = window_vectors(corpus, model, config)
    metadata = configuration(corpus) | {"experiment_config_sha256": config_digest(config),
        "chunk_ids_sha256": config_digest([chunk["id"] for chunk in corpus.chunks]),
        "vectors_sha256": sha256(vectors.tobytes()).hexdigest()}
    np.savez(path, vectors=vectors, metadata=json.dumps(metadata, sort_keys=True))
    return counts, transient_bytes


def build_pair(root, corpus, model, config):
    """Rebuild both variants without writing to the runtime index directory."""
    archive_digest = sha256((root / VECTORS).read_bytes()).hexdigest()
    builds = {"single": [], "multi": []}
    window_counts = []
    with TemporaryDirectory() as scratch:
        paths = {key: Path(scratch) / f"{key}.npz" for key in builds}
        for rep in range(config["build_repeats"]):
            for variant in list(builds)[::1 if rep % 2 == 0 else -1]:
                path = paths[variant]
                if variant == "single":
                    _, elapsed = timed(build_embeddings, corpus, model, path)
                    count = [1] * len(corpus.chunks)
                    transient = len(corpus.chunks) * 384 * np.dtype(np.float32).itemsize
                else:
                    (count, transient), elapsed = timed(trial_artifact, corpus, model, path, config)
                    if window_counts and window_counts != count:
                        raise ValueError("Window counts changed across builds")
                    window_counts = count
                with np.load(path, allow_pickle=False) as item:
                    matrix = item["vectors"]
                builds[variant].append({
                    "repeat": rep + 1, "build_ms": elapsed, "windows": sum(count),
                    "artifact_bytes": path.stat().st_size, "artifact_sha256": sha256(path.read_bytes()).hexdigest(),
                    "matrix_bytes": matrix.nbytes, "window_vector_bytes": transient,
                    "vector_sha256": sha256(matrix.tobytes()).hexdigest(),
                })
        single = Dense(corpus, model, paths["single"])
        trial = copy(single)
        with np.load(paths["multi"], allow_pickle=False) as item:
            trial.vectors = item["vectors"]
        if len(trial.vectors) != len(corpus.chunks) or trial.vectors.dtype != np.float32:
            raise ValueError("Trial matrix has changed shape or dtype")
    if sha256((root / VECTORS).read_bytes()).hexdigest() != archive_digest:
        raise ValueError("Runtime index changed during the experiment")
    return single, trial, builds, window_counts


def latency_summary(samples):
    median, p95 = np.percentile(samples, [50, 95])
    return {"samples": len(samples), "min_ms": min(samples), "median_ms": float(median),
            "p95_ms": float(p95), "max_ms": max(samples)}


def measure_queries(retrievers, questions, config):
    keys = list(retrievers)
    sample_rows = {key: [] for key in keys}
    rankings = {key: {} for key in keys}
    stable = True
    top_k = config["top_k"]
    for repeat in range(config["query_repeats"] + 1):
        for position, question in enumerate(questions):
            query = question["question"]
            offset = (position + repeat) % len(keys)
            for key in keys[offset:] + keys[:offset]:
                hits, elapsed = timed(retrievers[key].search, query, top_k)
                if not repeat:
                    continue
                ranks = [(hit["id"], hit["score"]) for hit in hits]
                original = rankings[key].setdefault(query, hits)
                stable &= ranks == [(hit["id"], hit["score"]) for hit in original]
                sample_rows[key].append({"question_id": question["id"], "repeat": repeat, "ms": elapsed})
    latency = {}
    for key, rows in sample_rows.items():
        latency[key] = latency_summary([row["ms"] for row in rows]) | {
            "repeats": [latency_summary([row["ms"] for row in rows if row["repeat"] == repeat])
                        for repeat in range(1, config["query_repeats"] + 1)], "raw": rows}
    return rankings, latency, stable


def comparisons(results, audit):
    changes = {}
    for method in METHODS:
        single, trial = (results[f"{method}_{variant}"]["questions"] for variant in ("single", "multi"))
        rows = {}
        for old, new in zip(single, trial, strict=True):
            question_id = old["question_id"]
            row = {"same_ranked_ids": old["retrieved_chunk_ids"] == new["retrieved_chunk_ids"]}
            if old["answerable"]:
                row |= {key: new[key] - old[key] for key in ("precision_at_k", "recall_at_k",
                    "reciprocal_rank_at_k", "unit_recall_at_k", "complete_evidence_coverage_at_k")}
                row["gained_units"] = [i for i, (a, b) in
                    enumerate(zip(old["evidence_units_covered"], new["evidence_units_covered"], strict=True), 1)
                    if b and not a]
                row["lost_units"] = [i for i, (a, b) in
                    enumerate(zip(old["evidence_units_covered"], new["evidence_units_covered"], strict=True), 1)
                    if a and not b]
            rows[question_id] = row
        changes[method] = rows
    affected = []
    for chunk in audit["chunks"]:
        for span in chunk["development_span_overlaps"]:
            if not span["evidence_tokens_omitted"]:
                continue
            question_id, unit = span["question_id"], span["unit"]
            affected.append(span | {"chunk_id": chunk["chunk_id"], "methods": {
                key: {"rank": ids.index(chunk["chunk_id"]) + 1 if chunk["chunk_id"] in ids else None,
                      "unit_covered": row["evidence_units_covered"][unit - 1]}
                for key, result in results.items()
                for row in result["questions"] if row["question_id"] == question_id
                for ids in [row["retrieved_chunk_ids"]]
            }})
    return changes, affected


def decision(results, changes, builds, latency, checks, config):
    rules = config["decision"]
    gains = {method: results[f"{method}_multi"]["summary"]["questions_with_complete_evidence"]
              - results[f"{method}_single"]["summary"]["questions_with_complete_evidence"] for method in METHODS}
    build_ratio = float(np.median([x["build_ms"] for x in builds["multi"]]) /
                        np.median([x["build_ms"] for x in builds["single"]]))
    size_ratio = max(x["artifact_bytes"] for x in builds["multi"]) / builds["single"][0]["artifact_bytes"]
    p95_ratios = {m: latency[f"{m}_multi"]["p95_ms"] / latency[f"{m}_single"]["p95_ms"] for m in METHODS}
    gates = {
        "coverage_gain": gains["hybrid"] >= rules["min_hybrid_gain"] or
            (gains["hybrid"] == 0 and gains["dense"] >= rules["min_dense_gain_on_hybrid_tie"]),
        "no_complete_question_loss": all(row.get("complete_evidence_coverage_at_k", 0) >= 0
                                          for items in changes.values() for row in items.values()),
        "macro_unit_recall_preserved": all(
            results[f"{m}_multi"]["summary"]["macro_unit_recall_at_k"] >=
            results[f"{m}_single"]["summary"]["macro_unit_recall_at_k"] for m in METHODS),
        "build_cost": build_ratio <= rules["max_build_median_ratio"],
        "query_p95_cost": max(p95_ratios.values()) <= rules["max_query_p95_ratio"],
        "artifact_cost": size_ratio <= rules["max_artifact_ratio"],
    }
    outcome = "Inconclusive" if not all(checks.values()) else (
        "Adopt recommendation (requires owner approval)" if all(gates.values()) else "Keep baseline")
    return {"outcome": outcome, "gates": gates, "complete_question_gains": gains,
            "median_build_ratio": build_ratio, "p95_query_ratios": p95_ratios, "artifact_ratio": size_ratio}


def run(root):
    status = subprocess.check_output(["git", "--no-optional-locks", "status", "--porcelain"], cwd=root)
    if status:
        raise ValueError("Commit and verify the experiment before running it")
    subprocess.run(["git", "diff", "--exit-code", SPEC_COMMIT, "--", str(SPEC), str(CONFIG)], cwd=root, check=True)
    provenance = baseline.provenance(root)
    config = json.loads((root / CONFIG).read_bytes())
    if config["max_tokens"] != TOKEN_LIMITS["encoder"] or config["batch_size"] != 8 or config["rrf_constant"] != 60:
        raise ValueError("Controlled encoder or fusion settings differ from the baseline")
    annotations, _, chunk_bytes = baseline.load_inputs(root)
    corpus = BM25.from_bytes(chunk_bytes)
    questions = annotations["questions"]
    archive = json.loads((root / ARCHIVE).read_bytes())
    if any(config[key] != archive["config"][key] for key in ("candidate_depth", "rrf_constant", "batch_size")):
        raise ValueError("Experiment controls differ from the archived configuration")
    audit = json.loads((root / AUDIT).read_bytes())
    chunk_ids = [chunk["id"] for chunk in corpus.chunks]
    if chunk_ids != [row["chunk_id"] for row in audit["chunks"]]:
        raise ValueError("Chunk IDs differ from the audited corpus order")
    model = Transformer("encoder")
    encode(model, [questions[0]["question"]])
    single, multi, builds, counts = build_pair(root, corpus, model, config)
    retrievers = {"dense_single": single, "dense_multi": multi,
                  "hybrid_single": Hybrid(corpus, single, depth=config["candidate_depth"]),
                  "hybrid_multi": Hybrid(corpus, multi, depth=config["candidate_depth"])}
    rankings, latency, rankings_stable = measure_queries(retrievers, questions, config)
    categories = {q["id"]: baseline.question_category(q) for q in questions}
    containment = baseline.containing_chunks(questions, corpus.chunks)
    results = {}
    for key, hits in rankings.items():
        scored = evaluate_rankings(corpus, annotations, config["top_k"], lambda query, _: hits[query])
        results[key], _ = baseline.record_cutoff(scored, categories, containment)
    checks = {"baseline_replay": True, "within_run_rank_stability": rankings_stable,
              "within_run_vector_stability": all(len({r["vector_sha256"] for r in rows}) == 1
                                                 for rows in builds.values())}
    for method in METHODS:
        reference = archive["results"][method][str(config["top_k"])]
        archived_rows = [{k: v for k, v in row.items() if k != "unit_recall_delta_vs_bm25"}
                         for row in reference["questions"]]
        checks["baseline_replay"] &= results[f"{method}_single"] == reference | {"questions": archived_rows}
    changes, affected = comparisons(results, audit)
    return provenance | {
        "benchmark_schema_version": 1, "experiment": config["version"], "split": "development",
        "spec_commit": SPEC_COMMIT, "spec_sha256": sha256((root / SPEC).read_bytes()).hexdigest(),
        "versioned_config_sha256": sha256((root / CONFIG).read_bytes()).hexdigest(),
        "config_sha256": config_digest(config), "config": config,
        "evaluation_protocol": baseline.PROTOCOL_VERSION,
        "archived_report_sha256": sha256((root / ARCHIVE).read_bytes()).hexdigest(),
        "archived_audit_sha256": sha256((root / AUDIT).read_bytes()).hexdigest(),
        "runtime_index_sha256": sha256((root / VECTORS).read_bytes()).hexdigest(),
        "input_sha256": {path.as_posix(): digest for path, digest in baseline.INPUT_SHA256.items()},
        "encoder": configuration(corpus), "hardware": {"processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(), "onnx_provider": model.session.get_providers(),
            "onnx_intra_op_threads": model.session.get_session_options().intra_op_num_threads},
        "chunk_ids_in_matrix_order": chunk_ids, "window_counts_in_matrix_order": counts,
        "structural_ceiling": baseline.structural_ceiling(containment),
        "builds": builds, "query_latency": latency, "results": results, "paired_changes": changes,
        "previously_truncated_evidence": affected, "checks": checks,
        "decision": decision(results, changes, builds, latency, checks, config),
    }


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(run(Path.cwd().resolve()), indent=2, allow_nan=False))
