"""Measure local retrieval stages once per development question; score their prefixes."""

from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter_ns

from scripts import benchmark_bm25_development as baseline
from src.runtime import VECTORS
from src.dense import Dense, configuration, encode
from src.evaluation import evaluate_rankings
from src.hybrid import fuse
from src.neural import MODELS, Transformer
from src.rerank import rerank
from src.retrieval import BM25, B, K1, TOKENIZER

DEPTH = 20
METHODS = ("bm25", "dense", "hybrid", "reranked")
CONFIG_SERIALIZATION = {
    "object": "config", "encoding": "UTF-8", "sort_keys": True, "ensure_ascii": False,
    "separators": [",", ":"], "allow_nan": False, "trailing_newline": False,
}


def config_digest(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def timed(operation, *args):
    start = perf_counter_ns()
    value = operation(*args)
    return value, (perf_counter_ns() - start) / 1e6


def run(root: Path) -> dict:
    provenance = baseline.provenance(root)
    annotations, _, payload = baseline.load_inputs(root)
    corpus = BM25.from_bytes(payload)
    dense = Dense(corpus, Transformer("encoder"), VECTORS)
    cross = Transformer("reranker")
    questions = annotations["questions"]
    containment = baseline.containing_chunks(questions, corpus.chunks)
    categories = {q["id"]: baseline.question_category(q) for q in questions}
    rankings = {m: {} for m in METHODS}
    traces, latency = {}, {}
    for q in questions:
        query = q["question"]
        sparse, s = timed(corpus.search, query, DEPTH)
        vector, e = timed(encode, dense.model, [query])
        semantic, d = timed(dense.rank_vector, vector[0], DEPTH)
        fused, f = timed(fuse, {"bm25": sparse, "dense": semantic}, DEPTH)
        final, r = timed(rerank, cross, query, fused, DEPTH)
        latency[q["id"]] = {"bm25": s, "query_embedding": e, "cosine_search": d,
                            "fusion": f, "pair_inference_and_sort": r}
        traces[q["id"]] = {}
        for method, hits in zip(METHODS, (sparse, semantic, fused, final), strict=True):
            rankings[method][query] = hits
            traces[q["id"]][method] = [{k: hit[k] for k in
                ("id", "score", "components", "initial_rank", "fusion_score") if k in hit} for hit in hits]
    results = {}
    for method in METHODS:
        results[method] = {}
        for k in baseline.CUTOFFS:
            scores = evaluate_rankings(corpus, annotations, k, lambda q, n: rankings[method][q][:n])
            cutoff, _ = baseline.record_cutoff(scores, categories, containment)
            results[method][str(k)] = cutoff
    archive = root / "reports/bm25-development-multi-document-v1/report.json"
    if results["bm25"] != json.loads(archive.read_bytes())["cutoffs"]:
        raise ValueError("Development BM25 differs from archived 6A results")
    for method in METHODS[1:]:
        for k, cutoff in results[method].items():
            for row, base in zip(cutoff["questions"], results["bm25"][k]["questions"], strict=True):
                if row["answerable"]:
                    row["unit_recall_delta_vs_bm25"] = row["unit_recall_at_k"] - base["unit_recall_at_k"]
    config = {"models": MODELS, "embedding": configuration(corpus), "rrf_constant": 60,
              "candidate_depth": DEPTH, "reranker_max_tokens": 512, "batch_size": 8,
              "bm25": {"k1": K1, "b": B, "tokenizer": TOKENIZER}, "cutoffs": baseline.CUTOFFS}
    return provenance | {"split": "development", "evaluation_protocol": baseline.PROTOCOL_VERSION,
        "input_sha256": {p.as_posix(): h for p, h in baseline.INPUT_SHA256.items()},
        "embedding_file_sha256": sha256(VECTORS.read_bytes()).hexdigest(),
        "benchmark_schema_version": 2, "config_sha256": config_digest(config),
        "config_serialization": CONFIG_SERIALIZATION,
        "config": config, "corpus_revision": corpus.corpus_revision,
        "ceiling": baseline.structural_ceiling(containment), "results": results,
        "candidates": traces, "stage_latency_ms": latency}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd().resolve()), indent=2, allow_nan=False))
