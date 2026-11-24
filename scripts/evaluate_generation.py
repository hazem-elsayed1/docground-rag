"""Collect a development generation run and score a separate human review."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import mean
from time import perf_counter

import dotenv
from pydantic import BaseModel, ConfigDict
from scripts.benchmark_bm25_development import INPUT_SHA256, load_inputs, provenance
from src.checkpoints import JournalError, parse_journal, run_attempts
from src.chunking import evidence_coverage
from src import groq
from src.gemini import MAX_OUTPUT_TOKENS, RESPONSE_SCHEMA_SHA256, Gemini
from src.generation import CONTEXT_BYTES, GenerationError, PROMPT_VERSION, SYSTEM, answer
from src.runtime import VECTORS, retriever

LOCAL = Path(".local/generation")
PROVIDERS = ("gemini", "groq")
DRAFT = Path("annotations/development-question-draft.md")
DRAFT_HASH = "46b0c870545f850828c731c7866ce91d75f774c40ca2a53db9cae84125b10799"


def references(root, questions):
    path = root / DRAFT
    if path.resolve() != path:
        raise ValueError("Expected fixed development reference path")
    draft = path.read_text(encoding="utf-8")
    if sha256(draft.encode()).hexdigest() != DRAFT_HASH:
        raise ValueError("Development reference digest mismatch")
    blocks = re.split(r"(?m)^### D\d+ · `([^`]+)`[^\n]*\n", draft)[1:]
    records = dict(zip(blocks[::2], blocks[1::2], strict=True))
    if records.keys() != {q["id"] for q in questions}:
        raise ValueError("Reference questions disagree with approved split")
    return {key: value.split("\n## ")[0].strip() for key, value in records.items()}


class ClaimJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    supported: bool
    adequately_cited: bool
    citation_support: list[bool]


class Judgment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    correct: bool
    claims: list[ClaimJudgment]


def ratio(num, denom):
    return num / denom if denom else None


def score(rows: list, reviews: dict) -> dict:
    answered = [row for row in rows if row["status"] == "answered"]
    if reviews.keys() != {row["id"] for row in answered}:
        raise ValueError("Every generated answer requires one human judgment")
    supported = cited = accurate = claims = citations = correct = faithful = 0
    for row in answered:
        review = Judgment.model_validate(reviews[row["id"]])
        if len(review.claims) != len(row["claims"]) or not review.claims:
            raise ValueError("Human judgments must match the generated claims in order")
        for claim, judgment in zip(row["claims"], review.claims, strict=True):
            if (len(judgment.citation_support) != len(claim["citations"])
                    or judgment.adequately_cited and not (judgment.supported and any(judgment.citation_support))):
                raise ValueError("Human judgments must match the cited evidence in order")
            supported += judgment.supported
            cited += judgment.adequately_cited
            accurate += sum(judgment.citation_support)
            claims += 1
            citations += len(claim["citations"])
        correct += row["answerable"] and review.correct
        faithful += all(claim.supported for claim in review.claims)
    evaluated = []
    failures = Counter()
    for row in rows:
        if row["status"] in ("answered", "insufficient", "conflicting"):
            evaluated.append(row)
        else:
            failures[row.get("error_code", row["status"])] += 1
    positives = sum(r["answerable"] for r in evaluated)
    abstained = [r for r in evaluated if r["status"] in ("insufficient", "conflicting")]
    tp = sum(not r["answerable"] for r in abstained)
    usages = [r.get("usage", {}).get("total_token_count") for r in rows]
    failure_count = failures.total()
    return {"question_count": len(rows), "scorable_questions": len(evaluated), "answered": len(answered),
        "errors": failure_count, "failure_rate": ratio(failure_count, len(rows)),
        "failures_by_code": dict(sorted(failures.items())),
        "correctness": ratio(correct, positives), "answer_faithfulness": ratio(faithful, len(answered)),
        "citation_correctness": ratio(accurate, citations), "citation_completeness": ratio(cited, claims),
        "unsupported_claim_rate": ratio(claims - supported, claims), "abstention_precision": ratio(tp, len(abstained)),
        "abstention_recall": ratio(tp, len(evaluated) - positives),
        "answerable_coverage": ratio(sum(r["answerable"] for r in answered), positives),
        "mean_end_to_end_ms": mean(r["end_to_end_ms"] for r in evaluated) if evaluated else None,
        "total_tokens": sum(usages) if usages and all(u is not None for u in usages) else None, "api_cost": None}


def evaluate_item(q, search, provider):
    start = perf_counter()
    trace = {}
    retrieved = start
    hits = []
    try:
        hits = search.search(q["question"], 10)
        retrieved = perf_counter()
        result = answer(q["question"], hits, provider, trace=trace)
    except GenerationError as exc:
        # Outages stop the batch. Final per-question outcomes, such as content blocks, do not.
        result = {"status": "error", "error_code": exc.code, "error": str(exc),
                  "retryable": exc.retryable,
                  "stop_run": exc.stops_collection}
    except Exception as exc:
        # Preserve the checkpoint without recording credentials, queries or stack traces.
        result = {"status": "error", "error_code": "infrastructure_error",
                  "error": type(exc).__name__, "retryable": False, "stop_run": True}
    stop = perf_counter()
    context_ids = set(trace.get("context_selection", {}).get("included_ids", []))
    context = [hit for hit in hits if hit["id"] in context_ids]
    containment = [any(evidence_coverage(unit, context)) for unit in q["evidence_units"]]
    return trace | result | {
        "id": q["id"], "answerable": q["answerable"], "retrieved": [h["id"] for h in hits],
        "approved_units_in_context": containment, "retrieval_ms": (retrieved - start) * 1000,
        "generation_validation_ms": (stop - retrieved) * 1000, "end_to_end_ms": (stop - start) * 1000,
    }


def run_directory(provider):
    """Gemini keeps the original paths. Every other provider has its own journal and review directory."""
    return LOCAL if provider == "gemini" else LOCAL / provider


def collect(questions, root, journal, *, retry_failed=False, resume=False, provider="gemini"):
    if provider == "groq":
        identity = {"provider": "groq", "provider_endpoint": groq.BASE_URL, "request_format": groq.REQUEST_FORMAT,
                    "min_request_interval_s": groq.MIN_INTERVAL_S, "timeout_s": groq.TIMEOUT_S}
        provider = groq.Groq()
    else:
        # The Gemini header keeps its original fields, so existing Gemini journals keep their identity format.
        identity, provider = {}, Gemini(attempts=1)
    model_details = provider.preflight()
    search = retriever("hybrid")
    metadata = provenance(root) | {
        "run_schema_version": 2,
        "prompt_version": PROMPT_VERSION, "prompt_sha256": sha256(SYSTEM.encode()).hexdigest(),
        "response_schema_sha256": RESPONSE_SCHEMA_SHA256,
        "model": provider.model, "model_details": model_details, "method": "hybrid",
        "top_k": 10, "depth": 20, "rrf_constant": 60,
        "input_sha256": {p.as_posix(): h for p, h in INPUT_SHA256.items()}, "reference_sha256": DRAFT_HASH,
        "embedding_sha256": sha256(VECTORS.read_bytes()).hexdigest(), "temperature": 0,
        "max_output_tokens": MAX_OUTPUT_TOKENS, "context_bytes": CONTEXT_BYTES,
        "confirm_free_tier": True, "max_attempts_per_question": 2,
    } | identity
    by_id = {q["id"]: q for q in questions}
    # Pacing waits before each start checkpoint, so it is never timed as latency or left as an unfinished attempt.
    run_attempts(journal, metadata, list(by_id), lambda qid: evaluate_item(by_id[qid], search, provider),
                 retry_failed=retry_failed, resume=resume, before_start=getattr(provider, "wait_for_slot", None))


def review_template(rows):
    return {r["id"]: {"correct": None, "claims": [
        {"supported": None, "adequately_cited": None, "citation_support": [None] * len(claim["citations"])}
        for claim in r["claims"]]} for r in rows if r["status"] == "answered"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("references", "collect", "score"))
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument("--retry-failed", action="store_true",
                              help="Authorize one retry for eligible failures only.")
    continuation.add_argument("--resume", action="store_true",
                              help="Run only questions with no recorded attempt in the existing journal.")
    parser.add_argument("--provider", choices=PROVIDERS, default="gemini",
                        help="Generation provider for collect and score. Each provider has its own journal.")
    args = parser.parse_args()
    continuing = args.retry_failed or args.resume
    if continuing and args.action != "collect":
        parser.error("--retry-failed and --resume require collect")
    if args.provider != "gemini" and args.action == "references":
        parser.error("--provider applies to collect and score")
    root = Path.cwd().resolve()
    annotations, _, _ = load_inputs(root)
    questions = annotations["questions"]
    refs = references(root, questions)
    if args.action == "references":
        print(json.dumps(refs, indent=2, ensure_ascii=True))
        return
    directory = run_directory(args.provider)
    journal = directory / "attempts.jsonl"
    if args.action == "collect":
        # Only provider collection needs credentials. References and scoring stay offline.
        dotenv.load_dotenv(override=False)
        if journal.exists() != continuing:
            parser.error("Start a new journal, or continue the existing one with --resume or --retry-failed.")
        directory.mkdir(parents=True, exist_ok=True)
        try:
            collect(questions, root, journal, retry_failed=args.retry_failed, resume=args.resume,
                    provider=args.provider)
        except GenerationError as exc:
            parser.exit(1, f"Provider preflight failed: {exc}\n")
        except JournalError as exc:
            parser.exit(1, f"{exc} See docs/citation-and-outage-protocol.md for recovery.\n")
    try:
        raw = journal.read_bytes()
        header, history = parse_journal(raw)
    except FileNotFoundError:
        parser.exit(1, "Development attempt journal is missing.\n")
    except JournalError as exc:
        parser.exit(1, f"{exc} See docs/citation-and-outage-protocol.md for recovery.\n")
    # Gemini headers predate the provider field. A journal is only ever scored as its own provider's run.
    if header.get("provider", "gemini") != args.provider:
        parser.error("Journal belongs to a different provider.")
    if not history.keys() <= refs.keys():
        parser.error("Journal contains questions outside the approved development split.")
    rows = [history[q["id"]][-1] | {"id": q["id"], "answerable": q["answerable"], "reference": refs[q["id"]]}
            for q in questions if q["id"] in history]
    digest = sha256(raw).hexdigest()
    if args.action == "collect":
        (directory / "review-template.json").write_text(
            json.dumps({"run_sha256": digest, "judgments": review_template(rows)}, indent=2), encoding="utf-8")
        if len(rows) != len(questions) or any(r["status"] in ("error", "interrupted") for r in rows):
            parser.exit(1, "Partial or failed generation run. See the attempt journal.\n")
        return
    reviews = json.loads((directory / "review.json").read_bytes())
    if reviews["run_sha256"] != digest:
        parser.error("Human review must match the journal digest.")
    metrics = score(rows, reviews["judgments"])
    initial_failures = sum(attempts[0]["status"] in ("error", "interrupted") for attempts in history.values())
    print(json.dumps(metrics | {
        "planned_questions": len(questions), "unattempted_questions": len(questions) - len(rows),
        "initial_failure_rate": ratio(initial_failures, len(history)),
        "retried_questions": sum(len(attempts) == 2 for attempts in history.values()),
    }, indent=2))


if __name__ == "__main__":
    main()
