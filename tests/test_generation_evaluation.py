from hashlib import sha256
import json
from pathlib import Path
import sys

import pytest
from scripts import evaluate_generation as evaluation
from scripts.evaluate_generation import score
from src.checkpoints import run_attempts
from src.generation import GenerationError


def test_review_metrics_distinguish_abstention_partial_support_and_service_failure():
    base = {"usage": {}, "end_to_end_ms": 5}
    rows = [base | {"id": "a", "answerable": True, "status": "answered",
                    "claims": [{"citations": [1, 2]}, {"citations": [3]}]},
            base | {"id": "b", "answerable": True, "status": "insufficient"},
            base | {"id": "c", "answerable": False, "status": "conflicting"},
            base | {"id": "d", "answerable": False, "status": "error", "error_code": "provider_error"},
            base | {"id": "e", "answerable": True, "status": "error", "error_code": "response_truncated"}]
    review = {"correct": False, "claims": [
        {"supported": True, "adequately_cited": True, "citation_support": [True, True]},
        {"supported": False, "adequately_cited": False, "citation_support": [False]}]}
    metrics = score(rows, {"a": review})
    assert metrics["citation_correctness"] == 2 / 3
    assert metrics["unsupported_claim_rate"] == metrics["citation_completeness"] == 0.5
    assert metrics["abstention_precision"] == metrics["answerable_coverage"] == 0.5
    assert metrics["abstention_recall"] == 1.0
    assert metrics["failure_rate"] == 0.4 and metrics["scorable_questions"] == 3
    assert metrics["failures_by_code"] == {"provider_error": 1, "response_truncated": 1}
    assert metrics["answer_faithfulness"] == metrics["correctness"] == 0
    assert metrics["api_cost"] is None and metrics["total_tokens"] is None
    with pytest.raises(ValueError):
        score(rows, {})
    with pytest.raises(ValueError):
        score(rows, {"a": review | {"claims": review["claims"][:1]}})

@pytest.mark.parametrize("action", ["references", "collect"])
def test_collector_loads_environment_only_for_collect(tmp_path, monkeypatch, action):
    events = []

    def collect(*args, **kwargs):
        events.append("collect")
        raise GenerationError("fixture preflight failure")

    monkeypatch.setattr(evaluation.dotenv, "load_dotenv", lambda *args, **kwargs: events.append(("load_dotenv", kwargs)))
    monkeypatch.setattr(evaluation, "LOCAL", tmp_path / "generation")
    monkeypatch.setattr(evaluation, "load_inputs", lambda root: ({"questions": [{"id": "q", "answerable": False}]}, None, None))
    monkeypatch.setattr(evaluation, "references", lambda root, questions: {"q": ""})
    monkeypatch.setattr(evaluation, "collect", collect)
    monkeypatch.setattr(sys, "argv", ["evaluate_generation", action])
    if action == "references":
        evaluation.main()
    else:
        # Collection stops on the fixture preflight failure.
        with pytest.raises(SystemExit):
            evaluation.main()
    assert events == ([("load_dotenv", {"override": False}), "collect"] if action == "collect" else [])


def test_score_uses_the_same_journal_snapshot_for_rows_and_review_digest(tmp_path, monkeypatch, capsys):
    journal = tmp_path / "attempts.jsonl"
    header = {"model": "example"}
    first = {"status": "error", "error_code": "provider_error", "retryable": True}
    last = {"status": "insufficient", "end_to_end_ms": 5}
    run_attempts(journal, header, ["q"], lambda _: first)
    updated = tmp_path / "updated.jsonl"
    updated.write_bytes(journal.read_bytes())
    run_attempts(updated, header, ["q"], lambda _: last, retry_failed=True)
    newer = updated.read_bytes()
    (tmp_path / "review.json").write_text(
        json.dumps({"run_sha256": sha256(newer).hexdigest(), "judgments": {}}), encoding="utf-8")
    question = {"id": "q", "answerable": False}
    monkeypatch.setattr(evaluation, "LOCAL", tmp_path)
    monkeypatch.setattr(evaluation, "load_inputs", lambda root: ({"questions": [question]}, None, None))
    monkeypatch.setattr(evaluation, "references", lambda root, questions: {"q": ""})
    monkeypatch.setattr(sys, "argv", ["evaluate_generation", "score"])

    # Model a writer completing a retry just before the scorer reads the snapshot.
    read_bytes = Path.read_bytes

    def read_after_update(path):
        if path == journal:
            path.write_bytes(newer)
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_after_update)
    dotenv_calls = []
    monkeypatch.setattr(evaluation.dotenv, "load_dotenv", lambda *args, **kwargs: dotenv_calls.append(kwargs))
    evaluation.main()
    result = json.loads(capsys.readouterr().out)
    assert (result["scorable_questions"], result["errors"], result["retried_questions"]) == (1, 0, 1)
    # Scoring completed offline without loading local configuration.
    assert dotenv_calls == []
