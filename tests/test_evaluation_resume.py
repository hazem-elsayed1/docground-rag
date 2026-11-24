"""Resuming a stopped development collection from its intact journal.

Requests go through evaluate_item, the real google-genai SDK and an in-memory httpx2 transport. The
question sent in each request is read back from the request body, so the tests observe exactly which
question IDs reach the provider.
"""
import json
import sys
from types import SimpleNamespace

import httpx2
import pytest

from scripts import evaluate_generation as evaluation
from scripts.evaluate_generation import evaluate_item
from src.checkpoints import JournalError, read_journal, run_attempts
from src.gemini import Gemini

HEADER = {"run_schema_version": 2, "git_commit": "synthetic", "prompt_sha256": "synthetic"}
IDS = ["q1", "q2", "q3", "q4"]
QUESTIONS = {qid: {"id": qid, "question": f"Synthetic question {qid}?", "answerable": False, "evidence_units": []}
             for qid in IDS}
BY_TEXT = {q["question"]: qid for qid, q in QUESTIONS.items()}
HIT = {"id": "chunk", "document_id": "revision", "section_id": "section", "heading_path": ["Synthetic"],
       "text": "Synthetic documentation evidence."}
SEARCH = SimpleNamespace(search=lambda question, top_k: [HIT])
ANSWER = {"candidates": [{"finishReason": "STOP", "content": {"role": "model", "parts": [
    {"text": json.dumps({"status": "insufficient", "claims": []})}]}}]}


def answered(request):
    return httpx2.Response(200, json=ANSWER)


def outage(request):
    return httpx2.Response(503, json={"error": {"code": 503, "message": "synthetic outage"}})


def interrupt(request):
    raise KeyboardInterrupt  # the process dies after the request may have reached the provider


@pytest.fixture
def provider(monkeypatch):
    """Replies are chosen per question ID. ``sent`` lists the question ID of every request in order."""
    monkeypatch.setenv("GEMINI_API_KEY", "fixture-credential")
    monkeypatch.setenv("DOCGROUND_FREE_TIER_CONFIRMED", "true")
    monkeypatch.setenv("GOOGLE_GENAI_HTTP_CLIENT", "httpx2")
    state = SimpleNamespace(sent=[], replies={})

    def respond(_transport, request):
        text = json.loads(request.content)["contents"][0]["parts"][0]["text"]
        qid = BY_TEXT[json.loads(text)["question"]]
        state.sent.append(qid)
        return state.replies.get(qid, answered)(request)

    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", respond)
    return state


def collect(journal, *, header=HEADER, ids=IDS, **mode):
    run_attempts(journal, header, ids, lambda qid: evaluate_item(QUESTIONS[qid], SEARCH, Gemini(attempts=1)), **mode)


def statuses(journal):
    return {qid: [attempt["status"] for attempt in attempts] for qid, attempts in read_journal(journal)[1].items()}


def test_outage_then_resume_sends_only_unattempted_questions(provider, tmp_path):
    journal = tmp_path / "attempts.jsonl"
    provider.replies["q2"] = outage
    collect(journal)
    assert provider.sent == ["q1", "q2"]
    assert statuses(journal) == {"q1": ["insufficient"], "q2": ["error"]}

    provider.sent.clear()
    collect(journal, resume=True)
    assert provider.sent == ["q3", "q4"]
    # Resume leaves the recorded outage alone. The explicit retry policy still governs it.
    assert statuses(journal) == {"q1": ["insufficient"], "q2": ["error"], "q3": ["insufficient"],
                                 "q4": ["insufficient"]}

    provider.sent.clear()
    del provider.replies["q2"]
    collect(journal, retry_failed=True)
    assert provider.sent == ["q2"]
    assert statuses(journal)["q2"] == ["error", "insufficient"]


def test_repeated_resume_sends_nothing_and_leaves_the_journal_unchanged(provider, tmp_path):
    journal = tmp_path / "attempts.jsonl"
    provider.replies["q3"] = outage
    collect(journal)
    collect(journal, resume=True)
    assert provider.sent == ["q1", "q2", "q3", "q4"]
    finished = journal.read_bytes()
    for _ in range(2):
        collect(journal, resume=True)
    assert provider.sent == ["q1", "q2", "q3", "q4"] and journal.read_bytes() == finished


def test_resume_after_a_second_outage_stops_again_and_continues_later(provider, tmp_path):
    journal = tmp_path / "attempts.jsonl"
    provider.replies.update(q1=outage, q3=outage)
    collect(journal)
    collect(journal, resume=True)
    assert provider.sent == ["q1", "q2", "q3"]
    collect(journal, resume=True)
    assert provider.sent == ["q1", "q2", "q3", "q4"]


def test_interrupted_attempt_is_never_resent_by_resume_or_retry(provider, tmp_path):
    journal = tmp_path / "attempts.jsonl"
    provider.replies["q2"] = interrupt
    with pytest.raises(KeyboardInterrupt):
        collect(journal)
    assert provider.sent == ["q1", "q2"] and statuses(journal) == {"q1": ["insufficient"], "q2": ["interrupted"]}
    assert not journal.with_suffix(".lock").exists()

    del provider.replies["q2"]
    collect(journal, resume=True)
    collect(journal, retry_failed=True)
    assert provider.sent == ["q1", "q2", "q3", "q4"]
    assert statuses(journal)["q2"] == ["interrupted"]


@pytest.mark.parametrize(("header", "ids"), [
    pytest.param(HEADER | {"prompt_sha256": "changed"}, IDS, id="changed_prompt"),
    pytest.param(HEADER | {"git_commit": "other"}, IDS, id="changed_code_revision"),
    pytest.param(HEADER, IDS[1:], id="journal_holds_questions_outside_the_plan"),
])
def test_resume_rejects_a_changed_run_identity(provider, tmp_path, header, ids):
    journal = tmp_path / "attempts.jsonl"
    provider.replies["q2"] = outage
    collect(journal)
    before = journal.read_bytes()
    with pytest.raises(JournalError, match="identity"):
        collect(journal, header=header, ids=ids, resume=True)
    assert provider.sent == ["q1", "q2"] and journal.read_bytes() == before
    assert not journal.with_suffix(".lock").exists()


@pytest.mark.parametrize(("tail", "diagnostic"), [
    pytest.param(b'{"event":"start","id":"q3"}', "unterminated", id="unterminated_start"),
    pytest.param(b'{"event":"finish","id":"q3","result":{"status":"ans', "unterminated", id="torn_finish"),
    pytest.param(b'{"event":\n', "Invalid journal JSON", id="invalid_json_line"),
    pytest.param(b'{"event":"finish","id":"q4","result":{"status":"answered"}}\n', "Invalid attempt event",
                 id="finish_without_start"),
])
def test_resume_refuses_a_damaged_journal(provider, tmp_path, tail, diagnostic):
    journal = tmp_path / "attempts.jsonl"
    provider.replies["q2"] = outage
    collect(journal)
    damaged = journal.read_bytes() + tail
    journal.write_bytes(damaged)
    with pytest.raises(JournalError, match=diagnostic):
        collect(journal, resume=True)
    assert provider.sent == ["q1", "q2"] and journal.read_bytes() == damaged
    assert not journal.with_suffix(".lock").exists()


def test_resume_refuses_a_locked_journal_and_keeps_the_lock(provider, tmp_path):
    journal = tmp_path / "attempts.jsonl"
    provider.replies["q2"] = outage
    collect(journal)
    before = journal.read_bytes()
    lock = journal.with_suffix(".lock")
    lock.touch()
    with pytest.raises(JournalError, match="locked"):
        collect(journal, resume=True)
    assert provider.sent == ["q1", "q2"] and journal.read_bytes() == before and lock.is_file()


def test_resume_and_retry_cannot_be_combined(tmp_path):
    with pytest.raises(ValueError, match="either"):
        run_attempts(tmp_path / "attempts.jsonl", HEADER, IDS, pytest.fail, retry_failed=True, resume=True)
    assert not (tmp_path / "attempts.lock").exists()


@pytest.fixture
def collector(tmp_path, monkeypatch):
    """The real collect(), with synthetic provenance, retrieval and embedding file. Counts requests by kind."""
    monkeypatch.setenv("GEMINI_API_KEY", "fixture-credential")
    monkeypatch.setenv("DOCGROUND_FREE_TIER_CONFIRMED", "true")
    monkeypatch.setenv("GOOGLE_GENAI_HTTP_CLIENT", "httpx2")
    vectors = tmp_path / "vectors.npz"
    vectors.write_bytes(b"synthetic")
    monkeypatch.setattr(evaluation, "provenance", lambda root: {"git_commit": "synthetic"})
    monkeypatch.setattr(evaluation, "retriever", lambda method: SEARCH)
    monkeypatch.setattr(evaluation, "VECTORS", vectors)
    state = SimpleNamespace(metadata=0, generation=[], replies={})

    def respond(_transport, request):
        if request.method == "GET":
            state.metadata += 1
            return httpx2.Response(200, json={"name": "models/synthetic", "version": "1",
                                              "supportedGenerationMethods": ["generateContent"]})
        text = json.loads(request.content)["contents"][0]["parts"][0]["text"]
        qid = BY_TEXT[json.loads(text)["question"]]
        state.generation.append(qid)
        return state.replies.get(qid, answered)(request)

    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", respond)
    journal = tmp_path / "attempts.jsonl"
    state.run = lambda **mode: evaluation.collect(list(QUESTIONS.values()), tmp_path, journal, **mode)
    state.journal = journal
    return state


def test_complete_journal_resume_makes_only_the_metadata_request(collector):
    collector.run()
    assert (collector.metadata, collector.generation) == (1, IDS)
    finished = collector.journal.read_bytes()
    collector.run(resume=True)
    assert (collector.metadata, collector.generation) == (2, IDS)
    assert collector.journal.read_bytes() == finished


@pytest.mark.parametrize("damage", ["locked", "torn_tail", "changed_identity"])
def test_refused_continuation_follows_the_metadata_request_and_sends_no_generation(collector, monkeypatch, damage):
    collector.replies["q2"] = outage
    collector.run()
    assert (collector.metadata, collector.generation) == (1, ["q1", "q2"])
    if damage == "locked":
        collector.journal.with_suffix(".lock").touch()
    elif damage == "torn_tail":
        collector.journal.write_bytes(collector.journal.read_bytes() + b'{"event":"start","id":"q3"}')
    else:
        monkeypatch.setattr(evaluation, "PROMPT_VERSION", "synthetic-changed-prompt")
    before = collector.journal.read_bytes()
    with pytest.raises(JournalError):
        collector.run(resume=True)
    assert (collector.metadata, collector.generation) == (2, ["q1", "q2"])
    assert collector.journal.read_bytes() == before


@pytest.fixture
def cli(tmp_path, monkeypatch):
    """Run the collector CLI with synthetic inputs, recording how collect() was invoked."""
    calls = []
    monkeypatch.setattr(evaluation.dotenv, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluation, "LOCAL", tmp_path / "generation")
    monkeypatch.setattr(evaluation, "load_inputs", lambda root: ({"questions": list(QUESTIONS.values())}, None, None))
    monkeypatch.setattr(evaluation, "references", lambda root, questions: {qid: "" for qid in QUESTIONS})
    monkeypatch.setattr(evaluation, "collect", lambda *args, **kwargs: calls.append(kwargs))

    def run(*arguments):
        monkeypatch.setattr(sys, "argv", ["evaluate_generation", *arguments])
        with pytest.raises(SystemExit) as exit_info:
            evaluation.main()
        return exit_info.value.code, calls
    return run


@pytest.mark.parametrize("arguments", [
    pytest.param(("collect", "--resume", "--retry-failed"), id="resume_with_retry"),
    pytest.param(("score", "--resume"), id="resume_without_collect"),
    pytest.param(("collect", "--resume"), id="resume_without_journal"),
])
def test_cli_rejects_invalid_resume_requests_before_collecting(cli, arguments):
    code, calls = cli(*arguments)
    assert code == 2 and calls == []


def test_cli_resume_continues_the_existing_journal(cli, tmp_path):
    journal = tmp_path / "generation" / "attempts.jsonl"
    journal.parent.mkdir()
    run_attempts(journal, HEADER, IDS, lambda qid: {"status": "error", "retryable": True, "stop_run": True})
    code, calls = cli("collect", "--resume")
    # The partial journal still exits non-zero after collection, as for any incomplete run.
    assert code == 1 and calls == [{"retry_failed": False, "resume": True, "provider": "gemini"}]
