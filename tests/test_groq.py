"""Groq evaluation provider: one request per attempt, sanitized failures, pacing and an isolated journal.

Requests go through the real provider and collector into an in-memory httpx2 transport. A fake clock
replaces waiting, so the 75-second pacing is observed without sleeping.
"""
import json
import logging
import sys
from hashlib import sha256
from types import SimpleNamespace

import httpx2
import pytest

from scripts import evaluate_generation as evaluation
from src import groq
from src.checkpoints import read_journal, run_attempts
from src.gemini import RESPONSE_SCHEMA_SHA256, _RESPONSE_SCHEMA
from src.generation import (PROMPT_VERSION, SYSTEM, ContentBlocked, EmptyResponse, GenerationError,
                            MalformedResponse, ResponseTruncated)

KEY = "fixture-groq-credential"
SECRET = "provider-private-text"
ANSWER = json.dumps({"status": "insufficient", "claims": []})
MODELS = {"object": "list", "data": [
    {"id": "other/model", "object": "model", "active": True},
    {"id": groq.MODEL, "object": "model", "owned_by": "OpenAI", "active": True, "context_window": 131072,
     "max_completion_tokens": 65536, "public_apps": None}]}
QUESTIONS = {qid: {"id": qid, "question": f"Synthetic question {qid}?", "answerable": False, "evidence_units": []}
             for qid in ("q1", "q2", "q3")}
BY_TEXT = {q["question"]: qid for qid, q in QUESTIONS.items()}
HIT = {"id": "chunk", "document_id": "revision", "section_id": "section", "heading_path": ["Synthetic"],
       "text": "Synthetic documentation evidence."}
SEARCH = SimpleNamespace(search=lambda question, top_k: [HIT])


def completion(content=ANSWER, finish="stop"):
    return {"id": "synthetic", "object": "chat.completion", "model": groq.MODEL,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 9, "total_tokens": 49,
                      "completion_tokens_details": {"reasoning_tokens": 5}}}


def ok(request):
    return httpx2.Response(200, json=completion())


def status(code, error_code="synthetic_code"):
    return lambda request: httpx2.Response(code, json={"error": {"message": SECRET, "type": "synthetic",
                                                                  "code": error_code}})


@pytest.fixture
def wire(monkeypatch):
    """Fake clock and transport. ``replies`` maps question IDs to responders; ``reply`` is the default."""
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    monkeypatch.setenv("DOCGROUND_FREE_TIER_CONFIRMED", "true")
    state = SimpleNamespace(now=1000.0, sleeps=[], requests=[], generated=[], sent_at=[], journal_tail=[],
                            replies={}, reply=ok, journal=None,
                            models=lambda request: httpx2.Response(200, json=MODELS))

    def sleep(seconds):
        state.sleeps.append(seconds)
        state.now += seconds

    def handle(_transport, request):
        state.requests.append(request)
        if request.method == "GET":
            return state.models(request)
        user = json.loads(request.content)["messages"][1]["content"]
        qid = BY_TEXT.get(json.loads(user).get("question")) if user.startswith("{") else None
        state.generated.append(qid)
        state.sent_at.append(state.now)
        if state.journal is not None and state.journal.exists():  # what is on disk when the request leaves
            state.journal_tail.append(json.loads(state.journal.read_bytes().splitlines()[-1]))
        return state.replies.get(qid, state.reply)(request)

    state.sleep = sleep
    monkeypatch.setattr(groq, "_monotonic", lambda: state.now)
    monkeypatch.setattr(groq, "_sleep", sleep)
    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", handle)
    return state


def failure(wire, caplog, call):
    with caplog.at_level(logging.DEBUG, logger="src.groq"), pytest.raises(GenerationError) as info:
        call()
    exc = info.value
    assert SECRET not in str(exc) and SECRET not in caplog.text and KEY not in caplog.text
    # Raised outside the handler: no reference to the transport exception, response or request.
    assert exc.__context__ is None and exc.__cause__ is None
    return exc, [vars(record) for record in caplog.records if record.name == "src.groq"]


def test_request_sends_the_v3_prompt_and_strict_schema_once(wire):
    text, usage = groq.Groq().generate('{"question":"Synthetic?","evidence":[]}')
    request, = wire.requests
    assert request.method == "POST" and str(request.url) == "https://api.groq.com/openai/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    body = json.loads(request.content)
    assert body == {"model": "openai/gpt-oss-20b", "temperature": 0, "max_completion_tokens": 4096,
                    "messages": [{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": '{"question":"Synthetic?","evidence":[]}'}],
                    "response_format": {"type": "json_schema", "json_schema": {
                        "name": "grounded_answer", "strict": True, "schema": _RESPONSE_SCHEMA}}}
    schema = body["response_format"]["json_schema"]["schema"]
    assert sha256(json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                  .encode()).hexdigest() == RESPONSE_SCHEMA_SHA256
    assert text == ANSWER
    assert usage == {"prompt_token_count": 40, "candidates_token_count": 9, "total_token_count": 49,
                     "thoughts_token_count": 5}


@pytest.mark.parametrize("missing", ["GROQ_API_KEY", "DOCGROUND_FREE_TIER_CONFIRMED"])
def test_free_tier_gate_blocks_before_any_request_or_wait(wire, monkeypatch, missing):
    monkeypatch.delenv(missing)
    provider = groq.Groq()
    for call in (provider.preflight, lambda: provider.generate("{}"), lambda: provider.generate("{}")):
        with pytest.raises(GenerationError, match="offline"):
            call()
    assert wire.requests == [] and wire.sleeps == []


@pytest.mark.parametrize(("code", "error_code", "kind", "retryable"), [
    (401, None, GenerationError, False), (403, None, GenerationError, False),
    (429, "rate_limit_exceeded", GenerationError, True), (500, None, GenerationError, True),
    (502, None, GenerationError, True), (503, None, GenerationError, True), (504, None, GenerationError, True),
    (400, "invalid_request_error", GenerationError, False), (404, "model_not_found", GenerationError, False),
    (413, "request_too_large", GenerationError, False), (302, None, GenerationError, False),
    (400, "json_validate_failed", MalformedResponse, False),
])
def test_http_statuses_map_to_sanitized_errors_with_one_request(wire, caplog, code, error_code, kind, retryable):
    wire.reply = status(code, error_code)
    exc, records = failure(wire, caplog, lambda: groq.Groq().generate("{}"))
    assert type(exc) is kind and exc.retryable is retryable
    # Outages and configuration faults stop collection. A schema failure is final for that question only.
    assert exc.stops_collection is (kind is GenerationError)
    assert len(wire.requests) == 1 and records[-1]["http_status"] == code


@pytest.mark.parametrize(("error", "retryable"), [
    (httpx2.ConnectTimeout, True), (httpx2.ReadTimeout, True), (httpx2.ConnectError, True), (httpx2.ReadError, True),
    (httpx2.RemoteProtocolError, True), (httpx2.ProxyError, True), (OSError, True),
    (httpx2.LocalProtocolError, False), (httpx2.UnsupportedProtocol, False), (httpx2.DecodingError, False),
])
def test_transport_failures_send_once_and_only_transient_ones_are_retryable(wire, caplog, error, retryable):
    def fail(request):
        raise error(SECRET)

    wire.reply = fail
    exc, records = failure(wire, caplog, lambda: groq.Groq().generate("{}"))
    assert type(exc) is GenerationError and exc.retryable is retryable and exc.stops_collection
    assert len(wire.requests) == 1 and records[-1]["error_type"] == error.__name__


@pytest.mark.parametrize("body", [
    b"not json", b"[]", b"7", b'"text"', b"null", b"{}", json.dumps({"choices": []}).encode(),
    json.dumps({"choices": ["text"]}).encode(), json.dumps({"choices": [{"finish_reason": "stop"}]}).encode(),
    json.dumps({"choices": [{"message": SECRET, "finish_reason": "stop"}]}).encode(),
])
def test_malformed_success_envelopes_are_nonretryable_provider_errors(wire, caplog, body):
    wire.reply = lambda request: httpx2.Response(200, content=body)
    exc, _ = failure(wire, caplog, lambda: groq.Groq().generate("{}"))
    assert type(exc) is GenerationError and not exc.retryable and exc.stops_collection
    assert len(wire.requests) == 1


@pytest.mark.parametrize(("content", "finish", "envelope"), [
    pytest.param(ANSWER, ["stop"], "finish_reason", id="finish_list"),
    pytest.param(ANSWER, {"reason": SECRET}, "finish_reason", id="finish_object"),
    pytest.param(ANSWER, 1, "finish_reason", id="finish_number"),
    pytest.param(ANSWER, False, "finish_reason", id="finish_boolean"),
    pytest.param([ANSWER], "stop", "content", id="content_list"),
    pytest.param({"text": SECRET}, "stop", "content", id="content_object"),
    pytest.param(7, "stop", "content", id="content_number"),
    pytest.param(True, "stop", "content", id="content_boolean"),
    pytest.param([], "length", "content", id="content_type_checked_before_finish"),
])
def test_wrongly_typed_completion_fields_are_nonretryable_provider_errors(wire, caplog, content, finish, envelope):
    wire.reply = lambda request: httpx2.Response(200, json=completion(content, finish))
    exc, records = failure(wire, caplog, lambda: groq.Groq().generate("{}"))
    assert type(exc) is GenerationError and exc.code == "provider_error"
    assert not exc.retryable and exc.stops_collection and len(wire.requests) == 1
    assert records[-1]["envelope"] == envelope


@pytest.mark.parametrize(("content", "finish", "kind", "label"), [
    ('{"status":"ans', "length", ResponseTruncated, "length"),
    (SECRET, "content_filter", ContentBlocked, "content_filter"),
    ("", "stop", EmptyResponse, "stop"),
    (None, "stop", EmptyResponse, "stop"),
    (None, "tool_calls", EmptyResponse, "tool_calls"),
    (None, None, EmptyResponse, None),
    ("", SECRET, EmptyResponse, "unknown"),
])
def test_per_question_outcomes_are_final_and_let_collection_continue(wire, caplog, content, finish, kind, label):
    wire.reply = lambda request: httpx2.Response(200, json=completion(content, finish))
    exc, records = failure(wire, caplog, lambda: groq.Groq().generate("{}"))
    assert type(exc) is kind and not exc.retryable and not exc.stops_collection
    assert records[-1]["finish_reason"] == label


def test_unknown_finish_reason_string_with_answer_text_is_an_answer(wire, caplog):
    wire.reply = lambda request: httpx2.Response(200, json=completion(ANSWER, SECRET))
    with caplog.at_level(logging.DEBUG, logger="src.groq"):
        text, _ = groq.Groq().generate("{}")
    assert text == ANSWER and SECRET not in caplog.text


def test_preflight_lists_models_once_and_records_bounded_details(wire):
    provider = groq.Groq()
    assert provider.preflight() == {"id": groq.MODEL, "owned_by": "OpenAI", "context_window": 131072,
                                    "max_completion_tokens": 65536}
    request, = wire.requests
    assert request.method == "GET" and str(request.url) == "https://api.groq.com/openai/v1/models"
    # The metadata request is not paced and does not delay the first generation request.
    provider.generate("{}")
    assert wire.sleeps == []


@pytest.mark.parametrize(("reply", "retryable"), [
    (lambda request: httpx2.Response(200, json={"data": [{"id": "other/model"}]}), False),
    (lambda request: httpx2.Response(200, json={"data": [{"id": groq.MODEL, "active": False}]}), False),
    (lambda request: httpx2.Response(200, json={"data": SECRET}), False),
    (status(503), True),
])
def test_preflight_rejects_an_unavailable_model_or_unreadable_listing(wire, caplog, reply, retryable):
    wire.models = reply
    exc, _ = failure(wire, caplog, groq.Groq().preflight)
    assert type(exc) is GenerationError and exc.retryable is retryable and len(wire.requests) == 1


def test_generation_requests_start_at_least_75_seconds_apart(wire):
    provider = groq.Groq()
    provider.generate("{}")
    wire.now += 10
    provider.generate("{}")
    assert wire.sleeps == [65.0]
    wire.now += 80
    provider.generate("{}")
    assert wire.sleeps == [65.0] and len(wire.requests) == 3


def test_pacing_counts_failed_starts_and_restarts_with_each_invocation(wire):
    wire.reply = status(503)
    provider = groq.Groq()
    with pytest.raises(GenerationError):
        provider.generate("{}")
    wire.reply = ok
    wire.now += 30
    provider.generate("{}")
    assert wire.sleeps == [45.0]
    # A separate command builds a new provider, so the operator waits before starting it (see the protocol doc).
    groq.Groq().generate("{}")
    assert wire.sleeps == [45.0]


@pytest.fixture
def collector(wire, tmp_path, monkeypatch):
    vectors = tmp_path / "vectors.npz"
    vectors.write_bytes(b"synthetic")
    monkeypatch.setattr(evaluation, "provenance", lambda root: {"git_commit": "synthetic"})
    monkeypatch.setattr(evaluation, "retriever", lambda method: SEARCH)
    monkeypatch.setattr(evaluation, "VECTORS", vectors)
    wire.journal = tmp_path / "groq" / "attempts.jsonl"
    wire.journal.parent.mkdir()
    wire.run = lambda **mode: evaluation.collect(list(QUESTIONS.values()), tmp_path, wire.journal,
                                                 provider="groq", **mode)
    return wire


def outcomes(journal):
    return {qid: [a.get("error_code", a["status"]) for a in attempts] for qid, attempts in read_journal(journal)[1].items()}


def test_groq_collection_records_its_own_identity_and_paces_each_question(collector):
    collector.run()
    header, history = read_journal(collector.journal)
    assert header["provider"] == "groq" and header["model"] == "openai/gpt-oss-20b"
    assert header["model_details"]["id"] == "openai/gpt-oss-20b"
    assert header["provider_endpoint"] == "https://api.groq.com/openai/v1"
    assert header["request_format"] == groq.REQUEST_FORMAT
    assert (header["min_request_interval_s"], header["timeout_s"]) == (75.0, 60.0)
    assert (header["prompt_version"], header["prompt_sha256"], header["response_schema_sha256"]) == (
        PROMPT_VERSION, sha256(SYSTEM.encode()).hexdigest(), RESPONSE_SCHEMA_SHA256)
    assert (header["method"], header["top_k"], header["depth"], header["rrf_constant"]) == ("hybrid", 10, 20, 60)
    assert (header["temperature"], header["max_output_tokens"], header["context_bytes"]) == (0, 4096, 16000)
    assert (header["confirm_free_tier"], header["max_attempts_per_question"]) == (True, 2)
    # One metadata request, then one generation request per question, 75 s apart on the fake clock.
    assert [r.method for r in collector.requests] == ["GET", "POST", "POST", "POST"]
    assert collector.generated == ["q1", "q2", "q3"] and collector.sleeps == [75.0, 75.0]
    assert all(history[qid][0]["usage"]["total_token_count"] == 49 for qid in QUESTIONS)


def test_pacing_waits_before_the_start_checkpoint_and_outside_measured_latency(collector, monkeypatch):
    retrieval_s = {"q1": 5.0, "q2": 0.0, "q3": 2.0}

    def search(question, top_k):
        collector.now += retrieval_s[BY_TEXT[question]]
        return [HIT]

    monkeypatch.setattr(evaluation, "retriever", lambda method: SimpleNamespace(search=search))
    monkeypatch.setattr(evaluation, "perf_counter", lambda: collector.now)  # attempt timing on the same fake clock
    collector.run()
    _, history = read_journal(collector.journal)
    rows = [history[qid][0] for qid in QUESTIONS]
    assert collector.sleeps == [75.0, 75.0]
    # Requests start at least 75 s apart even when an earlier question's retrieval was slower.
    assert [b - a for a, b in zip(collector.sent_at, collector.sent_at[1:])] == [75.0, 77.0]
    # The waits fall before each attempt starts, so recorded latency is retrieval and generation only.
    assert [row["end_to_end_ms"] for row in rows] == [5000.0, 0.0, 2000.0]
    assert [row["generation_validation_ms"] for row in rows] == [0.0, 0.0, 0.0]
    # Each start checkpoint is on disk before its generation request leaves.
    assert collector.journal_tail == [{"event": "start", "id": qid} for qid in QUESTIONS]


def test_interrupted_pacing_wait_leaves_the_question_unattempted_for_resume(collector, monkeypatch):
    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(groq, "_sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        collector.run()
    # q2 was interrupted while waiting, before its start checkpoint and before any request.
    assert collector.generated == ["q1"] and outcomes(collector.journal) == {"q1": ["insufficient"]}
    assert not collector.journal.with_suffix(".lock").exists()
    monkeypatch.setattr(groq, "_sleep", collector.sleep)
    collector.run(resume=True)
    assert collector.generated == ["q1", "q2", "q3"]
    assert outcomes(collector.journal) == {qid: ["insufficient"] for qid in QUESTIONS}


def test_groq_outage_stops_then_resume_and_one_explicit_retry_follow_the_policy(collector):
    collector.replies["q2"] = status(503)
    collector.run()
    assert outcomes(collector.journal) == {"q1": ["insufficient"], "q2": ["provider_error"]}
    collector.run(resume=True)
    assert collector.generated == ["q1", "q2", "q3"]
    collector.replies["q2"] = ok
    collector.run(retry_failed=True)
    assert collector.generated == ["q1", "q2", "q3", "q2"]
    finished = collector.journal.read_bytes()
    collector.run(retry_failed=True)
    collector.run(resume=True)
    assert collector.generated == ["q1", "q2", "q3", "q2"] and collector.journal.read_bytes() == finished
    assert outcomes(collector.journal)["q2"] == ["provider_error", "insufficient"]


@pytest.mark.parametrize(("reply", "sent", "expected"), [
    pytest.param(status(401), ["q1"], {"q1": ["provider_error"]}, id="authentication_stops_and_is_final"),
    pytest.param(status(400, "json_validate_failed"), ["q1", "q2", "q3"],
                 {"q1": ["malformed_response"], "q2": ["insufficient"], "q3": ["insufficient"]},
                 id="schema_failure_is_final_and_continues"),
    pytest.param(lambda request: httpx2.Response(200, content=b"[]"), ["q1"], {"q1": ["provider_error"]},
                 id="malformed_envelope_stops_and_is_final"),
    pytest.param(lambda request: httpx2.Response(200, json=completion(ANSWER, ["stop"])), ["q1"],
                 {"q1": ["provider_error"]}, id="list_finish_reason_stops_and_is_final"),
    pytest.param(lambda request: httpx2.Response(200, json=completion({"text": ANSWER})), ["q1"],
                 {"q1": ["provider_error"]}, id="object_content_stops_and_is_final"),
])
def test_deterministic_failures_are_never_retried(collector, reply, sent, expected):
    collector.replies["q1"] = reply
    collector.run()
    collector.replies.pop("q1")
    collector.run(retry_failed=True)
    assert collector.generated == sent and outcomes(collector.journal) == expected


def test_gemini_collection_header_keeps_its_original_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fixture-credential")
    monkeypatch.setenv("DOCGROUND_FREE_TIER_CONFIRMED", "true")
    monkeypatch.setenv("GOOGLE_GENAI_HTTP_CLIENT", "httpx2")
    vectors = tmp_path / "vectors.npz"
    vectors.write_bytes(b"synthetic")
    monkeypatch.setattr(evaluation, "provenance", lambda root: {"git_commit": "synthetic"})
    monkeypatch.setattr(evaluation, "retriever", lambda method: SEARCH)
    monkeypatch.setattr(evaluation, "VECTORS", vectors)

    def respond(_transport, request):
        if request.method == "GET":
            return httpx2.Response(200, json={"name": "models/synthetic", "version": "1",
                                              "supportedGenerationMethods": ["generateContent"]})
        return httpx2.Response(200, json={"candidates": [{"finishReason": "STOP", "content": {
            "role": "model", "parts": [{"text": ANSWER}]}}]})

    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", respond)
    journal = tmp_path / "attempts.jsonl"
    evaluation.collect([QUESTIONS["q1"]], tmp_path, journal)
    header, _ = read_journal(journal)
    assert set(header) == {
        "git_commit", "run_schema_version", "prompt_version", "prompt_sha256", "response_schema_sha256", "model",
        "model_details", "method", "top_k", "depth", "rrf_constant", "input_sha256", "reference_sha256",
        "embedding_sha256", "temperature", "max_output_tokens", "context_bytes", "confirm_free_tier",
        "max_attempts_per_question"}


@pytest.fixture
def cli(collector, tmp_path, monkeypatch, capsys):
    """The real CLI and collector with synthetic inputs. Seeds a historical Gemini journal and review files."""
    monkeypatch.setattr(evaluation.dotenv, "load_dotenv", lambda *args, **kwargs: None)
    local = tmp_path / "generation"
    monkeypatch.setattr(evaluation, "LOCAL", local)
    monkeypatch.setattr(evaluation, "load_inputs", lambda root: ({"questions": list(QUESTIONS.values())}, None, None))
    monkeypatch.setattr(evaluation, "references", lambda root, questions: {qid: "" for qid in QUESTIONS})
    local.mkdir()
    run_attempts(local / "attempts.jsonl", {"model": "gemini-synthetic"}, ["q1"],
                 lambda qid: {"status": "error", "error_code": "provider_error", "retryable": True, "stop_run": True})
    (local / "review-template.json").write_text('{"historical": true}', encoding="utf-8")
    (local / "review.json").write_text('{"historical": true}', encoding="utf-8")
    collector.gemini = {path.name: path.read_bytes() for path in local.iterdir()}
    collector.local = local
    collector.journal = local / "groq" / "attempts.jsonl"

    def run(*arguments):
        monkeypatch.setattr(sys, "argv", ["evaluate_generation", *arguments])
        code = 0
        try:
            evaluation.main()
        except SystemExit as exc:
            code = exc.code
        return code, capsys.readouterr()
    collector.cli = run
    return collector


def test_groq_cli_collects_and_scores_in_its_own_directory(cli):
    code, _ = cli.cli("collect", "--provider", "groq")
    groq_dir = cli.local / "groq"
    journal = groq_dir / "attempts.jsonl"
    template = json.loads((groq_dir / "review-template.json").read_text(encoding="utf-8"))
    assert code == 0 and template["run_sha256"] == sha256(journal.read_bytes()).hexdigest()
    assert read_journal(journal)[0]["provider"] == "groq"
    (groq_dir / "review.json").write_text(json.dumps({"run_sha256": template["run_sha256"], "judgments": {}}),
                                          encoding="utf-8")
    code, output = cli.cli("score", "--provider", "groq")
    metrics = json.loads(output.out)
    assert code == 0 and (metrics["question_count"], metrics["total_tokens"]) == (3, 147)
    # The historical Gemini journal and review files are untouched by Groq collection and scoring.
    assert {path.name: path.read_bytes() for path in cli.local.iterdir() if path.is_file()} == cli.gemini


def test_groq_review_must_match_the_groq_journal_digest(cli):
    cli.cli("collect", "--provider", "groq")
    (cli.local / "groq" / "review.json").write_text(json.dumps({"run_sha256": "0" * 64, "judgments": {}}),
                                                    encoding="utf-8")
    code, output = cli.cli("score", "--provider", "groq")
    assert code == 2 and "journal digest" in output.err


@pytest.mark.parametrize("arguments", [
    pytest.param(("collect", "--provider", "groq", "--resume"), id="resume_without_groq_journal"),
    pytest.param(("collect", "--provider", "groq", "--retry-failed"), id="retry_without_groq_journal"),
    pytest.param(("score", "--provider", "groq"), id="score_without_groq_journal"),
    pytest.param(("references", "--provider", "groq"), id="references_with_provider"),
])
def test_a_gemini_journal_never_stands_in_for_a_groq_journal(cli, arguments):
    code, _ = cli.cli(*arguments)
    assert code in (1, 2) and cli.requests == []
    assert not (cli.local / "groq" / "attempts.jsonl").exists()
    assert {path.name: path.read_bytes() for path in cli.local.iterdir() if path.is_file()} == cli.gemini


@pytest.mark.parametrize("provider", ["gemini", "groq"])
def test_scoring_refuses_a_journal_recorded_by_another_provider(cli, provider):
    other = {"gemini": {"provider": "groq", "model": groq.MODEL}, "groq": {"model": "gemini-synthetic"}}[provider]
    directory = evaluation.run_directory(provider)
    directory.mkdir(exist_ok=True)
    journal = directory / "attempts.jsonl"
    journal.unlink(missing_ok=True)
    run_attempts(journal, other, ["q1"], lambda qid: {"status": "insufficient", "end_to_end_ms": 1})
    (directory / "review.json").write_text(json.dumps({"run_sha256": sha256(journal.read_bytes()).hexdigest(),
                                                       "judgments": {}}), encoding="utf-8")
    code, output = cli.cli("score", "--provider", provider)
    assert code == 2 and "different provider" in output.err
