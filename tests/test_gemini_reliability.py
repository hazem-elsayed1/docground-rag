"""Offline provider reliability: classification, user-visible messages, retry eligibility and request counts.

Every request is answered by the real google-genai SDK through an in-memory httpx2 transport, so the
SDK's own retry loop and response parsing run unchanged. Backoff sleeps are recorded, not waited.
"""
import json
import logging
from types import SimpleNamespace
import warnings

from google.genai import types
import httpx2
import pytest

from scripts.evaluate_generation import evaluate_item
from src import gemini as gemini_module
from src.checkpoints import read_journal, run_attempts
from src.gemini import Gemini
from src.generation import ContentBlocked, EmptyResponse, GenerationError

KEY = "fixture-credential-0123456789"
SECRET = "raw-provider-detail"
CONTENTS = "synthetic evidence payload"
ANSWER = {"candidates": [{"finishReason": "STOP", "content": {"role": "model", "parts": [
    {"text": json.dumps({"status": "insufficient", "claims": []})}]}}], "usageMetadata": {"totalTokenCount": 3}}


def status(code):
    return lambda request: httpx2.Response(code, json={"error": {"code": code, "message": SECRET, "status": "X"}})


def transport(error_type):
    def fail(request):
        raise error_type(SECRET, request=request)
    return fail


def body(payload):
    return lambda request: httpx2.Response(200, json=payload)


def raw(content):
    return lambda request: httpx2.Response(200, content=content.encode(), headers={"content-type": "application/json"})


UNREADABLE = lambda request: httpx2.Response(200, text=f"<html>{SECRET}</html>")  # noqa: E731
BLOCKED = body({"promptFeedback": {"blockReason": "SAFETY"}})
EMPTY = body({"candidates": [{"finishReason": "STOP", "content": {"role": "model", "parts": [{"text": ""}]}}]})


@pytest.fixture
def provider(monkeypatch):
    """Scripted replies in request order. The last reply repeats for any further request."""
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setenv("DOCGROUND_FREE_TIER_CONFIRMED", "true")
    monkeypatch.setenv("GOOGLE_GENAI_HTTP_CLIENT", "httpx2")
    state = SimpleNamespace(requests=[], script=[], sleeps=[])
    monkeypatch.setattr("tenacity.nap.time.sleep", state.sleeps.append)

    def respond(_transport, request):
        state.requests.append(request)
        return state.script[min(len(state.requests), len(state.script)) - 1](request)

    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", respond)
    return state


def _leaks(error, caplog):
    records = " ".join(str(vars(record)) for record in caplog.records)
    surface = f"{error} {error.__cause__} {error.__context__} {records}"
    return [marker for marker in (SECRET, KEY, CONTENTS) if marker in surface]


# id, reply, message fragment, retryable, requests when the application allows two attempts
FAILURES = [
    ("http_401", status(401), "authentication failed", False, 1),
    ("http_403", status(403), "access denied", False, 1),
    ("http_400", status(400), "rejected the request", False, 1),
    ("http_404", status(404), "rejected the request", False, 1),
    ("http_429", status(429), "quota or rate limit", True, 2),
    ("http_500", status(500), "temporarily unavailable", True, 2),
    ("http_502", status(502), "temporarily unavailable", True, 2),
    ("http_503", status(503), "temporarily unavailable", True, 2),
    ("http_504", status(504), "temporarily unavailable", True, 2),
    ("read_timeout", transport(httpx2.ReadTimeout), "timed out", True, 2),
    ("write_timeout", transport(httpx2.WriteTimeout), "timed out", True, 2),
    ("pool_timeout", transport(httpx2.PoolTimeout), "timed out", True, 2),
    ("connect_timeout", transport(httpx2.ConnectTimeout), "timed out", True, 2),
    ("connect_error", transport(httpx2.ConnectError), "timed out", True, 2),
    # Transient but outside the SDK's own retry predicate: one request, still eligible for a journal retry.
    ("read_error", transport(httpx2.ReadError), "timed out", True, 1),
    ("remote_protocol_error", transport(httpx2.RemoteProtocolError), "timed out", True, 1),
    ("proxy_error", transport(httpx2.ProxyError), "timed out", True, 1),
    # Deterministic local request errors are never repeated.
    ("local_protocol_error", transport(httpx2.LocalProtocolError), "could not be sent", False, 1),
    ("unsupported_protocol", transport(httpx2.UnsupportedProtocol), "could not be sent", False, 1),
    ("too_many_redirects", transport(httpx2.TooManyRedirects), "could not be sent", False, 1),
    ("decoding_error", transport(httpx2.DecodingError), "unreadable response", False, 1),
    ("unreadable_success", UNREADABLE, "unreadable response", False, 1),
]


@pytest.mark.parametrize("attempts", [1, 2], ids=["collector", "application"])
@pytest.mark.parametrize(("reply", "fragment", "retryable", "application_requests"),
                         [pytest.param(*case[1:], id=case[0]) for case in FAILURES])
def test_provider_failures_are_classified_sanitized_and_bounded(
        provider, caplog, attempts, reply, fragment, retryable, application_requests):
    provider.script.append(reply)
    caplog.set_level(logging.DEBUG, logger="src.gemini")
    with pytest.raises(GenerationError) as error:
        Gemini(attempts=attempts).generate(CONTENTS)
    exc = error.value
    assert type(exc) is GenerationError and exc.code == "provider_error" and exc.stops_collection
    assert fragment in str(exc) and exc.retryable is retryable
    # The collector makes exactly one SDK attempt. The application repeats only transient failures once.
    expected = application_requests if attempts == 2 else 1
    assert len(provider.requests) == expected and len(provider.sleeps) == expected - 1
    assert all(0 <= delay <= 4 for delay in provider.sleeps)
    assert _leaks(exc, caplog) == []
    failure, = [record for record in caplog.records if record.name == "src.gemini"]
    assert (failure.error_code, failure.retryable) == ("provider_error", retryable)


@pytest.mark.parametrize(("payload", "error_type"), [
    pytest.param({"promptFeedback": {"blockReason": "SAFETY"}}, ContentBlocked, id="prompt_blocked"),
    pytest.param({"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}, "candidates": []}, ContentBlocked,
                 id="prompt_prohibited"),
    pytest.param({"candidates": [{"finishReason": "SAFETY"}]}, ContentBlocked, id="answer_safety"),
    pytest.param({"candidates": [{"finishReason": "RECITATION", "content": {"role": "model", "parts": [
        {"text": '{"status"'}]}}]}, ContentBlocked, id="answer_recitation_with_partial_text"),
    pytest.param({"candidates": []}, EmptyResponse, id="no_candidates"),
    pytest.param({"candidates": [{"finishReason": "STOP", "content": {"role": "model", "parts": [{"text": ""}]}}]},
                 EmptyResponse, id="empty_text"),
    pytest.param({"candidates": [{"finishReason": "OTHER"}]}, EmptyResponse, id="other_without_text"),
    pytest.param({"promptFeedback": {}}, EmptyResponse, id="feedback_without_block"),
    pytest.param({"promptFeedback": {"blockReason": "BLOCKED_REASON_UNSPECIFIED"}}, EmptyResponse,
                 id="unspecified_block_without_answer"),
])
@pytest.mark.parametrize("attempts", [1, 2], ids=["collector", "application"])
def test_content_outcomes_are_final_and_never_resent(provider, caplog, attempts, payload, error_type):
    provider.script.append(body(payload))
    caplog.set_level(logging.DEBUG, logger="src.gemini")
    with pytest.raises(GenerationError) as error:
        Gemini(attempts=attempts).generate(CONTENTS)
    exc = error.value
    assert type(exc) is error_type and not exc.retryable and not exc.stops_collection
    assert len(provider.requests) == 1 and provider.sleeps == []
    assert _leaks(exc, caplog) == []


def test_unspecified_block_reason_does_not_block_a_valid_answer(provider):
    provider.script.append(body(ANSWER | {"promptFeedback": {"blockReason": "BLOCKED_REASON_UNSPECIFIED"}}))
    text, usage = Gemini(attempts=1).generate(CONTENTS)
    assert json.loads(text) == {"status": "insufficient", "claims": []} and usage == {"total_token_count": 3}
    assert len(provider.requests) == 1


# What the pinned SDK exposes for each body: arrays, empty or unrelated objects, strings and null parse
# into a response with no candidates and no prompt feedback; a number raises TypeError inside the SDK;
# a non-list candidates value becomes one candidate without content or finish reason.
MALFORMED = {
    "array_empty": "[]",
    "array_wrapping_valid_response": json.dumps([ANSWER]),
    "object_empty": "{}",
    "object_unrelated": json.dumps({"unexpected": SECRET}),
    "string": json.dumps(SECRET),
    "null": "null",
    "number": "7",
    "candidates_not_a_list": json.dumps({"candidates": SECRET}),
    "candidate_without_fields": json.dumps({"candidates": [{}]}),
}


@pytest.mark.parametrize("attempts", [1, 2], ids=["collector", "application"])
@pytest.mark.parametrize("content", [pytest.param(value, id=key) for key, value in MALFORMED.items()])
def test_malformed_success_envelopes_are_unreadable_provider_failures(provider, caplog, attempts, content):
    provider.script.append(raw(content))
    caplog.set_level(logging.DEBUG, logger="src.gemini")
    with pytest.raises(GenerationError, match="unreadable response") as error:
        Gemini(attempts=attempts).generate(CONTENTS)
    exc = error.value
    assert type(exc) is GenerationError and exc.code == "provider_error"
    assert exc.stops_collection and not exc.retryable
    assert len(provider.requests) == 1 and provider.sleeps == []
    assert _leaks(exc, caplog) == []


MARKER = "SYNTHETIC_UNKNOWN_REASON_7F3A"
ANSWER_CONTENT = ANSWER["candidates"][0]["content"]


@pytest.mark.parametrize(("payload", "outcome", "labels"), [
    pytest.param({"candidates": [{"finishReason": MARKER, "content": ANSWER_CONTENT}]}, None, None,
                 id="unknown_finish_with_answer"),
    pytest.param({"candidates": [{"finishReason": MARKER}]}, EmptyResponse,
                 {"finish_reason": "unknown", "block_reason": None}, id="unknown_finish_without_answer"),
    pytest.param({"promptFeedback": {"blockReason": MARKER}}, ContentBlocked,
                 {"finish_reason": None, "block_reason": "unknown"}, id="unknown_block"),
    pytest.param({"promptFeedback": {"blockReason": MARKER}, "candidates": [{"finishReason": MARKER}]},
                 ContentBlocked, {"finish_reason": "unknown", "block_reason": "unknown"}, id="unknown_both"),
    pytest.param({"promptFeedback": {"blockReason": "SAFETY"}}, ContentBlocked,
                 {"finish_reason": None, "block_reason": "SAFETY"}, id="known_block_keeps_its_value"),
])
def test_unknown_provider_reasons_stay_out_of_errors_logs_and_warnings(provider, caplog, payload, outcome, labels):
    provider.script.append(body(payload))
    caplog.set_level(logging.DEBUG)  # every logger, including the SDK's own
    error = None
    with warnings.catch_warnings(record=True) as caught:
        try:
            Gemini(attempts=1).generate(CONTENTS)
        except GenerationError as exc:
            error = exc
    assert type(error) is outcome if outcome else error is None
    surface = " ".join([str(error), *(str(w.message) for w in caught), *(str(vars(r)) for r in caplog.records)])
    assert MARKER not in surface
    if labels:
        record = [r for r in caplog.records if r.name == "src.gemini"][-1]
        assert {key: getattr(record, key) for key in labels} == labels


def test_pinned_sdk_echoes_unknown_enum_values_and_the_adapter_filter_matches_it():
    """Documents the SDK limitation the adapter works around. It fails if the SDK wording changes."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        types.FinishReason(MARKER)
    warning, = caught
    assert MARKER in str(warning.message) and warning.filename.endswith("_common.py")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings(**gemini_module._SDK_ENUM_WARNING)
        types.BlockedReason(MARKER)
    assert caught == []


def test_preflight_maps_authentication_failure_without_sending_a_prompt(provider):
    provider.script.append(status(401))
    with pytest.raises(GenerationError, match="authentication failed") as error:
        Gemini(attempts=1).preflight()
    assert not error.value.retryable and len(provider.requests) == 1
    assert provider.requests[0].method == "GET" and SECRET not in str(error.value)


HIT = {"id": "chunk", "document_id": "revision", "section_id": "section", "heading_path": ["Synthetic"],
       "text": "Synthetic documentation evidence."}
SEARCH = SimpleNamespace(search=lambda question, top_k: [HIT])


def _execute(questions):
    return lambda qid: evaluate_item(questions[qid], SEARCH, Gemini(attempts=1))


def _questions(*ids):
    return {qid: {"id": qid, "question": f"Synthetic question {qid}?", "answerable": False, "evidence_units": []}
            for qid in ids}


def test_collection_continues_after_content_block_and_stops_at_outage(provider, tmp_path):
    questions = _questions("blocked", "outage", "unattempted")
    provider.script.extend([BLOCKED, status(503), body(ANSWER)])
    journal = tmp_path / "attempts.jsonl"
    run_attempts(journal, {"run": "synthetic"}, list(questions), _execute(questions))
    _, history = read_journal(journal)
    assert list(history) == ["blocked", "outage"]
    (blocked,), (outage,) = history["blocked"], history["outage"]
    assert (blocked["error_code"], blocked["retryable"], blocked["stop_run"]) == ("content_blocked", False, False)
    assert (outage["error_code"], outage["retryable"], outage["stop_run"]) == ("provider_error", True, True)
    assert len(provider.requests) == 2

    # The explicit retry repeats only the eligible outage, once, under the same journal configuration.
    run_attempts(journal, {"run": "synthetic"}, list(questions), _execute(questions), retry_failed=True)
    _, history = read_journal(journal)
    assert [attempt["status"] for attempt in history["outage"]] == ["error", "insufficient"]
    assert len(history["blocked"]) == 1 and "unattempted" not in history
    assert len(provider.requests) == 3


@pytest.mark.parametrize(("reply", "eligible"), [
    pytest.param(status(401), False, id="http_401"),
    pytest.param(status(403), False, id="http_403"),
    pytest.param(status(400), False, id="http_400"),
    pytest.param(UNREADABLE, False, id="unreadable_success"),
    pytest.param(BLOCKED, False, id="content_blocked"),
    pytest.param(EMPTY, False, id="empty_response"),
    pytest.param(transport(httpx2.LocalProtocolError), False, id="local_protocol_error"),
    pytest.param(raw("[]"), False, id="malformed_array"),
    pytest.param(transport(httpx2.ReadError), True, id="read_error"),
    pytest.param(status(429), True, id="http_429"),
    pytest.param(status(500), True, id="http_500"),
    pytest.param(transport(httpx2.ReadTimeout), True, id="read_timeout"),
    pytest.param(transport(httpx2.ConnectError), True, id="connect_error"),
])
def test_explicit_retry_repeats_only_eligible_failures_and_only_once(provider, tmp_path, reply, eligible):
    questions = _questions("q")
    provider.script.append(reply)
    journal = tmp_path / "attempts.jsonl"
    run_attempts(journal, {"run": "synthetic"}, ["q"], _execute(questions))
    assert len(provider.requests) == 1
    for _ in range(2):
        run_attempts(journal, {"run": "synthetic"}, ["q"], _execute(questions), retry_failed=True)
        assert len(provider.requests) == (2 if eligible else 1)
    _, history = read_journal(journal)
    assert len(history["q"]) == (2 if eligible else 1)
