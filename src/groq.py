"""Groq provider for development evaluation only. The Streamlit app does not use it.

Sends the v3 system instruction and the strict response schema through Groq's OpenAI-compatible chat
completions endpoint over the pinned httpx2 transport. Messages and logs never contain provider response
text, request content or credentials.

| Outcome                                   | Code               | Retryable | Stops collection |
|-------------------------------------------|--------------------|-----------|------------------|
| HTTP 401, 403                             | provider_error     | no        | yes              |
| HTTP 429, 500, 502, 503, 504              | provider_error     | yes       | yes              |
| HTTP 400 with error code                  | malformed_response | no        | no               |
| ``json_validate_failed``                  |                    |           |                  |
| Other HTTP errors, including 413          | provider_error     | no        | yes              |
| Timeout, network or proxy failure, server | provider_error     | yes       | yes              |
| broke the HTTP exchange                   |                    |           |                  |
| Local request error: protocol misuse,     | provider_error     | no        | yes              |
| unsupported scheme                        |                    |           |                  |
| Unreadable or malformed success response  | provider_error     | no        | yes              |
| ``finish_reason`` ``content_filter``      | content_blocked    | no        | no               |
| Well-formed response without answer text  | empty_response     | no        | no               |
| ``finish_reason`` ``length``              | response_truncated | no        | no               |

A success response is malformed when its body is not a JSON object whose first ``choices`` entry holds a
``message`` object, when ``finish_reason`` is neither a string nor null, or when ``message.content`` is
neither a string nor null. A null or empty ``content`` is a genuine empty response. Each call sends one HTTP
request: the client neither retries nor follows redirects. Generation requests from one provider instance
start at least ``MIN_INTERVAL_S`` seconds apart, which keeps one collection invocation under the Free Plan
per-minute token limit. The collector waits through ``wait_for_slot`` before writing each start checkpoint,
outside the timed attempt. The model metadata request is not paced.
"""
import logging
import os
import time

import httpx2

from src.gemini import MAX_OUTPUT_TOKENS, TRANSIENT_HTTP, TRANSIENT_TRANSPORT, _RESPONSE_SCHEMA
from src.generation import (ContentBlocked, EmptyResponse, GenerationError, MalformedResponse, ResponseTruncated,
                            SYSTEM)

MODEL = "openai/gpt-oss-20b"
BASE_URL = "https://api.groq.com/openai/v1"
REQUEST_FORMAT = "openai-chat-completions-json-schema-strict-v1"
MIN_INTERVAL_S = 75.0
TIMEOUT_S = 60.0
LOG = logging.getLogger(__name__)
TRANSPORT_MESSAGE = "Groq is unavailable or the request timed out. Try again shortly."
LOCAL_REQUEST_MESSAGE = "The Groq request could not be sent."
UNREADABLE_MESSAGE = "Groq returned an unreadable response."
KNOWN_FINISH = frozenset({"stop", "length", "content_filter", "tool_calls", "function_call"})
# Looked up at call time so tests can substitute a fake clock.
_monotonic, _sleep = time.monotonic, time.sleep


def _logged(error: GenerationError, **details) -> GenerationError:
    """Record a sanitized failure: class names, status codes and known labels only."""
    LOG.warning("Groq request failed", extra={"model_id": MODEL, "error_code": error.code,
                                               "retryable": error.retryable, **details})
    return error


def _error_code(response: httpx2.Response) -> str | None:
    """The provider's machine-readable error code, compared but never logged or returned to users."""
    try:
        body = response.json()
    except ValueError:
        return None
    error = body.get("error") if isinstance(body, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, str) else None


def _status_error(status: int, code: str | None) -> GenerationError:
    if status == 400 and code == "json_validate_failed":
        # The model's output failed the schema: a final outcome for this question, like a local parse failure.
        return MalformedResponse("The model returned a response in an unexpected format.")
    if status == 401:
        message = "Groq authentication failed. Check the configured API key."
    elif status == 403:
        message = "Groq access denied for the configured key or model."
    elif status == 429:
        message = "Groq rate limit reached. Wait before continuing."
    elif status in TRANSIENT_HTTP:
        message = "Groq is temporarily unavailable. Try again shortly."
    else:
        message = "Groq rejected the request."
    return GenerationError(message, retryable=status in TRANSIENT_HTTP)


def _usage(raw) -> dict:
    """Token counts under the keys the scorer reads, integers only."""
    raw = raw if isinstance(raw, dict) else {}
    details = raw.get("completion_tokens_details")
    details = details if isinstance(details, dict) else {}
    counts = {"prompt_token_count": raw.get("prompt_tokens"), "candidates_token_count": raw.get("completion_tokens"),
              "total_token_count": raw.get("total_tokens"), "thoughts_token_count": details.get("reasoning_tokens")}
    return {key: value for key, value in counts.items() if isinstance(value, int) and not isinstance(value, bool)}


def request_body(contents: str) -> dict:
    return {"model": MODEL, "temperature": 0, "max_completion_tokens": MAX_OUTPUT_TOKENS,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": contents}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "grounded_answer", "strict": True, "schema": _RESPONSE_SCHEMA}}}


class Groq:
    model = MODEL

    def __init__(self, *, min_interval: float = MIN_INTERVAL_S):
        self.min_interval = min_interval
        self._last_start = None

    def _headers(self) -> dict:
        # Reads the process environment only. The collector loads any local .env file.
        key = os.getenv("GROQ_API_KEY")
        if not key or os.getenv("DOCGROUND_FREE_TIER_CONFIRMED") != "true":
            raise GenerationError("Groq is offline. Configure a verified free-plan project and API key.")
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def wait_for_slot(self) -> None:
        """Sleep until ``min_interval`` has passed since the previous generation request started.

        The collector calls this before writing an attempt's start checkpoint, so the wait is neither
        timed as latency nor recorded as an attempt. ``generate`` repeats the check as a guard.
        """
        if self._last_start is not None:
            wait = self._last_start + self.min_interval - _monotonic()
            if wait > 0:
                _sleep(wait)

    def _send(self, method: str, path: str, *, paced: bool = False, **kwargs) -> dict:
        """Send one request and return its JSON object. Errors are raised outside the handlers, so they hold
        no reference to the transport exception, the response or the request."""
        headers = self._headers()
        if paced:
            self.wait_for_slot()
            self._last_start = _monotonic()
        failure = payload = None
        try:
            with httpx2.Client(base_url=BASE_URL, timeout=TIMEOUT_S, follow_redirects=False) as client:
                response = client.request(method, path, headers=headers, **kwargs)
        except TRANSIENT_TRANSPORT as exc:
            failure = _logged(GenerationError(TRANSPORT_MESSAGE, retryable=True), error_type=type(exc).__name__)
        except httpx2.DecodingError as exc:
            failure = _logged(GenerationError(UNREADABLE_MESSAGE), error_type=type(exc).__name__)
        except httpx2.RequestError as exc:
            failure = _logged(GenerationError(LOCAL_REQUEST_MESSAGE), error_type=type(exc).__name__)
        except OSError as exc:
            failure = _logged(GenerationError(TRANSPORT_MESSAGE, retryable=True), error_type=type(exc).__name__)
        if failure is None and response.status_code != 200:
            failure = _logged(_status_error(response.status_code, _error_code(response)),
                              http_status=response.status_code)
        if failure is None:
            try:
                payload = response.json()
            except ValueError as exc:
                failure = _logged(GenerationError(UNREADABLE_MESSAGE), error_type=type(exc).__name__)
        if failure is None and not isinstance(payload, dict):
            failure = _logged(GenerationError(UNREADABLE_MESSAGE), envelope="not_object")
        if failure is not None:
            raise failure
        return payload

    def preflight(self) -> dict:
        """Verify the model is listed for this key without sending any prompt."""
        models = self._send("GET", "/models").get("data")
        if not isinstance(models, list):
            raise _logged(GenerationError(UNREADABLE_MESSAGE), envelope="models")
        entry = next((item for item in models if isinstance(item, dict) and item.get("id") == MODEL), None)
        if entry is None or entry.get("active") is False:
            raise _logged(GenerationError("Configured Groq model is not available to this key."), model_state="absent")
        return {key: entry[key] for key in ("id", "owned_by", "context_window", "max_completion_tokens")
                if isinstance(entry.get(key), (str, int)) and not isinstance(entry.get(key), bool)}

    def generate(self, contents: str) -> tuple[str, dict]:
        payload = self._send("POST", "/chat/completions", paced=True, json=request_body(contents))
        choices = payload.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else None
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict):
            raise _logged(GenerationError(UNREADABLE_MESSAGE), envelope="missing")
        finish, text = choice.get("finish_reason"), message.get("content")
        # Types are checked before use. A wrong type is a malformed envelope, not a model answer.
        if finish is not None and not isinstance(finish, str):
            raise _logged(GenerationError(UNREADABLE_MESSAGE), envelope="finish_reason")
        if text is not None and not isinstance(text, str):
            raise _logged(GenerationError(UNREADABLE_MESSAGE), envelope="content")
        label = finish if finish in KNOWN_FINISH else (None if finish is None else "unknown")
        if finish == "length":
            raise _logged(ResponseTruncated("Groq reached its output limit. Try a narrower question."),
                          finish_reason=label)
        if finish == "content_filter":
            raise _logged(ContentBlocked("Groq declined to answer this request."), finish_reason=label)
        if not text:
            raise _logged(EmptyResponse("Groq returned no usable content."), finish_reason=label)
        usage = _usage(payload.get("usage"))
        LOG.info("Groq request completed", extra={"model_id": MODEL, "token_usage": usage})
        return text, usage
