"""Google Gemini Developer API provider. Credentials never enter app output.

Failure handling. Messages never contain provider response text, request content or credentials.

| Outcome                                   | Code               | Retryable | Stops collection |
|-------------------------------------------|--------------------|-----------|------------------|
| HTTP 401, 403                             | provider_error     | no        | yes              |
| HTTP 429, 500, 502, 503, 504              | provider_error     | yes       | yes              |
| Other HTTP errors                         | provider_error     | no        | yes              |
| Timeout, network or proxy failure, server | provider_error     | yes       | yes              |
| broke the HTTP exchange                   |                    |           |                  |
| Local request error: protocol misuse,     | provider_error     | no        | yes              |
| unsupported scheme, redirect loop         |                    |           |                  |
| Unreadable or malformed success response  | provider_error     | no        | yes              |
| Prompt or answer content block            | content_blocked    | no        | no               |
| Well-formed response without answer text  | empty_response     | no        | no               |
| Output token limit                        | response_truncated | no        | no               |

A success response is malformed when its body is not a JSON object carrying ``candidates`` or
``promptFeedback``, or when its first candidate has neither content nor a finish reason. The pinned
SDK turns a non-object JSON body into an empty response object, or raises ``TypeError`` for a number.
``BLOCKED_REASON_UNSPECIFIED`` means no block. Log fields carry known enum values, class names or
``"unknown"``, never raw provider values. The SDK warns with the raw value of an unknown enum. That
warning is filtered before each call, and the bounded log label replaces it.

The SDK repeats a request only when ``attempts`` exceeds one. It then retries the listed transient
HTTP statuses and httpx timeouts and connection errors. The application uses two attempts and the
development collector one. Other eligible transport failures are sent once and stay eligible for an
explicit collection retry.
"""
from hashlib import sha256
import json
import logging
import os
import warnings
import httpx2
import pydantic
from google import genai
from google.genai import errors, types
from src.generation import (ContentBlocked, EmptyResponse, GenerationError, ProviderNotConfigured,
                            ResponseTruncated, SYSTEM)

DEFAULT_MODEL = "gemini-3.1-flash-lite"
LOG = logging.getLogger(__name__)
TRANSIENT_HTTP = (429, 500, 502, 503, 504)
MAX_OUTPUT_TOKENS = 4096
TIMEOUT_MS = 30000
# Candidate finish reasons that withhold answer text for content reasons.
BLOCKED_FINISH = frozenset({
    types.FinishReason.SAFETY, types.FinishReason.RECITATION, types.FinishReason.BLOCKLIST,
    types.FinishReason.PROHIBITED_CONTENT, types.FinishReason.SPII, types.FinishReason.MODEL_ARMOR,
})
NO_BLOCK = types.BlockedReason.BLOCKED_REASON_UNSPECIFIED
# Values the pinned SDK knows. An unknown provider value becomes a pseudo-member outside this set.
KNOWN_REASONS = {kind: frozenset(member.value for member in kind) for kind in (types.FinishReason, types.BlockedReason)}
# Transient transport failures. Errors that arise from building or sending this request locally,
# redirect loops and undecodable bodies are deterministic and stay ineligible for a retry.
TRANSIENT_TRANSPORT = (httpx2.TimeoutException, httpx2.NetworkError, httpx2.RemoteProtocolError, httpx2.ProxyError)
TRANSPORT_MESSAGE = "Gemini is unavailable or the request timed out. Try again shortly."
LOCAL_REQUEST_MESSAGE = "The Gemini request could not be sent."
UNREADABLE_MESSAGE = "Gemini returned an unreadable response."
# google.genai._common warns "<raw value> is not a valid <Enum>" when coercing an unknown value.
_SDK_ENUM_WARNING = {"action": "ignore", "message": r"(?s).* is not a valid [A-Za-z]+\Z",
                     "category": UserWarning, "module": r"google\.genai\._common\Z"}

# Nested array bounds inflate Gemini's decoding grammar. Response enforces them locally.
_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "claims"],
    "properties": {
        "status": {"type": "string", "enum": ["answered", "insufficient", "conflicting"]},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "kind", "citations"],
                "properties": {
                    "text": {"type": "string"},
                    "kind": {"type": "string", "enum": ["explicit", "inference"]},
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["chunk_id", "document_id", "quote"],
                            "properties": {
                                "chunk_id": {"type": "string"},
                                "document_id": {"type": "string"},
                                "quote": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
    },
}
# Identifies the structured-output schema sent with every request, alongside the system prompt digest.
RESPONSE_SCHEMA_SHA256 = sha256(json.dumps(_RESPONSE_SCHEMA, sort_keys=True, separators=(",", ":"),
                                           ensure_ascii=False).encode("utf-8")).hexdigest()


def _provider_error(exc: errors.APIError) -> GenerationError:
    if exc.code == 401:
        message = "Gemini authentication failed. Check the configured API key."
    elif exc.code == 403:
        message = "Gemini access denied for the configured key or model."
    elif exc.code == 429:
        message = "Gemini quota or rate limit reached. Try again later."
    elif exc.code in TRANSIENT_HTTP:
        message = "Gemini is temporarily unavailable. Try again shortly."
    else:
        message = "Gemini rejected the request."
    return GenerationError(message, retryable=exc.code in TRANSIENT_HTTP)


def _logged(error: GenerationError, model: str, **details) -> GenerationError:
    """Record a sanitized failure: class names and status codes only, never provider text."""
    LOG.warning("Gemini request failed", extra={"model_id": model, "error_code": error.code,
                                                 "retryable": error.retryable, **details})
    return error


def _reason_label(reason) -> str | None:
    """A known finish or block reason value, or "unknown". Raw provider values never reach logs."""
    if reason is None:
        return None
    known = KNOWN_REASONS.get(type(reason), frozenset())
    return reason.value if reason.value in known else "unknown"


def _unreadable(model: str, **details) -> GenerationError:
    # A successful status without a usable envelope is a provider fault, not a model answer.
    return _logged(GenerationError(UNREADABLE_MESSAGE), model, **details)


def _sanitized_failure(exc: Exception, model: str, transport_error: GenerationError) -> GenerationError:
    """Map an SDK or transport exception to a public error. Log fields hold class names and codes only."""
    error_type = type(exc).__name__
    if isinstance(exc, errors.APIError):
        return _logged(_provider_error(exc), model, http_status=exc.code)
    if isinstance(exc, (OSError, *TRANSIENT_TRANSPORT)):
        return _logged(transport_error, model, error_type=error_type)
    if isinstance(exc, (httpx2.DecodingError, json.JSONDecodeError, pydantic.ValidationError, TypeError)):
        return _unreadable(model, error_type=error_type)
    return _logged(GenerationError(LOCAL_REQUEST_MESSAGE), model, error_type=error_type)


def _call(operation, model: str, transport_error: GenerationError):
    """Run one SDK call. The public error is raised outside the handler, so it holds no reference to
    the SDK exception, its response text or the request."""
    # Re-applied per call because callers, including test runners, may reset warning filters.
    warnings.filterwarnings(**_SDK_ENUM_WARNING)
    failure = None
    try:
        return operation()
    except (errors.APIError, OSError, httpx2.RequestError, json.JSONDecodeError, pydantic.ValidationError,
            TypeError) as exc:
        failure = _sanitized_failure(exc, model, transport_error)
    raise failure


class Gemini:
    def __init__(self, *, attempts: int = 2):
        # Reads the process environment only. Entry points load any local .env file.
        self.model = os.getenv("GEMINI_MODEL") or DEFAULT_MODEL
        self.attempts = attempts

    def _client(self):
        key = os.getenv("GEMINI_API_KEY")
        if not key or os.getenv("DOCGROUND_FREE_TIER_CONFIRMED") != "true":
            raise ProviderNotConfigured("Gemini is offline. Configure a verified free-tier project and API key.")
        options = types.HttpOptions(base_url="https://generativelanguage.googleapis.com", timeout=TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(attempts=self.attempts, initial_delay=1,
                                                max_delay=3, http_status_codes=list(TRANSIENT_HTTP)))
        return genai.Client(api_key=key, vertexai=False, http_options=options)

    def preflight(self) -> dict:
        """Verify availability to the configured key without sending any prompts."""
        def fetch():
            with self._client() as client:
                return client.models.get(model=self.model)

        model = _call(fetch, self.model, GenerationError("Gemini model availability could not be checked."))
        if "generateContent" not in (model.supported_actions or []):
            raise GenerationError("Configured Gemini model does not advertise text generation.")
        return {"name": model.name, "version": model.version}

    def generate(self, contents: str) -> tuple[str, dict]:
        def send():
            with self._client() as client:
                return client.models.generate_content(model=self.model, contents=contents,
                    config=types.GenerateContentConfig(system_instruction=SYSTEM, temperature=0,
                        response_mime_type="application/json", response_json_schema=_RESPONSE_SCHEMA,
                        max_output_tokens=MAX_OUTPUT_TOKENS))

        response = _call(send, self.model, GenerationError(TRANSPORT_MESSAGE, retryable=True))
        return self._answer_text(response)

    def _answer_text(self, response: types.GenerateContentResponse) -> tuple[str, dict]:
        if not {"candidates", "prompt_feedback"} & response.model_fields_set:
            raise _unreadable(self.model, envelope="missing")
        candidates = response.candidates or []
        if candidates and candidates[0].finish_reason is None and candidates[0].content is None:
            raise _unreadable(self.model, envelope="empty_candidate")
        if any(c.finish_reason == types.FinishReason.MAX_TOKENS for c in candidates):
            raise ResponseTruncated("Gemini reached its output limit. Try a narrower question.")
        block_reason = getattr(response.prompt_feedback, "block_reason", None)
        if block_reason == NO_BLOCK:
            block_reason = None
        finish_reason = candidates[0].finish_reason if candidates else None
        labels = {"block_reason": _reason_label(block_reason), "finish_reason": _reason_label(finish_reason)}
        if block_reason is not None or finish_reason in BLOCKED_FINISH:
            raise _logged(ContentBlocked("Gemini declined to answer this request."), self.model, **labels)
        text = response.text
        if not text:
            raise _logged(EmptyResponse("Gemini returned no usable content."), self.model, **labels)
        meta = response.usage_metadata
        usage = {} if meta is None else meta.model_dump(mode="json", exclude_none=True)
        LOG.info("Gemini request completed", extra={"model_id": self.model, "token_usage": usage})
        return text, usage
