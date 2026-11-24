"""Exercise the installed SDK wire format with an in-memory HTTP transport."""
import json
import httpx2
import pytest
from src.gemini import Gemini
from src.generation import Response, ResponseTruncated


def _local_response_shape():
    local_schema = Response.model_json_schema()

    def structural_fields(schema):
        if "$ref" in schema:
            schema = local_schema["$defs"][schema["$ref"].rsplit("/", 1)[-1]]
        shape = {
            key: schema[key]
            for key in ("type", "enum", "required", "additionalProperties")
            if key in schema
        }
        if "properties" in schema:
            shape["properties"] = {
                name: structural_fields(field) for name, field in schema["properties"].items()
            }
        if "items" in schema:
            shape["items"] = structural_fields(schema["items"])
        return shape

    return structural_fields(local_schema)


def _assert_no_length_constraints(schema):
    if isinstance(schema, dict):
        forbidden = {"minItems", "maxItems", "minLength", "maxLength"} & schema.keys()
        assert not forbidden, f"Gemini wire schema contains length constraints: {sorted(forbidden)}"
        for child in schema.values():
            _assert_no_length_constraints(child)
    elif isinstance(schema, list):
        for child in schema:
            _assert_no_length_constraints(child)


@pytest.mark.parametrize("finish", ["STOP", "MAX_TOKENS"])
def test_real_sdk_wire_contract_after_rate_limit(monkeypatch, finish):
    monkeypatch.setenv("GEMINI_API_KEY", "fixture-only")
    monkeypatch.setenv("DOCGROUND_FREE_TIER_CONFIRMED", "true")
    monkeypatch.setenv("GOOGLE_GENAI_HTTP_CLIENT", "httpx2")
    requests = []

    def respond(transport, request):
        requests.append(request)
        if len(requests) == 1:
            return httpx2.Response(429, json={"error": {"code": 429, "message": "slow down"}})
        return httpx2.Response(200, json={"candidates": [{"finishReason": finish,
            "content": {"role": "model", "parts": [{"text": "{}"}]}}],
            "usageMetadata": {"totalTokenCount": 12}})

    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", respond)
    if finish == "STOP":
        assert Gemini().generate("fixture evidence") == ("{}", {"total_token_count": 12})
    else:
        with pytest.raises(ResponseTruncated, match="output limit") as error:
            Gemini().generate("fixture evidence")
        assert error.value.code == "response_truncated" and not error.value.retryable
    payload = json.loads(requests[-1].content)
    assert requests[-1].url.host == "generativelanguage.googleapis.com"
    # Nested array bounds in the old wire schema exceeded Gemini's serving state limit.
    config = payload["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    wire_schema = config["responseJsonSchema"]
    _assert_no_length_constraints(wire_schema)
    assert wire_schema == _local_response_shape()
    assert payload["contents"][0]["parts"][0]["text"] == "fixture evidence"
    assert payload["systemInstruction"]["parts"][0]["text"]
