"""The prompt states the citation contract the validator enforces, and its recorded identity is what is sent."""
import ast
from hashlib import sha256
from importlib.metadata import distribution, packages_distributions
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import httpx2
import pytest

from scripts import evaluate_generation as evaluation
from src.checkpoints import read_journal
from src.gemini import RESPONSE_SCHEMA_SHA256
from src.generation import (PROMPT_VERSION, SYSTEM, Claim, InvalidCitation, Reference, Response, answer,
                            build_context, resolve)

ROOT = Path(__file__).resolve().parents[1]
# Each prompt version names one system instruction, structured-output schema and evidence payload format.
# Changing any of them requires a new PROMPT_VERSION and a new entry here.
PROMPT_VERSIONS = {
    "grounded-claims-v3": ("30f06abad39a1ce97c7a08f3f339b695e66f05c7985cc217bdaa2f1b64b0f6a0",
                           "9e5d1235dc964a98a721b87092cc87c5dc6559641974aec6521ed341a98b4613",
                           "d530e3f0e2041b7ea98004a6f1174a8aeb422e0de20be5c46f7b6a9d45cdf525"),
}
FIELDS = ("Use a pipeline.", "pipe = Pipeline(steps)\n    pipe.fit(X)", "Pipelines chain steps.")


def _chunk():
    fields, text, offset = [], "", 300
    for index, value in enumerate(FIELDS):
        if text:
            text, offset = text + "\n\n", offset + 2
        fields.append({"path": f"/blocks/{index}/text", "start": offset, "end": offset + len(value), "anchor": None})
        text, offset = text + value, offset + len(value)
    return {"id": "chunk", "document_id": "revision", "manifest_id": "doc", "section_id": "section",
            "heading_path": ["Pipelines"], "text": text, "start": 300, "end": 300 + len(text), "anchor": None,
            "snapshot_url": "https://example.org/doc.html", "fields": fields}


def payload_format_sha256():
    """Digest of the user content answer() sends for a fixed synthetic question and evidence item."""
    provider = Mock(generate=Mock(return_value=(json.dumps({"status": "insufficient", "claims": []}), {})))
    answer("Synthetic question?", [_chunk()], provider)
    return sha256(provider.generate.call_args.args[0].encode("utf-8")).hexdigest()


def test_prompt_names_the_payload_it_describes_and_no_hidden_structure():
    row = json.loads(build_context([_chunk()])[0])[0]
    assert all(key in SYSTEM for key in row), sorted(row)
    # Source fields are not sent. The prompt asks for one passage and treats a blank line only as a cue,
    # because a single code field can contain one.
    assert "field" not in SYSTEM.lower() and "from one passage of its text" in SYSTEM
    assert "blank line usually marks" in SYSTEM and "code example can itself contain blank lines" in SYSTEM


def test_prompt_states_every_local_response_limit():
    limits = {str(meta.max_length) for model in (Response, Claim, Reference)
              for info in model.model_fields.values() for meta in info.metadata if getattr(meta, "max_length", None)}
    minimum = next(meta.min_length for meta in Claim.model_fields["citations"].metadata
                   if getattr(meta, "min_length", None))
    assert limits == {"12", "8", "1200", "1600"} and minimum == 1
    assert all(re.search(rf"\b{limit}\b", SYSTEM) for limit in limits) and "1 to 8 citations" in SYSTEM


def test_every_quote_that_follows_the_prompt_rules_is_accepted_by_the_validator():
    chunk = _chunk()
    text, sources, accepted, rejected = chunk["text"], {"chunk": chunk}, 0, 0
    for start in range(len(text)):
        for end in range(start + 1, len(text) + 1):
            quote = text[start:end]
            # A sufficient observable condition: unique, no blank line, no leading or trailing line break.
            # Quotes containing a blank line depend on the source fields; see the test below.
            follows_rules = (quote.strip() and text.count(quote) == 1 and "\n\n" not in quote
                             and not quote.startswith("\n") and not quote.endswith("\n"))
            ref = Reference(chunk_id="chunk", document_id="revision", quote=quote)
            try:
                resolve(ref, sources)
                accepted += 1
            except InvalidCitation:
                assert not follows_rules, repr(quote)
                rejected += 1
            except ValueError:
                assert not follows_rules
    # Spanning the separator between passages is exactly what the validator rejects.
    assert accepted and rejected


BLANK_LINE_TEXT = "pipe.fit(X)\n\npipe.predict(X)"


def _blank_line_chunk(chunk_id, field_texts):
    fields, offset = [], 700
    for value in field_texts:
        fields.append({"path": f"/blocks/{len(fields)}/text", "start": offset, "end": offset + len(value),
                       "anchor": None})
        offset += len(value) + 2
    text = "\n\n".join(field_texts)
    return {"id": chunk_id, "document_id": "revision", "manifest_id": "doc", "section_id": "section",
            "heading_path": ["Pipelines"], "text": text, "start": 700, "end": 700 + len(text), "anchor": None,
            "snapshot_url": "https://example.org/doc.html", "fields": fields}


def test_blank_line_inside_one_field_is_quotable_but_the_same_text_across_two_fields_is_not():
    one_field = _blank_line_chunk("code", [BLANK_LINE_TEXT])
    two_fields = _blank_line_chunk("split", BLANK_LINE_TEXT.split("\n\n"))
    assert one_field["text"] == two_fields["text"] == BLANK_LINE_TEXT
    # The model receives identical evidence text either way, so a blank line can only be a cue.
    rows = [json.loads(build_context([chunk])[0])[0] for chunk in (one_field, two_fields)]
    assert [row["text"] for row in rows] == [BLANK_LINE_TEXT, BLANK_LINE_TEXT]

    citation = resolve(Reference(chunk_id="code", document_id="revision", quote=BLANK_LINE_TEXT), {"code": one_field})
    assert (citation["field"], citation["start"], citation["end"]) == ("/blocks/0/text", 700, 700 + len(BLANK_LINE_TEXT))
    with pytest.raises(InvalidCitation, match="could not be located in its source field"):
        resolve(Reference(chunk_id="split", document_id="revision", quote=BLANK_LINE_TEXT), {"split": two_fields})
    # Each passage of the split chunk is still quotable on its own.
    for quote, start in (("pipe.fit(X)", 700), ("pipe.predict(X)", 713)):
        assert resolve(Reference(chunk_id="split", document_id="revision", quote=quote),
                       {"split": two_fields})["start"] == start


def test_prompt_version_identifies_the_instruction_schema_and_payload_format():
    assert PROMPT_VERSION in PROMPT_VERSIONS, "register a new prompt version"
    assert PROMPT_VERSIONS[PROMPT_VERSION] == (
        sha256(SYSTEM.encode()).hexdigest(), RESPONSE_SCHEMA_SHA256, payload_format_sha256())


def test_recorded_identity_is_what_the_provider_receives(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fixture-credential")
    monkeypatch.setenv("DOCGROUND_FREE_TIER_CONFIRMED", "true")
    monkeypatch.setenv("GOOGLE_GENAI_HTTP_CLIENT", "httpx2")
    vectors = tmp_path / "vectors.npz"
    vectors.write_bytes(b"synthetic")
    monkeypatch.setattr(evaluation, "provenance", lambda root: {"git_commit": "synthetic"})
    monkeypatch.setattr(evaluation, "retriever", lambda method: SimpleNamespace(search=lambda q, k: [_chunk()]))
    monkeypatch.setattr(evaluation, "VECTORS", vectors)
    sent = []

    def respond(_transport, request):
        if request.method == "GET":
            return httpx2.Response(200, json={"name": "models/synthetic", "version": "1",
                                              "supportedGenerationMethods": ["generateContent"]})
        sent.append(json.loads(request.content))
        return httpx2.Response(200, json={"candidates": [{"finishReason": "STOP", "content": {"role": "model",
            "parts": [{"text": json.dumps({"status": "insufficient", "claims": []})}]}}]})

    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", respond)
    journal = tmp_path / "attempts.jsonl"
    question = {"id": "q", "question": "Synthetic question?", "answerable": False, "evidence_units": []}
    evaluation.collect([question], tmp_path, journal)
    header, _ = read_journal(journal)
    wire, = sent
    system = wire["systemInstruction"]["parts"][0]["text"]
    schema = wire["generationConfig"]["responseJsonSchema"]
    assert header["prompt_version"] == PROMPT_VERSION
    assert header["prompt_sha256"] == sha256(system.encode("utf-8")).hexdigest()
    assert header["response_schema_sha256"] == sha256(json.dumps(
        schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    assert sha256(wire["contents"][0]["parts"][0]["text"].encode()).hexdigest() == payload_format_sha256()


def _pins(name):
    pins = {}
    for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line and not line.startswith("-"):
            package, version = line.split("==")
            pins[re.sub(r"[-_.]+", "-", package).lower()] = version
    return pins


def test_direct_pins_match_the_lock_and_cover_every_third_party_module_used():
    direct = _pins("requirements.txt") | _pins("requirements-app.txt")
    lock = _pins("requirements-windows-py314.lock")
    assert {name: (version, lock.get(name)) for name, version in direct.items() if lock.get(name) != version} == {}
    owners = packages_distributions()

    def distributions_for(module):
        parts = module.split(".")
        candidates = owners.get(parts[0], [parts[0]])
        if len(candidates) > 1:  # a shared namespace such as google: match the distribution's own files
            prefix = "/".join(parts[:2])
            candidates = [name for name in candidates if any(
                str(file).replace("\\", "/").startswith(prefix) for file in distribution(name).files or [])] or candidates
        return candidates

    used = {}
    for path in [ROOT / "app.py", *ROOT.glob("src/*.py"), *ROOT.glob("scripts/*.py"), *ROOT.glob("experiments/*.py"),
                 *ROOT.glob("tests/*.py")]:
        source = path.read_text(encoding="utf-8")
        modules = {alias.name for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Import)
                   for alias in node.names}
        # "from google import genai" names google.genai, which identifies one distribution in a namespace.
        modules |= {f"{node.module}.{alias.name}" for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 for alias in node.names}
        modules |= set(re.findall(r"""monkeypatch\.setattr\(\s*["']([A-Za-z_][\w.]*)\.""", source))
        for module in modules:
            top = module.split(".")[0]
            if top in sys.stdlib_module_names or top in {"src", "scripts", "experiments", "tests"}:
                continue
            for name in distributions_for(module):
                used.setdefault(re.sub(r"[-_.]+", "-", name).lower(), set()).add(path.name)
    assert {name: sorted(files) for name, files in used.items() if name not in direct} == {}
