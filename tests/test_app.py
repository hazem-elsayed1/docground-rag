from pathlib import Path
from hashlib import sha256
import json
import httpx2
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from src.corpus import build_corpus
from src import groq
from src.generation import ContentBlocked, GenerationError, ResponseTruncated, SYSTEM

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_empty_submission_and_retrieval_failure_are_safe(monkeypatch, tmp_path):
    st.cache_resource.clear()
    monkeypatch.setattr("src.runtime.CHUNKS", tmp_path / "absent")
    at = AppTest.from_file(str(APP)).run()
    at.button[0].click().run()
    assert at.warning and not at.exception
    at.text_area[0].input("How does a pipeline work?")
    at.button[0].click().run()
    assert at.error and str(tmp_path) not in at.error[0].value and not at.exception


@pytest.fixture
def evidence_app(tmp_path, monkeypatch):
    st.cache_resource.clear()
    url = "https://scikit-learn.org/1.9/guide.html"
    html = (f'<title>Fixture — scikit-learn 1.9.1 documentation</title><link rel="canonical" href="{url}">'
            '<article><h1 id="guide">Guide</h1><p>Use a pipeline for the workflow.</p></article>').encode()
    (tmp_path / "guide.html").write_bytes(html)
    manifest = {"manifest_schema_version": 1, "corpus_version": "test", "documentation_version": "1.9.1",
                "attribution": {"project": "scikit-learn", "copyright": "The scikit-learn developers",
                                "license": "BSD-3-Clause",
                                "license_url": "https://github.com/scikit-learn/scikit-learn/blob/1.9.1/COPYING"},
                "documents": [{"id": "guide", "url": url, "sha256": sha256(html).hexdigest()}]}
    _, chunks = build_corpus(manifest, tmp_path)
    payload = json.dumps(chunks).encode()
    path = tmp_path / "chunks.json"
    path.write_bytes(payload)
    monkeypatch.setattr("src.runtime.CHUNKS", path)
    monkeypatch.setattr("src.runtime.CHUNK_HASH", sha256(payload).hexdigest())
    at = AppTest.from_file(str(APP)).run()
    return at.selectbox[0].select("bm25").run()


def test_no_hits_abstains_and_retains_session(evidence_app):
    at = evidence_app
    at.text_area[0].input("Unknown question")
    at.button[0].click().run()
    assert "No documentation excerpts were selected" in at.info[0].value and not at.exception
    at.run()
    assert at.info and at.session_state["history"][0]["result"]["context_ids"] == []


@pytest.mark.parametrize(("outcome", "level", "message"), [
    ("invalid JSON", "error", "unexpected format"),
    (json.dumps({"status": "answered", "claims": [{"text": "invented",
        "kind": "explicit", "citations": [{"chunk_id": "invented", "document_id": "invented",
                                         "quote": "a pipeline"}]}]}), "error", "outside the supplied evidence"),
    (json.dumps({"status": "insufficient", "claims": []}), "info", "The model reported insufficient evidence"),
    (json.dumps({"status": "conflicting", "claims": []}), "info", "evidence conflicts"),
    (ResponseTruncated("Gemini reached its output limit."), "error", "output limit"),
    (ContentBlocked("Gemini declined to answer this request."), "error", "declined to answer"),
    (GenerationError("Gemini is unavailable or the request timed out."), "error", "request timed out"),
])
def test_generation_outcomes_render_separately(evidence_app, monkeypatch, outcome, level, message):
    def generate(self, contents):
        if isinstance(outcome, GenerationError):
            raise outcome
        return outcome, {}

    monkeypatch.setattr("src.gemini.Gemini.generate", generate)
    at = evidence_app
    at.text_area[0].input("What is a pipeline?")
    at.button[0].click().run()
    assert not at.exception and at.expander
    assert message in getattr(at, level)[0].value
    assert not getattr(at, "info" if level == "error" else "error")
    assert not any(item.value == "invented" for item in at.text)


def test_unexpected_generation_failure_is_not_reported_as_a_search_failure(evidence_app, monkeypatch):
    def generate(self, contents):
        raise RuntimeError("synthetic-internal-detail")

    monkeypatch.setattr("src.gemini.Gemini.generate", generate)
    at = evidence_app
    at.text_area[0].input("What is a pipeline?")
    at.button[0].click().run()
    assert not at.exception and "The answer could not be generated" in at.error[0].value
    assert "Documentation search is unavailable" not in at.error[0].value
    assert "synthetic-internal-detail" not in at.error[0].value
    assert at.expander, "retrieved sources must stay inspectable"


def test_entrypoint_loads_configuration_before_the_provider_reads_it(evidence_app, monkeypatch):
    seen = []

    def load_dotenv(*args, **kwargs):
        seen.append((args, kwargs))
        monkeypatch.setenv("GEMINI_MODEL", "gemini-from-entrypoint")
        return True

    def generate(self, contents):
        seen.append(self.model)
        return json.dumps({"status": "insufficient", "claims": []}), {}

    monkeypatch.setattr("dotenv.load_dotenv", load_dotenv)
    monkeypatch.setattr("src.gemini.Gemini.generate", generate)
    at = evidence_app
    at.text_area[0].input("What is a pipeline?")
    at.button[0].click().run()
    assert not at.exception and "The model reported insufficient evidence" in at.info[0].value
    assert seen == [((), {"override": False}), "gemini-from-entrypoint"]


def test_answer_shows_verified_citation_and_marks_claim_support_unreviewed(evidence_app, monkeypatch):
    def generate(self, contents):
        source, = json.loads(contents)["evidence"]
        citation = {key: source[key] for key in ("chunk_id", "document_id")}
        citation["quote"] = "Use a pipeline for the workflow."
        return json.dumps({"status": "answered", "claims": [
            {"text": "A pipeline is recommended.", "kind": "explicit", "citations": [citation]}]}), {}

    monkeypatch.setattr("src.gemini.Gemini.generate", generate)
    at = evidence_app
    at.text_area[0].input("What is a pipeline?")
    at.button[0].click().run()
    assert not at.exception and not at.error and not at.info
    assert "Quote locations verified. Claim support has not been reviewed." in at.warning[0].value
    assert any(item.value == "A pipeline is recommended." for item in at.text)
    assert len(at.get("link_button")) == 1
    # The claim comes first, outside the source control that holds its quote, link and technical details.
    source, retrieved = at.expander
    assert (source.label, retrieved.label) == ("Show source", "Inspect retrieved sources")
    assert [item.value for item in source.text] == ["Use a pipeline for the workflow."]
    assert len(source.get("link_button")) == 1 and source.get("link_button")[0].proto.url.startswith("https://")
    details = source.caption[-1].value
    assert all(part in details for part in ("guide", "/blocks/0/text", "chars "))
    order = [getattr(node, "value", None) or getattr(node, "label", None) for node in at.main]
    positions = [order.index(value) for value in ("A pipeline is recommended.", "Show source", at.warning[0].value,
                                                  "Inspect retrieved sources")]
    assert positions == sorted(positions)


QUOTE = "Use a pipeline for the workflow."
SECRET = "provider-private-text"


def grounded(request, quote=QUOTE):
    source, = json.loads(json.loads(request.content)["messages"][1]["content"])["evidence"]
    return httpx2.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(
        {"status": "answered", "claims": [{"text": "A pipeline is recommended.", "kind": "explicit", "citations": [
            {"chunk_id": source["chunk_id"], "document_id": source["document_id"], "quote": quote}]}]})}}]})


@pytest.fixture
def groq_http(monkeypatch):
    """Select Groq through the environment and answer its HTTP requests in memory."""
    monkeypatch.setenv("DOCGROUND_GENERATION_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "fixture-credential")
    monkeypatch.setenv("DOCGROUND_FREE_TIER_CONFIRMED", "true")
    gemini, sleeps = [], []
    monkeypatch.setattr("src.gemini.Gemini.generate", lambda self, contents: gemini.append(contents))
    monkeypatch.setattr(groq, "_sleep", sleeps.append)
    state = {"requests": [], "reply": grounded}

    def handle(_transport, request):
        state["requests"].append(request)
        return state["reply"](request)

    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", handle)
    yield state
    # Recorded instead of raised, because the app would report an exception as a failed answer.
    assert gemini == [] and sleeps == []


def submit(at, question="What is a pipeline?"):
    at.text_area[0].input(question)
    return at.button[0].click().run()


def sources_visible(at):
    panels = [panel for panel in at.expander if panel.label == "Inspect retrieved sources"]
    return panels and any(QUOTE in item.value for item in panels[0].text)


def shows_sources_only(at, name, key):
    notice = at.info[0].value
    return (not at.error and not at.exception and not at.warning and sources_visible(at) and "sources only" in notice
            and f"{name} is not configured" in notice and key in notice and "DOCGROUND_FREE_TIER_CONFIRMED" in notice)


def test_groq_form_submission_sends_one_request_and_shows_a_verified_answer(evidence_app, groq_http):
    at = submit(evidence_app)
    request, = groq_http["requests"]
    assert (request.method, str(request.url)) == ("POST", groq.BASE_URL + "/chat/completions")
    body = json.loads(request.content)
    assert body == groq.request_body(body["messages"][1]["content"]) and body["messages"][0]["content"] == SYSTEM
    assert json.loads(body["messages"][1]["content"])["question"] == "What is a pipeline?"
    assert not at.exception and not at.error and sources_visible(at)
    assert any(item.value == "A pipeline is recommended." for item in at.text) and len(at.get("link_button")) == 1
    # A page rerun shows the stored turn. Only another submission sends another request, without waiting.
    at.run()
    assert len(groq_http["requests"]) == 1
    submit(at)
    assert len(groq_http["requests"]) == 2 and not at.exception


def transport_failure(request):
    raise httpx2.ReadTimeout(SECRET)


@pytest.mark.parametrize(("reply", "message"), [
    (lambda request: httpx2.Response(503, json={"error": {"message": SECRET}}), "temporarily unavailable"),
    (lambda request: httpx2.Response(429, json={"error": {"message": SECRET}}), "rate limit"),
    (lambda request: httpx2.Response(401, json={"error": {"message": SECRET}}), "authentication failed"),
    (transport_failure, "timed out"),
    (lambda request: httpx2.Response(200, content=SECRET.encode()), "unreadable response"),
    (lambda request: httpx2.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
        "content": SECRET}}]}), "unexpected format"),
    (lambda request: grounded(request, quote="not in the evidence"), "could not be located"),
])
def test_groq_form_failures_send_one_request_and_keep_sources_visible(evidence_app, groq_http, reply, message):
    groq_http["reply"] = reply
    at = submit(evidence_app)
    assert len(groq_http["requests"]) == 1 and not at.exception
    assert message in at.error[0].value and SECRET not in at.error[0].value and not at.info and not at.warning
    assert sources_visible(at) and not any(item.value == "A pipeline is recommended." for item in at.text)


def test_groq_submission_without_evidence_sends_no_request(evidence_app, groq_http):
    at = submit(evidence_app, "Unknown question")
    assert groq_http["requests"] == [] and "No documentation excerpts" in at.info[0].value and not at.exception


@pytest.mark.parametrize("missing", ["GROQ_API_KEY", "DOCGROUND_FREE_TIER_CONFIRMED"])
def test_groq_submission_without_configuration_sends_no_request(evidence_app, groq_http, monkeypatch, missing):
    monkeypatch.delenv(missing)
    at = submit(evidence_app)
    assert groq_http["requests"] == [] and shows_sources_only(at, "Groq", "GROQ_API_KEY")


def test_gemini_submission_without_configuration_shows_sources_only(evidence_app, monkeypatch):
    requests = []
    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", lambda _transport, request: requests.append(request))
    at = submit(evidence_app)
    assert requests == [] and shows_sources_only(at, "Gemini", "GEMINI_API_KEY")


@pytest.mark.parametrize(("provider", "label"), [
    (None, "Gemini · gemini-3.1-flash-lite"), ("gemini", "Gemini · gemini-3.1-flash-lite"),
    ("groq", "Groq · openai/gpt-oss-20b"), ("other", "not recognized"),
])
def test_selected_provider_is_shown_before_any_submission_without_a_request(
        evidence_app, groq_http, monkeypatch, provider, label):
    if provider is None:
        monkeypatch.delenv("DOCGROUND_GENERATION_PROVIDER")
    else:
        monkeypatch.setenv("DOCGROUND_GENERATION_PROVIDER", provider)
    at = evidence_app.run()
    assert f"Answer provider: {label}" in [item.value for item in at.caption]
    assert groq_http["requests"] == [] and not at.exception and not at.error


def test_several_citations_share_one_source_control(evidence_app, groq_http):
    def two(request):
        body = json.loads(grounded(request).content)
        answer = json.loads(body["choices"][0]["message"]["content"])
        first, = answer["claims"][0]["citations"]
        answer["claims"][0]["citations"].append(first | {"quote": "for the workflow."})
        body["choices"][0]["message"]["content"] = json.dumps(answer)
        return httpx2.Response(200, json=body)

    groq_http["reply"] = two
    at = submit(evidence_app)
    panel, _ = at.expander
    assert panel.label == "Show 2 sources" and len(panel.get("link_button")) == 2 and not at.exception
    assert [item.value for item in panel.text] == [QUOTE, "for the workflow."]


def test_unknown_provider_sends_no_request_and_keeps_sources_visible(evidence_app, groq_http, monkeypatch):
    monkeypatch.setenv("DOCGROUND_GENERATION_PROVIDER", "other")
    at = submit(evidence_app)
    assert groq_http["requests"] == [] and "gemini or groq" in at.error[0].value and sources_visible(at)


@pytest.mark.parametrize("value", [None, "", "gemini"])
def test_gemini_stays_the_default_even_when_a_groq_key_is_set(evidence_app, groq_http, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("DOCGROUND_GENERATION_PROVIDER")
    else:
        monkeypatch.setenv("DOCGROUND_GENERATION_PROVIDER", value)
    calls = []

    def generate(self, contents):
        calls.append(contents)
        return json.dumps({"status": "insufficient", "claims": []}), {}

    monkeypatch.setattr("src.gemini.Gemini.generate", generate)
    at = submit(evidence_app)
    assert len(calls) == 1 and groq_http["requests"] == [] and "insufficient evidence" in at.info[0].value


def test_env_example_first_run_selects_groq_but_sends_nothing_until_configured(evidence_app, groq_http, monkeypatch):
    lines = (APP.parent / ".env.example").read_text(encoding="utf-8").splitlines()
    example = dict(line.split("=", 1) for line in lines if line and not line.startswith("#"))
    assert example == {"DOCGROUND_GENERATION_PROVIDER": "groq", "GROQ_API_KEY": "", "GEMINI_API_KEY": "",
                       "DOCGROUND_FREE_TIER_CONFIRMED": "false", "GEMINI_MODEL": "gemini-3.1-flash-lite"}
    for name, value in example.items():
        monkeypatch.setenv(name, value)
    at = submit(evidence_app)
    assert groq_http["requests"] == [] and shows_sources_only(at, "Groq", "GROQ_API_KEY")
    # A key alone is not enough while the free-tier confirmation stays false.
    monkeypatch.setenv("GROQ_API_KEY", "fixture-credential")
    submit(at)
    assert groq_http["requests"] == [] and shows_sources_only(at, "Groq", "GROQ_API_KEY")
