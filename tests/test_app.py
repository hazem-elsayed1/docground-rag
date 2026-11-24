from pathlib import Path
from hashlib import sha256
import json
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from src.corpus import build_corpus
from src.generation import ContentBlocked, GenerationError, ResponseTruncated

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
    assert "Citation locations verified. Claim support has not been reviewed." in at.warning[0].value
    assert any(item.value == "A pipeline is recommended." for item in at.text)
    assert len(at.get("link_button")) == 1
