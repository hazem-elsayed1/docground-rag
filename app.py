"""Streamlit interface to the pinned DocGround application."""
import logging
import os
import re

import dotenv
import streamlit as st
from src.generation import GenerationError, ProviderNotConfigured, answer, check_question
from src.gemini import Gemini
from src.groq import Groq
from src.runtime import METHODS, retriever

# The application entry point loads local configuration. Existing variables take precedence.
dotenv.load_dotenv(override=False)
PROVIDERS = {"gemini": ("Gemini", Gemini, "GEMINI_API_KEY"), "groq": ("Groq", Groq, "GROQ_API_KEY")}
PROVIDER = os.getenv("DOCGROUND_GENERATION_PROVIDER") or "gemini"


def generation_provider():
    """Build the configured provider without a request. Gemini is used when unset. Unknown names are refused."""
    if PROVIDER not in PROVIDERS:
        raise GenerationError("Set DOCGROUND_GENERATION_PROVIDER to gemini or groq.")
    return PROVIDERS[PROVIDER][1]()


def provider_label():
    if PROVIDER not in PROVIDERS:
        return "not recognized"
    return f"{PROVIDERS[PROVIDER][0]} · {generation_provider().model}"


METHOD_LABELS = {"hybrid": "Hybrid · BM25 + dense", "bm25": "BM25", "dense": "Dense · MiniLM",
                 "reranked": "Hybrid + cross-encoder"}
SUPPORT_NOTICE = ("Each quote was found at its cited location in the documentation. Whether the quote "
                  "supports the claim has not been reviewed, so read the quote before relying on the claim.")
ABSTENTION_TITLES = {"insufficient": "Not enough evidence to answer", "conflicting": "The evidence conflicts"}
ABSTENTION_HELP = ("DocGround answers only from the four pinned scikit-learn 1.9.1 chapters. Try a narrower "
                   "question, or review the retrieved passages in Retrieval details.")


def literal(text: str) -> str:
    """Show model and user text literally: no Markdown links, emphasis, HTML or color directives."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|<>~:$])", r"\\\1", text)


def section_label(hit: dict | None) -> str:
    return " › ".join(hit["heading_path"]) if hit and hit.get("heading_path") else "Cited documentation passage"


st.set_page_config(page_title="DocGround · Documentation search", layout="centered")
st.title("DocGround")
st.markdown("Ask a question about four pinned scikit-learn 1.9.1 documentation chapters. DocGround answers "
            "only from those chapters and shows the exact quote and source link for every claim.")
load_search = st.cache_resource(retriever, max_entries=1, show_spinner=False)
with st.form("ask"):
    question = st.text_area("Question", max_chars=2000,
                            placeholder="For example: Why should the test set not be used to choose hyperparameters?")
    submitted = st.form_submit_button("Search and answer", type="primary")
st.caption("Answer provider: " + provider_label())
with st.expander("Search settings"):
    method = st.selectbox("Retrieval method", METHODS, format_func=METHOD_LABELS.__getitem__)

if submitted:
    try:
        query = check_question(question)
    except ValueError as exc:
        st.warning(str(exc))
    else:
        turn = {"question": query, "method": method, "hits": []}
        with st.spinner("Searching documentation and checking the answer…"):
            try:
                turn["hits"] = load_search(method).search(query, 10)
            except Exception as exc:
                logging.getLogger(__name__).error("Search failed", extra={"error_type": type(exc).__name__})
                turn["error"] = "Documentation search is unavailable. Check the corpus and local model installation."
            else:
                # Retrieved sources stay inspectable whatever happens during answer generation.
                try:
                    turn["result"] = answer(query, turn["hits"], generation_provider())
                except ProviderNotConfigured:
                    name, _, key = PROVIDERS[PROVIDER]
                    turn["notice"] = (f"Showing sources only. {name} is not configured, so no answer was requested. To "
                                      f"enable answers, set {key} and DOCGROUND_FREE_TIER_CONFIRMED=true, then restart.")
                except GenerationError as exc:
                    turn["error"] = str(exc)
                except Exception as exc:
                    logging.getLogger(__name__).error("Answer failed", extra={"error_type": type(exc).__name__})
                    turn["error"] = "The answer could not be generated. The retrieved sources are shown below."
        st.session_state["history"] = (st.session_state.get("history", []) + [turn])[-5:]

for turn_index, turn in enumerate(reversed(st.session_state.get("history", []))):
    st.divider()
    st.caption("Question")
    st.markdown(f"**{literal(turn['question'])}**")
    hits_by_id = {hit["id"]: hit for hit in turn["hits"]}
    result = turn.get("result")
    if "error" in turn:
        st.error(turn["error"])
    elif "notice" in turn:
        st.info(turn["notice"])
    elif result["status"] != "answered":
        with st.container(border=True):
            st.subheader(ABSTENTION_TITLES.get(result["status"], "No answer"), anchor=False)
            st.info(result["explanation"])
            st.caption(ABSTENTION_HELP)
    else:
        count = len(result["claims"])
        st.subheader("Answer" if count == 1 else f"Answer · {count} claims", anchor=False)
        st.warning(SUPPORT_NOTICE)
        for claim_index, claim in enumerate(result["claims"]):
            with st.container(border=True):
                if claim["kind"] == "inference":
                    st.caption("Inference: deduced from the quoted evidence, not stated directly")
                st.markdown(literal(claim["text"]))
                for cite_index, citation in enumerate(claim["citations"]):
                    st.caption("Exact quote · location verified" if len(claim["citations"]) == 1
                               else f"Exact quote {cite_index + 1} of {len(claim['citations'])} · location verified")
                    # Plain text: characters, line breaks and code indentation appear exactly as quoted.
                    st.text(citation["quote"])
                    st.caption("Source: " + literal(section_label(hits_by_id.get(citation["chunk_id"]))))
                    st.link_button("Open source passage", citation["url"],
                                   key=f"cite-{turn_index}-{claim_index}-{cite_index}")
    if turn["hits"] or result:
        with st.expander("Retrieval details", expanded="notice" in turn):
            st.caption(f'Retrieval method: {METHOD_LABELS.get(turn["method"], turn["method"])} · '
                       f'{len(turn["hits"])} passages retrieved')
            if result and any(row["reason"] == "byte_budget" for row in result["context_selection"]["dropped"]):
                st.caption("Some retrieved passages exceeded the evidence budget and were not sent to the model.")
            if result and result["status"] == "answered":
                st.markdown("**Citation locations**")
                for claim_index, claim in enumerate(result["claims"], 1):
                    for citation in claim["citations"]:
                        st.caption(f'Claim {claim_index} · {citation["manifest_id"]} · {citation["field"]} · '
                                   f'{citation["chunk_id"]} · chars {citation["start"]}–{citation["end"]}')
            if turn["hits"]:
                st.markdown("**Retrieved passages**")
            for rank, hit in enumerate(turn["hits"], 1):
                used = bool(result) and hit["id"] in result["context_ids"]
                st.markdown(f"{rank}\\. {literal(section_label(hit))}")
                st.caption(f'score {hit["score"]:.5g} · {hit["id"]}' + (" · included in answer context" if used else ""))
                st.text(hit["text"])
