"""Streamlit interface to the pinned DocGround application."""
import logging

import dotenv
import streamlit as st
from src.generation import GenerationError, answer, check_question
from src.gemini import Gemini
from src.runtime import METHODS, retriever

# The application entry point loads local configuration. Existing variables take precedence.
dotenv.load_dotenv(override=False)

st.set_page_config(page_title="DocGround · Documentation search", layout="centered")
st.title("DocGround")
st.caption("Ask a question about the four pinned scikit-learn 1.9.1 documentation chapters.")
load_search = st.cache_resource(retriever, max_entries=1, show_spinner=False)
method = st.selectbox("Retrieval method", METHODS, format_func={
    "hybrid": "Hybrid · BM25 + dense", "bm25": "BM25", "dense": "Dense · MiniLM",
    "reranked": "Hybrid + cross-encoder"}.__getitem__)
with st.form("ask"):
    question = st.text_area("Question", max_chars=2000)
    submitted = st.form_submit_button("Search and answer", type="primary")

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
                    turn["result"] = answer(query, turn["hits"], Gemini())
                except GenerationError as exc:
                    turn["error"] = str(exc)
                except Exception as exc:
                    logging.getLogger(__name__).error("Answer failed", extra={"error_type": type(exc).__name__})
                    turn["error"] = "The answer could not be generated. The retrieved sources are shown below."
        st.session_state["history"] = (st.session_state.get("history", []) + [turn])[-5:]

for turn_index, turn in enumerate(reversed(st.session_state.get("history", []))):
    st.divider()
    st.text(turn["question"])
    if "error" in turn:
        st.error(turn["error"])
    else:
        result = turn["result"]
        if result["status"] == "answered":
            st.warning("Citation locations verified. Claim support has not been reviewed.")
        else:
            st.info(result["explanation"])
        if any(row["reason"] == "byte_budget" for row in result["context_selection"]["dropped"]):
            st.caption("Some retrieved chunks exceeded the evidence budget and were excluded.")
        for claim_index, claim in enumerate(result["claims"]):
            st.text(("Inference · " if claim["kind"] == "inference" else "") + claim["text"])
            for cite_index, citation in enumerate(claim["citations"]):
                st.text(citation["quote"])
                st.link_button("Open cited passage", citation["url"],
                               key=f"cite-{turn_index}-{claim_index}-{cite_index}")
                st.caption(f'{citation["manifest_id"]} · {citation["chunk_id"]} · chars {citation["start"]}–{citation["end"]}')
    if turn["hits"]:
        with st.expander("Inspect retrieved sources"):
            for rank, hit in enumerate(turn["hits"], 1):
                st.text(f'{rank}. {" › ".join(hit["heading_path"])}')
                st.caption(f'{turn["method"]} · score {hit["score"]:.5g} · {hit["id"]}')
                st.text(hit["text"])
                if "result" in turn and hit["id"] in turn["result"]["context_ids"]:
                    st.caption("Included in answer context")
