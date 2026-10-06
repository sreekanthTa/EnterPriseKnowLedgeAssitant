"""Streamlit frontend for the hybrid RAG pipeline in 02.retrieval.py.

Run with:
    uv run streamlit run streamlit_app.py

Each question goes through the full pipeline: DSPy query rewriting ->
Hugging Face embeddings -> Pinecone semantic search + BM25 keyword search ->
Reciprocal Rank Fusion -> DSPy answer generation over the retrieved context.

The pipeline lives in 02.retrieval.py. Because the module name starts with a
digit it cannot be imported with a normal ``import`` statement, so it is loaded
from its file path and cached for the lifetime of the Streamlit session.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path


logger = logging.getLogger(__name__)

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
RETRIEVAL_PATH = BASE_DIR / "02.retrieval.py"

MODEL_TIMEOUT_SECONDS = 45
MAX_QUERY_LENGTH = 2000


@st.cache_resource(show_spinner="Loading retrieval pipeline...")
def load_pipeline(mtime: float):
    """Import 02.retrieval.py once and return the module.

    Importing the module also loads the PDF, builds the chunks and the BM25
    retriever, and configures DSPy, so this is cached as a resource instead of
    re-running on every interaction. ``mtime`` is part of the cache key so that
    editing 02.retrieval.py invalidates the cached module automatically.
    """
    spec = importlib.util.spec_from_file_location("retrieval", RETRIEVAL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load retrieval module from {RETRIEVAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def format_history(messages: list[dict], max_turns: int = 6) -> str:
    """Format the most recent turns for the retrieval/answer prompts."""
    recent = messages[-max_turns * 2:]
    return "\n".join(
        f"{'User' if message['role'] == 'user' else 'Assistant'}: {message['content']}"
        for message in recent
    )


def fmt(value):
    return "—" if value is None else value


def render_sources(results: list[dict]) -> None:
    """Render the retrieved chunks that grounded the answer."""
    with st.expander(f"Sources ({len(results)})", expanded=False):
        for i, result in enumerate(results, start=1):
            with st.container(border=True):
                st.markdown(f"**Source {i}** · `{result['chunk_id']}`")
                st.caption(
                    f"RRF {result['rrf_score']:.4f} · "
                    f"Similarity {result['similarity_score']:.4f} · "
                    f"Semantic rank {fmt(result['semantic_rank'])} · "
                    f"Keyword rank {fmt(result['keyword_rank'])}"
                )
                st.write(result["chunk"])


def main() -> None:
    st.set_page_config(
        page_title="Enterprise Knowledge Assistant",
        page_icon="🔎",
        layout="wide",
    )

    st.title("🔎 Enterprise Knowledge Assistant")
    st.caption(
        "Ask a question about the indexed document. Answers are grounded in "
        "retrieved context via hybrid search (Pinecone semantics + BM25 keywords, "
        "fused with Reciprocal Rank Fusion)."
    )

    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Chunks to retrieve", min_value=1, max_value=20, value=5)
        show_sources = st.toggle("Show retrieved sources", value=True)
        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.pop("messages", None)
            st.rerun()
        st.divider()
        st.caption(
            "Pipeline: DSPy query rewrite → HF embeddings → "
            "Pinecone + BM25 → RRF → DSPy answer"
        )

    with st.sidebar:
        if st.button("Check system health", use_container_width=True):
            try:
                with st.spinner("Loading retrieval pipeline..."):
                    pipeline = load_pipeline(RETRIEVAL_PATH.stat().st_mtime)
                with st.spinner("Checking system health..."):
                    health = pipeline.health_check()
                st.write("Healthy" if health["ok"] else "Unhealthy")
                for name, component in health.items():
                    if name == "ok":
                        continue
                    label = "OK" if component["ok"] else "Failed"
                    st.caption(f"{name}: {label}")
            except Exception:
                logger.exception("Health check failed")
                st.error("Health check failed. Check the server logs.")

    st.session_state.setdefault("messages", [])

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and show_sources and message.get("sources"):
                render_sources(message["sources"])

    question = st.chat_input("Ask a question about the indexed document")
    if not question or not question.strip():
        return

    question = question.strip()
    if len(question) > MAX_QUERY_LENGTH:
        st.error(
            f"Please keep your question under {MAX_QUERY_LENGTH} characters."
        )
        return

    conversation_history = format_history(st.session_state["messages"])
    st.session_state["messages"].append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Loading retrieval pipeline..."):
                pipeline = load_pipeline(RETRIEVAL_PATH.stat().st_mtime)
            with st.spinner("Retrieving relevant chunks..."):
                results = pipeline.retrieve_relevant_chunks(
                    question,
                    top_k=top_k,
                    conversation_history=conversation_history,
                )
        except Exception:
            logger.exception("Retrieval failed")
            st.error("Retrieval failed. Please try again.")
            st.stop()

        answer = getattr(results, "direct_response", "")
        if not answer:
            try:
                with st.spinner(
                    f"Generating answer (timeout: {MODEL_TIMEOUT_SECONDS}s)..."
                ):
                    answer = pipeline.generate_answer(
                        question,
                        results,
                        conversation_history,
                    )
            except Exception:
                logger.exception("Answer generation failed")
                st.error("Answer generation failed or timed out. Please try again.")

        if not str(answer).strip():
            answer = "I could not generate an answer from the retrieved context."

        st.markdown(answer)

        if show_sources and results:
            render_sources(results)

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": answer,
            "sources": list(results),
        }
    )


if __name__ == "__main__":
    main()
