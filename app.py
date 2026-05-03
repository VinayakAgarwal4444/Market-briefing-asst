"""Streamlit chat UI for the RAG Teaching Assistant.

Bootstrap order matters: ``st.secrets`` is mirrored into ``os.environ``
*before* ``config`` (and therefore any other project module) is imported,
so the same ``os.getenv``-based config works locally and on Streamlit Cloud.
"""
from __future__ import annotations

import os

import streamlit as st

# --- Streamlit Cloud secrets -> env vars (must run before our imports) ----- #
try:
    for _key, _value in st.secrets.items():
        # Don't clobber a value the developer set in their shell.
        if isinstance(_value, str) and _key not in os.environ:
            os.environ[_key] = _value
except (FileNotFoundError, AttributeError):
    # No secrets.toml locally -- that's fine, we'll fall through to .env.
    pass

# --- Project imports (must come AFTER the secrets shim) -------------------- #
from config import EMBED_PROVIDER, LLM_PROVIDER, RETRIEVAL_K  # noqa: E402
from src.chain import get_chain  # noqa: E402
from src.llm import ProviderUnavailableError  # noqa: E402
from src.retriever import VectorstoreMissingError  # noqa: E402


def _model_caption() -> str:
    """One-line description of which models are wired up right now."""
    if LLM_PROVIDER == "groq":
        from config import GROQ_LLM_MODEL

        llm_label = f"Groq · `{GROQ_LLM_MODEL}`"
    else:
        from config import OLLAMA_LLM_MODEL

        llm_label = f"Ollama · `{OLLAMA_LLM_MODEL}`"

    if EMBED_PROVIDER == "hf_endpoint":
        from config import HF_EMBED_MODEL

        emb_label = f"HF · `{HF_EMBED_MODEL}`"
    else:
        from config import OLLAMA_EMBED_MODEL

        emb_label = f"Ollama · `{OLLAMA_EMBED_MODEL}`"

    return (
        f"Grounded in pandas / NumPy / scikit-learn docs · "
        f"LLM: {llm_label} · embed: {emb_label} · top-k = {RETRIEVAL_K}"
    )


st.set_page_config(
    page_title="Python Data Science TA",
    page_icon=":books:",
    layout="centered",
)

st.title("Python Data Science Teaching Assistant")
st.caption(_model_caption())

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.markdown(f"- [{s['title']}]({s['url']})")

prompt = st.chat_input("Ask something about pandas, NumPy, or scikit-learn...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                chain = get_chain()
                result = chain.invoke({"input": prompt})
            except ProviderUnavailableError as exc:
                st.error(str(exc))
                st.stop()
            except VectorstoreMissingError as exc:
                st.error(str(exc))
                st.stop()

        answer = result.get("answer", "")
        st.markdown(answer)

        sources: list[dict] = []
        seen: set[str] = set()
        for doc in result.get("context", []):
            url = doc.metadata.get("source_url", "")
            title = doc.metadata.get("title", "(untitled)")
            if url and url not in seen:
                seen.add(url)
                sources.append({"title": title, "url": url})

        if sources:
            with st.expander(f"Sources ({len(sources)})"):
                for s in sources:
                    st.markdown(f"- [{s['title']}]({s['url']})")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

with st.sidebar:
    st.subheader("About")
    st.markdown(
        "This assistant answers questions strictly from the official "
        "**pandas**, **NumPy**, and **scikit-learn** documentation it has "
        "been indexed on. If the answer isn't in the docs, it will say so "
        "instead of guessing.\n\n"
        "_Built with LangChain · FAISS · Streamlit._"
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
