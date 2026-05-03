"""Build the grounded retrieval-augmented generation chain.

Uses the modern LCEL API: create_retrieval_chain wrapping
create_stuff_documents_chain so the returned object exposes both
the answer and the retrieved `context` documents (used for citations).
"""
from __future__ import annotations

from functools import lru_cache

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from src.llm import get_llm
from src.retriever import get_retriever


SYSTEM_PROMPT = (
    "You are a teaching assistant for Intro to Python Data Science. "
    "Answer the student's question using the retrieved context below. "
    "Synthesize across the snippets and quote specific terms from them. "
    "Cite the source URL after each fact in square brackets, like "
    "[https://...].\n\n"
    "If the context truly does not address the question, say so plainly "
    "and suggest where the student might look. But if the context "
    "contains relevant material, do NOT refuse -- give the best answer "
    "the context supports.\n\n"
    "Context:\n{context}"
)


@lru_cache(maxsize=1)
def get_chain():
    """Return a cached RAG chain. Invoke with `chain.invoke({'input': question})`."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )
    document_chain = create_stuff_documents_chain(get_llm(), prompt)
    return create_retrieval_chain(get_retriever(), document_chain)


def ask(question: str) -> dict:
    """Run a single question through the chain. Returns the full result dict
    with keys: 'input', 'context' (list[Document]), 'answer'.
    """
    return get_chain().invoke({"input": question})
