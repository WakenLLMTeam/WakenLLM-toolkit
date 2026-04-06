import os

from deerflow.rag.ragflow import RAGFlowProvider
from deerflow.rag.retriever import Retriever


def build_retriever() -> Retriever | None:
    provider = (os.getenv("RAG_PROVIDER") or "").strip().lower()
    if not provider:
        return None
    if provider == "ragflow":
        return RAGFlowProvider()
    raise ValueError(f"Unsupported RAG provider: {provider}")