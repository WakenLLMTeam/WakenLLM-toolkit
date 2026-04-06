import os

from src.rag.ragflow import RAGFlowProvider
from src.rag.retriever import Retriever


def build_retriever() -> Retriever | None:
    provider = (os.getenv("RAG_PROVIDER") or "").strip().lower()
    if not provider:
        return None
    if provider == "ragflow":
        return RAGFlowProvider()
    raise ValueError(f"Unsupported RAG provider: {provider}")
