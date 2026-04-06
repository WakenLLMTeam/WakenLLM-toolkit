import os

from deerflow.rag.ragflow import RAGFlowProvider


def test_ragflow_provider_builds_document_urls_from_retrieval_chunks(monkeypatch):
    monkeypatch.setenv("RAGFLOW_API_URL", "http://ragflow.internal:9380")
    monkeypatch.setenv("RAGFLOW_API_KEY", "test-key")
    monkeypatch.delenv("RAGFLOW_BASE_URL", raising=False)

    provider = RAGFlowProvider()
    docs = {}

    provider._merge_retrieval_result(
        {
            "doc_aggs": [{"doc_id": "doc-123", "doc_name": "Battery Report"}],
            "chunks": [
                {
                    "document_id": "doc-123",
                    "dataset_id": "dataset-456",
                    "content": "Important finding",
                    "similarity": 0.91,
                }
            ],
        },
        docs,
    )

    document = docs["doc-123"]
    assert document.title == "Battery Report"
    assert document.dataset_id == "dataset-456"
    assert document.url == "http://ragflow.internal:9380/chunk/parsed/chunks?id=dataset-456&doc_id=doc-123"
    assert document.to_dict()["url"] == document.url


def test_ragflow_provider_prefers_explicit_base_url(monkeypatch):
    monkeypatch.setenv("RAGFLOW_API_URL", "http://ragflow-api.internal:9380")
    monkeypatch.setenv("RAGFLOW_API_KEY", "test-key")
    monkeypatch.setenv("RAGFLOW_BASE_URL", "https://ragflow.example.com/app/")

    provider = RAGFlowProvider()

    assert provider._document_view_url("dataset-1", "doc-2") == (
        "https://ragflow.example.com/app/chunk/parsed/chunks?id=dataset-1&doc_id=doc-2"
    )