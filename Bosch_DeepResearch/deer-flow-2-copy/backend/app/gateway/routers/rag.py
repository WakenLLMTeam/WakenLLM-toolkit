import logging
import os

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from deerflow.rag import MetadataField, Resource, build_retriever

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rag", tags=["rag"])


class RAGConfigResponse(BaseModel):
    provider: str | None = Field(None, description="Configured RAG provider")


class RAGResourcesResponse(BaseModel):
    resources: list[Resource] = Field(default_factory=list)


class RAGMetadataFieldsResponse(BaseModel):
    fields: list[MetadataField] = Field(default_factory=list)


class RAGMetadataFieldValuesResponse(BaseModel):
    values: list[str] = Field(default_factory=list)


class RAGMetadataMatchCountResponse(BaseModel):
    count: int = Field(0, description="Number of files satisfying metadata conditions")


@router.get("/config", response_model=RAGConfigResponse)
async def rag_config() -> RAGConfigResponse:
    return RAGConfigResponse(provider=os.getenv("RAG_PROVIDER"))


@router.get("/resources", response_model=RAGResourcesResponse)
async def rag_resources(query: str | None = Query(default=None)) -> RAGResourcesResponse:
    retriever = build_retriever()
    if retriever is None:
        return RAGResourcesResponse(resources=[])
    try:
        resources = retriever.list_resources(query)
        return RAGResourcesResponse(resources=resources)
    except Exception as exc:
        logger.warning("Failed to list RAG resources: %s", exc)
        return RAGResourcesResponse(resources=[])


@router.get("/metadata-fields", response_model=RAGMetadataFieldsResponse)
async def rag_metadata_fields(uri: str = Query(..., description="RAG resource URI")) -> RAGMetadataFieldsResponse:
    retriever = build_retriever()
    if retriever is None:
        return RAGMetadataFieldsResponse(fields=[])

    try:
        fields = retriever.list_metadata_fields(uri)
        return RAGMetadataFieldsResponse(fields=fields)
    except Exception as exc:
        logger.warning("Failed to list metadata fields for '%s': %s", uri, exc)
        return RAGMetadataFieldsResponse(fields=[])


@router.get("/metadata-field-values", response_model=RAGMetadataFieldValuesResponse)
async def rag_metadata_field_values(
    uri: str = Query(..., description="RAG resource URI"),
    field_name: str = Query(..., description="Metadata field name"),
    limit: int = Query(100, ge=1, le=500),
) -> RAGMetadataFieldValuesResponse:
    retriever = build_retriever()
    if retriever is None:
        return RAGMetadataFieldValuesResponse(values=[])

    try:
        values = retriever.list_metadata_field_values(uri, field_name, limit)
        return RAGMetadataFieldValuesResponse(values=values)
    except Exception as exc:
        logger.warning(
            "Failed to list metadata field values for '%s'/'%s': %s",
            uri,
            field_name,
            exc,
        )
        return RAGMetadataFieldValuesResponse(values=[])


@router.post("/metadata-match-count", response_model=RAGMetadataMatchCountResponse)
async def rag_metadata_match_count(payload: dict) -> RAGMetadataMatchCountResponse:
    uri = payload.get("uri")
    metadata_condition = payload.get("metadata_condition")

    if not isinstance(uri, str) or not uri.strip():
        return RAGMetadataMatchCountResponse(count=0)

    retriever = build_retriever()
    if retriever is None:
        return RAGMetadataMatchCountResponse(count=0)

    try:
        count = retriever.count_matching_documents(uri, metadata_condition)
        return RAGMetadataMatchCountResponse(count=max(0, int(count)))
    except Exception as exc:
        logger.warning("Failed to count metadata matches for '%s': %s", uri, exc)
        return RAGMetadataMatchCountResponse(count=0)
