import logging
from typing import Type

from langchain_core.tools import ArgsSchema, BaseTool
from pydantic import BaseModel, Field

from deerflow.rag import MetadataCondition, Resource, Retriever, build_retriever

logger = logging.getLogger(__name__)


class RetrieverInput(BaseModel):
    keywords: str = Field(description="Search keywords to look up")


class RetrieverTool(BaseTool):
    name: str = "RAG_retrieval"
    description: str = "Retrieve information from local RAG resources."
    args_schema: ArgsSchema = RetrieverInput

    retriever: Retriever = Field(...)
    resources: list[Resource] = Field(default_factory=list)
    metadata_condition: MetadataCondition | None = Field(default=None)

    def _run(self, keywords: str) -> list[dict] | str:
        logger.info("RAG_retrieval query: %s", keywords)
        documents = self.retriever.query_relevant_documents(
            keywords,
            self.resources,
            self.metadata_condition,
        )
        if not documents:
            return "No results found from the local knowledge base."
        return [doc.to_dict() for doc in documents]

    async def _arun(self, keywords: str) -> list[dict] | str:
        logger.info("RAG_retrieval async query: %s", keywords)
        documents = await self.retriever.query_relevant_documents_async(
            keywords,
            self.resources,
            self.metadata_condition,
        )
        if not documents:
            return "No results found from the local knowledge base."
        return [doc.to_dict() for doc in documents]


def get_retriever_tool(
    resources: list[Resource],
    metadata_condition: MetadataCondition | None = None,
) -> RetrieverTool | None:
    if not resources:
        return None
    retriever = build_retriever()
    if retriever is None:
        return None
    return RetrieverTool(
        retriever=retriever,
        resources=resources,
        metadata_condition=metadata_condition,
    )
