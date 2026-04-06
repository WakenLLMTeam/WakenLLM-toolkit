import abc
from typing import Any

from pydantic import BaseModel, Field


class Chunk:
    content: str
    similarity: float

    def __init__(self, content: str, similarity: float):
        self.content = content
        self.similarity = similarity


class Document:
    id: str
    dataset_id: str | None = None
    url: str | None = None
    title: str | None = None
    chunks: list[Chunk] = []

    def __init__(
        self,
        id: str,
        url: str | None = None,
        title: str | None = None,
        chunks: list[Chunk] = [],
    ):
        self.id = id
        self.url = url
        self.title = title
        self.chunks = chunks

    def to_dict(self) -> dict:
        data = {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "content": "\n\n".join([chunk.content for chunk in self.chunks]),
        }
        if self.url:
            data["url"] = self.url
        if self.title:
            data["title"] = self.title
        return data


class Resource(BaseModel):
    uri: str = Field(..., description="The URI of the resource")
    title: str = Field(..., description="The title of the resource")
    description: str | None = Field("", description="The description of the resource")


class MetadataField(BaseModel):
    name: str = Field(..., description="Metadata field name")
    type: str = Field("string", description="Metadata field type")


MetadataCondition = dict[str, Any]


class Retriever(abc.ABC):
    @abc.abstractmethod
    def list_resources(self, query: str | None = None) -> list[Resource]:
        pass

    @abc.abstractmethod
    async def list_resources_async(self, query: str | None = None) -> list[Resource]:
        pass

    @abc.abstractmethod
    def query_relevant_documents(
        self,
        query: str,
        resources: list[Resource] = [],
        metadata_condition: MetadataCondition | None = None,
    ) -> list[Document]:
        pass

    @abc.abstractmethod
    async def query_relevant_documents_async(
        self,
        query: str,
        resources: list[Resource] = [],
        metadata_condition: MetadataCondition | None = None,
    ) -> list[Document]:
        pass

    @abc.abstractmethod
    def list_metadata_fields(self, resource_uri: str) -> list[MetadataField]:
        pass

    @abc.abstractmethod
    async def list_metadata_fields_async(self, resource_uri: str) -> list[MetadataField]:
        pass

    @abc.abstractmethod
    def list_metadata_field_values(
        self,
        resource_uri: str,
        field_name: str,
        limit: int = 100,
    ) -> list[str]:
        pass

    @abc.abstractmethod
    async def list_metadata_field_values_async(
        self,
        resource_uri: str,
        field_name: str,
        limit: int = 100,
    ) -> list[str]:
        pass

    @abc.abstractmethod
    def count_matching_documents(
        self,
        resource_uri: str,
        metadata_condition: MetadataCondition | None = None,
    ) -> int:
        pass

    @abc.abstractmethod
    async def count_matching_documents_async(
        self,
        resource_uri: str,
        metadata_condition: MetadataCondition | None = None,
    ) -> int:
        pass
