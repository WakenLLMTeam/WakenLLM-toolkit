import asyncio
import json
import os
from urllib.parse import quote_plus, urlparse
from urllib.request import Request, urlopen

from deerflow.rag.retriever import Chunk, Document, MetadataCondition, MetadataField, Resource, Retriever


class RAGFlowProvider(Retriever):
    api_url: str
    api_key: str
    page_size: int = 10

    def __init__(self):
        api_url = os.getenv("RAGFLOW_API_URL")
        if not api_url:
            raise ValueError("RAGFLOW_API_URL is not set")
        self.api_url = api_url.rstrip("/")

        api_key = os.getenv("RAGFLOW_API_KEY")
        if not api_key:
            raise ValueError("RAGFLOW_API_KEY is not set")
        self.api_key = api_key

        page_size = os.getenv("RAGFLOW_PAGE_SIZE")
        if page_size:
            self.page_size = int(page_size)

    def _ragflow_base_url(self) -> str:
        base_url = (os.getenv("RAGFLOW_BASE_URL") or "").strip()
        if base_url:
            return base_url.rstrip("/")

        parsed = urlparse(self.api_url)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
        return self.api_url.rstrip("/")

    def _document_view_url(self, dataset_id: str | None, document_id: str | None) -> str | None:
        if not dataset_id or not document_id:
            return None
        return f"{self._ragflow_base_url()}/chunk/parsed/chunks?id={dataset_id}&doc_id={document_id}"

    def query_relevant_documents(
        self,
        query: str,
        resources: list[Resource] = [],
        metadata_condition: MetadataCondition | None = None,
    ) -> list[Document]:
        dataset_ids: list[str] = []
        document_ids: list[str] = []
        document_ids_by_dataset: dict[str, list[str]] = {}
        for resource in resources:
            dataset_id, document_id = parse_uri(resource.uri)
            dataset_ids.append(dataset_id)
            if document_id:
                document_ids.append(document_id)
                document_ids_by_dataset.setdefault(dataset_id, []).append(document_id)

        per_dataset_condition = None
        if isinstance(metadata_condition, dict):
            maybe_map = metadata_condition.get("per_dataset")
            if isinstance(maybe_map, dict):
                per_dataset_condition = maybe_map

        docs: dict[str, Document] = {}

        if per_dataset_condition:
            unique_dataset_ids = list(dict.fromkeys(dataset_ids))
            for dataset_id in unique_dataset_ids:
                payload = {
                    "question": query,
                    "dataset_ids": [dataset_id],
                    "document_ids": document_ids_by_dataset.get(dataset_id, []),
                    "page_size": self.page_size,
                    "toc_enhance": True,
                }

                dataset_condition = per_dataset_condition.get(dataset_id)
                if dataset_condition:
                    payload["metadata_condition"] = dataset_condition

                data = self._post_retrieval(payload)
                self._merge_retrieval_result(data, docs)

            return list(docs.values())

        payload = {
            "question": query,
            "dataset_ids": dataset_ids,
            "document_ids": document_ids,
            "page_size": self.page_size,
            "toc_enhance": True,
        }
        if metadata_condition and not per_dataset_condition:
            payload["metadata_condition"] = metadata_condition

        data = self._post_retrieval(payload)
        self._merge_retrieval_result(data, docs)

        return list(docs.values())

    async def query_relevant_documents_async(
        self,
        query: str,
        resources: list[Resource] = [],
        metadata_condition: MetadataCondition | None = None,
    ) -> list[Document]:
        return await asyncio.to_thread(
            self.query_relevant_documents,
            query,
            resources,
            metadata_condition,
        )

    def list_resources(self, query: str | None = None) -> list[Resource]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        request_url = f"{self.api_url}/api/v1/datasets"
        if query:
            request_url = f"{request_url}?name={quote_plus(query)}"

        request = Request(request_url, headers=headers, method="GET")
        with urlopen(request, timeout=20) as response:
            status_code = response.getcode()
            response_body = response.read().decode("utf-8")

        if status_code != 200:
            raise Exception(f"Failed to list resources: {response_body}")

        result = json.loads(response_body)
        resources: list[Resource] = []
        for item in result.get("data", []):
            resources.append(
                Resource(
                    uri=f"rag://dataset/{item.get('id')}",
                    title=item.get("name", ""),
                    description=item.get("description", ""),
                )
            )
        return resources

    async def list_resources_async(self, query: str | None = None) -> list[Resource]:
        return await asyncio.to_thread(self.list_resources, query)

    def list_metadata_fields(self, resource_uri: str) -> list[MetadataField]:
        dataset_id, document_id = parse_uri(resource_uri)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"doc_ids": [document_id]} if document_id else {}

        request = Request(
            f"{self.api_url}/api/v1/datasets/{dataset_id}/metadata/summary",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="GET",
        )
        with urlopen(request, timeout=20) as response:
            status_code = response.getcode()
            response_body = response.read().decode("utf-8")

        if status_code != 200:
            raise Exception(f"Failed to list metadata fields: {response_body}")

        result = json.loads(response_body)
        summary = (result.get("data") or {}).get("summary") or {}
        fields: list[MetadataField] = []
        for name, item in summary.items():
            if not isinstance(name, str) or not name.strip():
                continue
            field_type = "string"
            if isinstance(item, dict):
                field_type = str(item.get("type") or "string")
            fields.append(MetadataField(name=name, type=field_type))
        return fields

    async def list_metadata_fields_async(self, resource_uri: str) -> list[MetadataField]:
        return await asyncio.to_thread(self.list_metadata_fields, resource_uri)

    def list_metadata_field_values(
        self,
        resource_uri: str,
        field_name: str,
        limit: int = 100,
    ) -> list[str]:
        dataset_id, document_id = parse_uri(resource_uri)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"doc_ids": [document_id]} if document_id else {}

        request = Request(
            f"{self.api_url}/api/v1/datasets/{dataset_id}/metadata/summary",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="GET",
        )
        with urlopen(request, timeout=20) as response:
            status_code = response.getcode()
            response_body = response.read().decode("utf-8")

        if status_code != 200:
            raise Exception(f"Failed to list metadata field values: {response_body}")

        result = json.loads(response_body)
        summary = (result.get("data") or {}).get("summary") or {}
        item = summary.get(field_name)
        if not isinstance(item, dict):
            return []

        def normalize_value(raw_value: object) -> str | None:
            if raw_value is None:
                return None
            if isinstance(raw_value, str):
                text = raw_value.strip()
                return text or None
            if isinstance(raw_value, (int, float, bool)):
                return str(raw_value)
            if isinstance(raw_value, (list, tuple)):
                if not raw_value:
                    return None
                return normalize_value(raw_value[0])
            if isinstance(raw_value, dict):
                for key in ("name", "value", "label", "key"):
                    if key in raw_value:
                        return normalize_value(raw_value.get(key))
                return None
            return None

        values: list[str] = []
        for key in ("values", "distinct_values", "sample_values", "examples"):
            raw = item.get(key)
            if isinstance(raw, list):
                for value in raw:
                    text = normalize_value(value)
                    if text:
                        values.append(text)

        deduped: list[str] = []
        seen = set()
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            deduped.append(value)

        return deduped[: max(1, int(limit))]

    async def list_metadata_field_values_async(
        self,
        resource_uri: str,
        field_name: str,
        limit: int = 100,
    ) -> list[str]:
        return await asyncio.to_thread(
            self.list_metadata_field_values,
            resource_uri,
            field_name,
            limit,
        )

    def count_matching_documents(
        self,
        resource_uri: str,
        metadata_condition: MetadataCondition | None = None,
    ) -> int:
        dataset_id, document_id = parse_uri(resource_uri)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        params = [
            "page=1",
            "page_size=1",
        ]
        if document_id:
            params.append(f"id={quote_plus(document_id)}")
        if metadata_condition:
            params.append(
                f"metadata_condition={quote_plus(json.dumps(metadata_condition, ensure_ascii=False))}"
            )

        request_url = f"{self.api_url}/api/v1/datasets/{dataset_id}/documents?{'&'.join(params)}"
        request = Request(request_url, headers=headers, method="GET")
        with urlopen(request, timeout=20) as response:
            status_code = response.getcode()
            response_body = response.read().decode("utf-8")

        if status_code != 200:
            raise Exception(f"Failed to count matching documents: {response_body}")

        result = json.loads(response_body)
        data = result.get("data") or {}
        total = data.get("total")
        if isinstance(total, int):
            return total
        return 0

    async def count_matching_documents_async(
        self,
        resource_uri: str,
        metadata_condition: MetadataCondition | None = None,
    ) -> int:
        return await asyncio.to_thread(
            self.count_matching_documents,
            resource_uri,
            metadata_condition,
        )

    def _post_retrieval(self, payload: dict) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        request = Request(
            f"{self.api_url}/api/v1/retrieval",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=30) as response:
            status_code = response.getcode()
            response_body = response.read().decode("utf-8")

        if status_code != 200:
            raise Exception(f"Failed to query documents: {response_body}")

        result = json.loads(response_body)
        return result.get("data") or {}

    def _merge_retrieval_result(self, data: dict, docs: dict[str, Document]) -> None:
        doc_titles: dict[str, str] = {}
        for doc in data.get("doc_aggs", []):
            doc_id = doc.get("doc_id")
            if not doc_id:
                continue
            doc_name = doc.get("doc_name")
            if isinstance(doc_name, str) and doc_name:
                doc_titles[doc_id] = doc_name

        for chunk in data.get("chunks", []):
            doc_id = chunk.get("document_id")
            if not doc_id:
                continue
            dataset_id = chunk.get("dataset_id")
            doc = docs.get(doc_id)
            if not doc:
                doc = Document(
                    id=doc_id,
                    url=self._document_view_url(dataset_id, doc_id),
                    title=doc_titles.get(doc_id),
                    chunks=[],
                )
                docs[doc_id] = doc
            if getattr(doc, "title", None) is None and doc_titles.get(doc_id):
                setattr(doc, "title", doc_titles[doc_id])
            if getattr(doc, "dataset_id", None) is None and dataset_id:
                setattr(doc, "dataset_id", dataset_id)
            if getattr(doc, "url", None) is None:
                setattr(doc, "url", self._document_view_url(getattr(doc, "dataset_id", None), doc_id))
            doc.chunks.append(
                Chunk(
                    content=chunk.get("content", ""),
                    similarity=chunk.get("similarity", 0.0),
                )
            )


def parse_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "rag":
        raise ValueError(f"Invalid URI: {uri}")
    return parsed.path.split("/")[1], parsed.fragment