import { getBackendBaseURL } from "../config";

export interface RAGResource {
  uri: string;
  title: string;
  description?: string;
}

export interface RAGConfig {
  provider: string | null;
}

export interface RAGMetadataField {
  name: string;
  type: string;
}

export interface MetadataCondition {
  logic: "and" | "or";
  conditions: Array<{
    name: string;
    comparison_operator: string;
    value: string | number | boolean;
  }>;
}

export interface PerDatasetMetadataCondition {
  per_dataset: Record<string, MetadataCondition>;
}

export type MetadataConditionPayload =
  | MetadataCondition
  | PerDatasetMetadataCondition;

export async function queryRAGConfig(): Promise<RAGConfig> {
  const url = `${getBackendBaseURL()}/api/rag/config`;
  const response = await fetch(url);
  if (!response.ok) {
    return { provider: null };
  }
  return (await response.json()) as RAGConfig;
}

export async function queryRAGResources(query: string): Promise<RAGResource[]> {
  const url = `${getBackendBaseURL()}/api/rag/resources?query=${encodeURIComponent(query)}`;
  const response = await fetch(url);
  if (!response.ok) {
    return [];
  }
  const json = (await response.json()) as { resources?: RAGResource[] };
  return json.resources ?? [];
}

export async function queryRAGMetadataFields(
  uri: string,
): Promise<RAGMetadataField[]> {
  const url = `${getBackendBaseURL()}/api/rag/metadata-fields?uri=${encodeURIComponent(uri)}`;
  const response = await fetch(url);
  if (!response.ok) {
    return [];
  }
  const json = (await response.json()) as { fields?: RAGMetadataField[] };
  return json.fields ?? [];
}

export async function queryRAGMetadataFieldValues(
  uri: string,
  fieldName: string,
  limit = 100,
): Promise<string[]> {
  const url = `${getBackendBaseURL()}/api/rag/metadata-field-values?uri=${encodeURIComponent(uri)}&field_name=${encodeURIComponent(fieldName)}&limit=${encodeURIComponent(String(limit))}`;
  const response = await fetch(url);
  if (!response.ok) {
    return [];
  }
  const json = (await response.json()) as { values?: string[] };
  return Array.isArray(json.values) ? json.values : [];
}

export async function queryRAGMetadataMatchCount(
  uri: string,
  metadataCondition: MetadataCondition | null,
): Promise<number> {
  const url = `${getBackendBaseURL()}/api/rag/metadata-match-count`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      uri,
      metadata_condition: metadataCondition,
    }),
  });

  if (!response.ok) {
    return 0;
  }
  const json = (await response.json()) as { count?: number };
  return typeof json.count === "number" ? json.count : 0;
}
