"use client";

import { FilterIcon } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { PromptInputButton } from "@/components/ai-elements/prompt-input";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { MetadataCondition, RAGMetadataField } from "@/core/rag/api";
import { cn } from "@/lib/utils";

const STRING_OPERATORS = [
  "=",
  "≠",
  "contains",
  "not contains",
  "start with",
  "empty",
  "not empty",
] as const;

const NUMBER_OPERATORS = [
  "=",
  "≠",
  ">",
  "<",
  "≥",
  "≤",
  "empty",
  "not empty",
] as const;

type MetadataOperator =
  | (typeof STRING_OPERATORS)[number]
  | (typeof NUMBER_OPERATORS)[number];

type FieldState = Record<
  string,
  {
    operator: MetadataOperator;
    value: string;
  }
>;

export type DatasetFilterTab = {
  uri: string;
  datasetId: string;
  title: string;
  fields: RAGMetadataField[];
  value: MetadataCondition | null;
};

type MetadataFilterEditorProps = {
  datasets: DatasetFilterTab[];
  counts: Record<string, number | null>;
  fieldValuesByDataset: Record<string, Record<string, string[]>>;
  onLoadFieldValues: (datasetUri: string, fieldName: string) => Promise<void>;
  onPreviewCount: (
    datasetUri: string,
    metadataCondition: MetadataCondition | null,
  ) => Promise<number>;
  onApply: (valueByDataset: Record<string, MetadataCondition | null>) => void;
  onApplyComplete?: () => void;
};

function operatorNeedsValue(operator: string) {
  return operator !== "empty" && operator !== "not empty";
}

function operatorsForField(field: RAGMetadataField): readonly MetadataOperator[] {
  if (field.type === "number") {
    return NUMBER_OPERATORS;
  }
  return STRING_OPERATORS;
}

function buildFieldState(
  fields: RAGMetadataField[],
  condition: MetadataCondition | null,
): FieldState {
  const state: FieldState = {};

  for (const field of fields) {
    const operators = operatorsForField(field);
    const firstOperator: MetadataOperator = operators[0] ?? "=";
    state[field.name] = {
      operator: firstOperator,
      value: "",
    };
  }

  if (!condition?.conditions?.length) {
    return state;
  }

  for (const item of condition.conditions) {
    if (!state[item.name]) {
      continue;
    }
    state[item.name] = {
      operator: item.comparison_operator as MetadataOperator,
      value: item.value != null ? String(item.value) : "",
    };
  }

  return state;
}

function conditionFromState(
  logic: "and" | "or",
  fields: RAGMetadataField[],
  fieldState: FieldState,
): MetadataCondition | null {
  const conditions: MetadataCondition["conditions"] = [];

  for (const field of fields) {
    const state = fieldState[field.name];
    if (!state) {
      continue;
    }

    const needsValue = operatorNeedsValue(state.operator);
    const trimmed = state.value.trim();

    if (needsValue && !trimmed) {
      continue;
    }

    if (field.type === "number" && needsValue && Number.isNaN(Number(trimmed))) {
      continue;
    }

    conditions.push({
      name: field.name,
      comparison_operator: state.operator,
      value: field.type === "number" && needsValue ? Number(trimmed) : trimmed,
    });
  }

  if (!conditions.length) {
    return null;
  }

  return {
    logic,
    conditions,
  };
}

export function MetadataFilterEditor({
  datasets,
  counts,
  fieldValuesByDataset,
  onLoadFieldValues,
  onPreviewCount,
  onApply,
  onApplyComplete,
}: MetadataFilterEditorProps) {
  const [activeDatasetUri, setActiveDatasetUri] = useState<string>(datasets[0]?.uri ?? "");
  const [logicByDataset, setLogicByDataset] =
    useState<Record<string, "and" | "or">>({});
  const [fieldStateByDataset, setFieldStateByDataset] =
    useState<Record<string, FieldState>>({});
  const [countByDataset, setCountByDataset] = useState<Record<string, number | null>>({});
  const [applyAllWarning, setApplyAllWarning] = useState<string>("");

  const hasAnyActiveFilters = useMemo(
    () => datasets.some((dataset) => Boolean(dataset.value?.conditions?.length)),
    [datasets],
  );

  useEffect(() => {
    const nextLogic: Record<string, "and" | "or"> = {};
    const nextState: Record<string, FieldState> = {};

    for (const dataset of datasets) {
      nextLogic[dataset.uri] = dataset.value?.logic ?? "and";
      nextState[dataset.uri] = buildFieldState(dataset.fields, dataset.value);
    }

    setLogicByDataset(nextLogic);
    setFieldStateByDataset(nextState);
    if (datasets.length && !datasets.some((item) => item.uri === activeDatasetUri)) {
      setActiveDatasetUri(datasets[0]!.uri);
    }
  }, [datasets, activeDatasetUri]);

  useEffect(() => {
    setCountByDataset((prev) => {
      const next: Record<string, number | null> = {};
      for (const dataset of datasets) {
        next[dataset.uri] = counts[dataset.uri] ?? null;
      }
      const prevKeys = Object.keys(prev);
      if (
        prevKeys.length === Object.keys(next).length &&
        prevKeys.every((key) => prev[key] === next[key])
      ) {
        return prev;
      }
      return next;
    });
  }, [datasets, counts]);

  const activeDataset =
    datasets.find((dataset) => dataset.uri === activeDatasetUri) ?? datasets[0] ?? null;

  useEffect(() => {
    if (!activeDataset) {
      return;
    }
    for (const field of activeDataset.fields) {
      const existing = fieldValuesByDataset[activeDataset.uri]?.[field.name];
      if (existing !== undefined) {
        continue;
      }
      void onLoadFieldValues(activeDataset.uri, field.name);
    }
  }, [activeDataset, fieldValuesByDataset, onLoadFieldValues]);

  useEffect(() => {
    if (!activeDataset) {
      return;
    }

    const logic = logicByDataset[activeDataset.uri] ?? "and";
    const state = fieldStateByDataset[activeDataset.uri] ?? {};
    const condition = conditionFromState(logic, activeDataset.fields, state);

    let cancelled = false;
    const timer = setTimeout(() => {
      void onPreviewCount(activeDataset.uri, condition).then((count) => {
        if (cancelled) {
          return;
        }
        setCountByDataset((prev) => ({
          ...prev,
          [activeDataset.uri]: count,
        }));
      });
    }, 250);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [activeDataset, logicByDataset, fieldStateByDataset, onPreviewCount]);

  if (!datasets.length) {
    return null;
  }

  const handleApply = () => {
    const output: Record<string, MetadataCondition | null> = {};
    for (const dataset of datasets) {
      const logic = logicByDataset[dataset.uri] ?? "and";
      const state = fieldStateByDataset[dataset.uri] ?? {};
      output[dataset.uri] = conditionFromState(logic, dataset.fields, state);
    }
    onApply(output);
    onApplyComplete?.();
  };

  const handleClearCurrent = () => {
    if (!activeDataset) {
      return;
    }
    setFieldStateByDataset((prev) => ({
      ...prev,
      [activeDataset.uri]: buildFieldState(activeDataset.fields, null),
    }));
    setLogicByDataset((prev) => ({
      ...prev,
      [activeDataset.uri]: "and",
    }));
    setCountByDataset((prev) => ({
      ...prev,
      [activeDataset.uri]: null,
    }));
    setApplyAllWarning("");
  };

  const handleApplyToAll = () => {
    if (!activeDataset) {
      return;
    }

    const sourceLogic = logicByDataset[activeDataset.uri] ?? "and";
    const sourceState = fieldStateByDataset[activeDataset.uri] ?? {};
    const sourceCondition = conditionFromState(
      sourceLogic,
      activeDataset.fields,
      sourceState,
    );

    const sourceFieldNames = new Set(
      sourceCondition?.conditions.map((item) => item.name) ?? [],
    );

    const incompatibleTitles: string[] = [];
    const nextByDataset: Record<string, MetadataCondition | null> = {
      [activeDataset.uri]: sourceCondition,
    };

    for (const dataset of datasets) {
      if (dataset.uri === activeDataset.uri) {
        continue;
      }
      const targetFields = new Set(dataset.fields.map((field) => field.name));
      const missingField = Array.from(sourceFieldNames).find(
        (field) => !targetFields.has(field),
      );
      if (missingField) {
        incompatibleTitles.push(dataset.title);
        continue;
      }
      nextByDataset[dataset.uri] = sourceCondition;
    }

    setLogicByDataset((prev) => {
      const next = { ...prev };
      for (const dataset of datasets) {
        if (nextByDataset[dataset.uri] !== undefined) {
          next[dataset.uri] = sourceLogic;
        }
      }
      return next;
    });

    setFieldStateByDataset((prev) => {
      const next = { ...prev };
      for (const dataset of datasets) {
        if (nextByDataset[dataset.uri] === undefined) {
          continue;
        }
        next[dataset.uri] = buildFieldState(dataset.fields, sourceCondition);
      }
      return next;
    });

    if (incompatibleTitles.length > 0) {
      setApplyAllWarning(
        `Warning: Could not apply to ${incompatibleTitles.join(", ")} because required metadata fields are missing.`,
      );
    } else {
      setApplyAllWarning("");
    }
  };

  return (
    <>
      <Tabs value={activeDataset?.uri} onValueChange={setActiveDatasetUri}>
        <TabsList className="mb-4 flex w-full justify-start overflow-x-auto">
          {datasets.map((dataset) => (
            <TabsTrigger
              key={dataset.uri}
              value={dataset.uri}
              className="max-w-[200px] truncate"
            >
              {dataset.title}
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>

      {activeDataset && (
        <>
          <div className="mb-2 flex items-center gap-3">
            <div className="text-sm font-medium">Combine rules</div>
            <Select
              value={logicByDataset[activeDataset.uri] ?? "and"}
              onValueChange={(value) =>
                setLogicByDataset((prev) => ({
                  ...prev,
                  [activeDataset.uri]: value as "and" | "or",
                }))
              }
            >
              <SelectTrigger className="h-9 w-[220px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="and">All conditions (AND)</SelectItem>
                <SelectItem value="or">Any condition (OR)</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="text-muted-foreground mb-3 text-xs">
            {(() => {
              const count = countByDataset[activeDataset.uri];
              if (count == null) {
                return "There are -- files satisfying these conditions.";
              }
              return count === 1
                ? "There is 1 file satisfying these conditions."
                : `There are ${count} files satisfying these conditions.`;
            })()}
          </div>

          <div className="max-h-[50vh] overflow-y-auto pr-1">
            <div className="grid gap-3">
              {activeDataset.fields.length === 0 ? (
                <div className="text-muted-foreground rounded-md border p-3 text-sm">
                  This dataset has no available meta-data fields.
                </div>
              ) : (
                activeDataset.fields.map((field) => {
                  const operators = operatorsForField(field);
                  const firstOperator: MetadataOperator = operators[0] ?? "=";
                  const state =
                    fieldStateByDataset[activeDataset.uri]?.[field.name] ?? {
                      operator: firstOperator,
                      value: "",
                    };

                  return (
                    <div
                      key={field.name}
                      className="grid gap-2 rounded-lg border px-3 py-3"
                    >
                      <div className="flex flex-wrap items-center gap-3">
                        <div className="w-[180px] text-sm font-medium">{field.name}</div>
                        <Select
                          value={state.operator}
                          onValueChange={(value) =>
                            setFieldStateByDataset((prev) => ({
                              ...prev,
                              [activeDataset.uri]: {
                                ...(prev[activeDataset.uri] ?? {}),
                                [field.name]: {
                                  operator: value as MetadataOperator,
                                  value:
                                    prev[activeDataset.uri]?.[field.name]?.value ?? "",
                                },
                              },
                            }))
                          }
                        >
                          <SelectTrigger className="h-9 w-[170px]">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {operators.map((operator) => (
                              <SelectItem key={operator} value={operator}>
                                {operator}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        <Input
                          className={cn("h-9 flex-1")}
                          disabled={!operatorNeedsValue(state.operator)}
                          list={`metadata-values-${activeDataset.uri}-${field.name}`}
                          placeholder={field.type === "number" ? "e.g. 42" : "Enter value"}
                          value={state.value}
                          onChange={(event) =>
                            setFieldStateByDataset((prev) => ({
                              ...prev,
                              [activeDataset.uri]: {
                                ...(prev[activeDataset.uri] ?? {}),
                                [field.name]: {
                                  operator:
                                    prev[activeDataset.uri]?.[field.name]?.operator ??
                                    firstOperator,
                                  value: event.target.value,
                                },
                              },
                            }))
                          }
                        />
                        <datalist id={`metadata-values-${activeDataset.uri}-${field.name}`}>
                          {(fieldValuesByDataset[activeDataset.uri]?.[field.name] ?? []).map(
                            (value) => (
                              <option key={value} value={value} />
                            ),
                          )}
                        </datalist>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </>
      )}

      {applyAllWarning && <div className="mt-3 text-sm text-amber-600">{applyAllWarning}</div>}

      <div className="mt-4 flex items-center justify-end gap-2">
        <Button variant="ghost" onClick={handleClearCurrent}>
          Clear current tab
        </Button>
        <Button variant="outline" onClick={handleApplyToAll}>
          Apply to all
        </Button>
        <Button onClick={handleApply}>Apply filters</Button>
      </div>
    </>
  );
}

export function MetadataFilterDialog({
  datasets,
  counts,
  fieldValuesByDataset,
  onLoadFieldValues,
  onPreviewCount,
  onApply,
}: MetadataFilterEditorProps) {
  const [open, setOpen] = useState(false);
  const hasAnyActiveFilters = useMemo(
    () => datasets.some((dataset) => Boolean(dataset.value?.conditions?.length)),
    [datasets],
  );

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <PromptInputButton
          className={cn(
            "gap-1 px-2! text-xs font-normal",
            hasAnyActiveFilters && "text-primary",
          )}
        >
          <FilterIcon className="size-3" />
          Meta-data Filter
        </PromptInputButton>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[760px]">
        <DialogHeader>
          <DialogTitle>Meta-data filters</DialogTitle>
          <DialogDescription>
            Configure filters per selected dataset. Each subtab controls one dataset.
          </DialogDescription>
        </DialogHeader>
        <MetadataFilterEditor
          datasets={datasets}
          counts={counts}
          fieldValuesByDataset={fieldValuesByDataset}
          onLoadFieldValues={onLoadFieldValues}
          onPreviewCount={onPreviewCount}
          onApply={onApply}
          onApplyComplete={() => setOpen(false)}
        />
      </DialogContent>
    </Dialog>
  );
}
