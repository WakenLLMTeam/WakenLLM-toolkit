"use client";

import type { ChatStatus } from "ai";
import {
  CheckIcon,
  DatabaseIcon,
  GraduationCapIcon,
  LightbulbIcon,
  PaperclipIcon,
  PlusIcon,
  SparklesIcon,
  SearchIcon,
  RocketIcon,
  UploadIcon,
  XIcon,
  ZapIcon,
} from "lucide-react";
import { useSearchParams } from "next/navigation";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ComponentProps,
  type ReactNode,
} from "react";

import {
  PromptInput,
  PromptInputActionMenu,
  PromptInputActionMenuContent,
  PromptInputActionMenuItem,
  PromptInputActionMenuTrigger,
  PromptInputAttachment,
  PromptInputAttachments,
  PromptInputBody,
  PromptInputButton,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputTools,
  usePromptInputAttachments,
  usePromptInputController,
  type PromptInputMessage,
} from "@/components/ai-elements/prompt-input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ConfettiButton } from "@/components/ui/confetti-button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenuGroup,
  DropdownMenuLabel,
  DropdownMenuItem,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { getBackendBaseURL } from "@/core/config";
import { useI18n } from "@/core/i18n/hooks";
import { useModels } from "@/core/models/hooks";
import {
  type MetadataCondition,
  type MetadataConditionPayload,
  queryRAGConfig,
  queryRAGMetadataFieldValues,
  queryRAGMetadataFields,
  queryRAGMetadataMatchCount,
  queryRAGResources,
  type RAGMetadataField,
  type RAGResource,
} from "@/core/rag/api";
import type { AgentThreadContext } from "@/core/threads";
import { textOfMessage } from "@/core/threads/utils";
import { cn } from "@/lib/utils";

import {
  ModelSelector,
  ModelSelectorContent,
  ModelSelectorInput,
  ModelSelectorItem,
  ModelSelectorList,
  ModelSelectorName,
  ModelSelectorTrigger,
} from "../ai-elements/model-selector";
import { Suggestion, Suggestions } from "../ai-elements/suggestion";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from "../ui/dropdown-menu";

import { useThread } from "./messages/context";
import { ModeHoverGuide } from "./mode-hover-guide";
import {
  MetadataFilterDialog,
  MetadataFilterEditor,
} from "./metadata-filter-dialog";
import { Tooltip } from "./tooltip";

type InputMode = "flash" | "thinking" | "pro" | "ultra";

function getResolvedMode(
  mode: InputMode | undefined,
  supportsThinking: boolean,
): InputMode {
  if (!supportsThinking && mode !== "flash") {
    return "flash";
  }
  if (mode) {
    return mode;
  }
  return supportsThinking ? "pro" : "flash";
}

export function InputBox({
  className,
  disabled,
  autoFocus,
  status = "ready",
  context,
  extraHeader,
  isNewThread,
  threadId,
  initialValue,
  onContextChange,
  onSubmit,
  onStop,
  ...props
}: Omit<ComponentProps<typeof PromptInput>, "onSubmit"> & {
  assistantId?: string | null;
  status?: ChatStatus;
  disabled?: boolean;
  context: Omit<
    AgentThreadContext,
    "thread_id" | "is_plan_mode" | "thinking_enabled" | "subagent_enabled"
  > & {
    mode: "flash" | "thinking" | "pro" | "ultra" | undefined;
    reasoning_effort?: "minimal" | "low" | "medium" | "high";
  };
  extraHeader?: ReactNode;
  isNewThread?: boolean;
  threadId: string;
  initialValue?: string;
  onContextChange?: (
    context: Omit<
      AgentThreadContext,
      "thread_id" | "is_plan_mode" | "thinking_enabled" | "subagent_enabled"
    > & {
      mode: "flash" | "thinking" | "pro" | "ultra" | undefined;
      reasoning_effort?: "minimal" | "low" | "medium" | "high";
    },
  ) => void;
  onSubmit?: (message: PromptInputMessage) => void;
  onStop?: () => void;
}) {
  const { t } = useI18n();
  const searchParams = useSearchParams();
  const { textInput } = usePromptInputController();
  const attachments = usePromptInputAttachments();
  const [modelDialogOpen, setModelDialogOpen] = useState(false);
  const { models } = useModels();
  const [ragProvider, setRagProvider] = useState<string | null>(null);
  const [ragSuggestions, setRagSuggestions] = useState<RAGResource[]>([]);
  const [ragPickerOpen, setRagPickerOpen] = useState(false);
  const [ragLoading, setRagLoading] = useState(false);
  const [ragDialogOpen, setRagDialogOpen] = useState(false);
  const [ragDialogTab, setRagDialogTab] = useState<"select" | "metadata">(
    "select",
  );
  const [ragDialogQuery, setRagDialogQuery] = useState("");
  const [ragDialogLoading, setRagDialogLoading] = useState(false);
  const [ragDialogResources, setRagDialogResources] = useState<RAGResource[]>([]);
  const [selectedRagResources, setSelectedRagResources] = useState<
    RAGResource[]
  >([]);
  const [ragDialogSelectedResources, setRagDialogSelectedResources] = useState<
    RAGResource[]
  >([]);
  const [metadataFieldsByUri, setMetadataFieldsByUri] =
    useState<Record<string, RAGMetadataField[]>>({});
  const [metadataConditionByUri, setMetadataConditionByUri] =
    useState<Record<string, MetadataCondition | null>>({});
  const [ragDialogMetadataConditionByUri, setRagDialogMetadataConditionByUri] =
    useState<Record<string, MetadataCondition | null>>({});
  const [metadataCountByUri, setMetadataCountByUri] =
    useState<Record<string, number | null>>({});
  const [metadataFieldValuesByUri, setMetadataFieldValuesByUri] =
    useState<Record<string, Record<string, string[]>>>({});
  const ragQueryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const ragDialogQueryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(
    null,
  );

  const isRagEnabled = useMemo(() => !!ragProvider, [ragProvider]);

  const activeMentionQuery = useMemo(() => {
    if (!isRagEnabled) {
      return null;
    }
    const match = textInput.value.match(/(?:^|\s)@([\w.-]*)$/);
    if (!match) {
      return null;
    }
    return match[1] ?? "";
  }, [isRagEnabled, textInput.value]);

  const metadataBoundResources = useMemo(
    () =>
      uniqueRagResources([
        ...selectedRagResources,
        ...ragDialogSelectedResources,
      ]),
    [selectedRagResources, ragDialogSelectedResources],
  );

  useEffect(() => {
    let mounted = true;
    const loadRagConfig = async () => {
      try {
        const config = await queryRAGConfig();
        if (mounted) {
          setRagProvider(config.provider);
        }
      } catch {
        if (mounted) {
          setRagProvider(null);
        }
      }
    };
    void loadRagConfig();
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    if (!isRagEnabled || activeMentionQuery === null) {
      setRagPickerOpen(false);
      setRagSuggestions([]);
      setRagLoading(false);
      if (ragQueryTimerRef.current) {
        clearTimeout(ragQueryTimerRef.current);
      }
      return;
    }

    setRagPickerOpen(true);
    setRagLoading(true);
    if (ragQueryTimerRef.current) {
      clearTimeout(ragQueryTimerRef.current);
    }

    ragQueryTimerRef.current = setTimeout(() => {
      void queryRAGResources(activeMentionQuery)
        .then((resources) => {
          setRagSuggestions(resources);
        })
        .catch(() => {
          setRagSuggestions([]);
        })
        .finally(() => {
          setRagLoading(false);
        });
    }, 150);

    return () => {
      if (ragQueryTimerRef.current) {
        clearTimeout(ragQueryTimerRef.current);
      }
    };
  }, [activeMentionQuery, isRagEnabled]);

  useEffect(() => {
    if (!ragDialogOpen || !isRagEnabled) {
      setRagDialogLoading(false);
      if (ragDialogQueryTimerRef.current) {
        clearTimeout(ragDialogQueryTimerRef.current);
      }
      return;
    }

    setRagDialogLoading(true);
    if (ragDialogQueryTimerRef.current) {
      clearTimeout(ragDialogQueryTimerRef.current);
    }

    ragDialogQueryTimerRef.current = setTimeout(() => {
      void queryRAGResources(ragDialogQuery.trim())
        .then((resources) => {
          setRagDialogResources(resources);
        })
        .catch(() => {
          setRagDialogResources([]);
        })
        .finally(() => {
          setRagDialogLoading(false);
        });
    }, 150);

    return () => {
      if (ragDialogQueryTimerRef.current) {
        clearTimeout(ragDialogQueryTimerRef.current);
      }
    };
  }, [ragDialogOpen, ragDialogQuery, isRagEnabled]);

  useEffect(() => {
    const activeUris = new Set(metadataBoundResources.map((resource) => resource.uri));
    if (!activeUris.size) {
      setMetadataFieldsByUri((prev) =>
        Object.keys(prev).length ? {} : prev,
      );
      setMetadataConditionByUri((prev) =>
        Object.keys(prev).length ? {} : prev,
      );
      setMetadataCountByUri((prev) =>
        Object.keys(prev).length ? {} : prev,
      );
      setMetadataFieldValuesByUri((prev) =>
        Object.keys(prev).length ? {} : prev,
      );
      return;
    }

    for (const resource of metadataBoundResources) {
      const uri = resource.uri;
      if (metadataFieldsByUri[uri] !== undefined) {
        continue;
      }
      void queryRAGMetadataFields(uri)
        .then((fields) => {
          setMetadataFieldsByUri((prev) => ({
            ...prev,
            [uri]: fields,
          }));
          setMetadataConditionByUri((prev) => ({
            ...prev,
            [uri]: sanitizeMetadataCondition(prev[uri] ?? null, fields),
          }));
        })
        .catch(() => {
          setMetadataFieldsByUri((prev) => ({
            ...prev,
            [uri]: [],
          }));
          setMetadataConditionByUri((prev) => ({
            ...prev,
            [uri]: null,
          }));
        });
    }

    setMetadataFieldsByUri((prev) => {
      const next: Record<string, RAGMetadataField[]> = {};
      for (const uri of Object.keys(prev)) {
        const fields = prev[uri];
        if (activeUris.has(uri)) {
          next[uri] = fields ?? [];
        }
      }
      if (
        Object.keys(next).length === Object.keys(prev).length &&
        Object.keys(prev).every((key) => next[key] === prev[key])
      ) {
        return prev;
      }
      return next;
    });
    setMetadataConditionByUri((prev) => {
      const next: Record<string, MetadataCondition | null> = {};
      for (const uri of Object.keys(prev)) {
        const condition = prev[uri];
        if (activeUris.has(uri)) {
          next[uri] = condition ?? null;
        }
      }
      if (
        Object.keys(next).length === Object.keys(prev).length &&
        Object.keys(prev).every((key) => next[key] === prev[key])
      ) {
        return prev;
      }
      return next;
    });
    setMetadataCountByUri((prev) => {
      const next: Record<string, number | null> = {};
      for (const uri of Object.keys(prev)) {
        const count = prev[uri];
        if (activeUris.has(uri)) {
          next[uri] = count ?? null;
        }
      }
      if (
        Object.keys(next).length === Object.keys(prev).length &&
        Object.keys(prev).every((key) => next[key] === prev[key])
      ) {
        return prev;
      }
      return next;
    });
    setMetadataFieldValuesByUri((prev) => {
      const next: Record<string, Record<string, string[]>> = {};
      for (const uri of Object.keys(prev)) {
        const values = prev[uri];
        if (activeUris.has(uri)) {
          next[uri] = values ?? {};
        }
      }
      if (
        Object.keys(next).length === Object.keys(prev).length &&
        Object.keys(prev).every((key) => next[key] === prev[key])
      ) {
        return prev;
      }
      return next;
    });
  }, [metadataBoundResources, metadataFieldsByUri]);
  const { thread, isMock } = useThread();
  const promptRootRef = useRef<HTMLDivElement | null>(null);

  const [followups, setFollowups] = useState<string[]>([]);
  const [followupsHidden, setFollowupsHidden] = useState(false);
  const [followupsLoading, setFollowupsLoading] = useState(false);
  const lastGeneratedForAiIdRef = useRef<string | null>(null);
  const wasStreamingRef = useRef(false);

  const [confirmOpen, setConfirmOpen] = useState(false);
  const [pendingSuggestion, setPendingSuggestion] = useState<string | null>(
    null,
  );

  useEffect(() => {
    if (models.length === 0) {
      return;
    }
    const currentModel = models.find((m) => m.name === context.model_name);
    const fallbackModel = currentModel ?? models[0]!;
    const supportsThinking = fallbackModel.supports_thinking ?? false;
    const nextModelName = fallbackModel.name;
    const nextMode = getResolvedMode(context.mode, supportsThinking);

    if (context.model_name === nextModelName && context.mode === nextMode) {
      return;
    }

    onContextChange?.({
      ...context,
      model_name: nextModelName,
      mode: nextMode,
    });
  }, [context, models, onContextChange]);

  const selectedModel = useMemo(() => {
    if (models.length === 0) {
      return undefined;
    }
    return models.find((m) => m.name === context.model_name) ?? models[0];
  }, [context.model_name, models]);

  const supportThinking = useMemo(
    () => selectedModel?.supports_thinking ?? false,
    [selectedModel],
  );

  const supportReasoningEffort = useMemo(
    () => selectedModel?.supports_reasoning_effort ?? false,
    [selectedModel],
  );

  const handleModelSelect = useCallback(
    (model_name: string) => {
      const model = models.find((m) => m.name === model_name);
      if (!model) {
        return;
      }
      onContextChange?.({
        ...context,
        model_name,
        mode: getResolvedMode(context.mode, model.supports_thinking ?? false),
        reasoning_effort: context.reasoning_effort,
      });
      setModelDialogOpen(false);
    },
    [onContextChange, context, models],
  );

  const handleModeSelect = useCallback(
    (mode: InputMode) => {
      onContextChange?.({
        ...context,
        mode: getResolvedMode(mode, supportThinking),
        reasoning_effort: mode === "ultra" ? "high" : mode === "pro" ? "medium" : mode === "thinking" ? "low" : "minimal",
      });
    },
    [onContextChange, context, supportThinking],
  );

  const handleReasoningEffortSelect = useCallback(
    (effort: "minimal" | "low" | "medium" | "high") => {
      onContextChange?.({
        ...context,
        reasoning_effort: effort,
      });
    },
    [onContextChange, context],
  );

  const handleSubmit = useCallback(
    (message: PromptInputMessage) => {
      if (status === "submitted" || status === "streaming") {
        onStop?.();
        return;
      }

      if (!message.text || !message.text.trim()) {
        return;
      }

      const textWithLinks = expandMentionTokensToLinks(
        message.text,
        selectedRagResources,
      );

      const metadataByDataset: Record<string, MetadataCondition> = {};
      for (const resource of selectedRagResources) {
        const condition = metadataConditionByUri[resource.uri];
        if (condition?.conditions?.length) {
          metadataByDataset[resource.uri] = condition;
        }
      }

      let metadataPayload: MetadataConditionPayload | undefined;
      const values = Object.values(metadataByDataset);
      if (values.length === 1) {
        metadataPayload = values[0];
      } else if (values.length > 1) {
        metadataPayload = { per_dataset: metadataByDataset };
      }

      onSubmit?.({
        ...message,
        text: textWithLinks,
        ragResources: selectedRagResources,
        metadataCondition: metadataPayload,
      });

      setSelectedRagResources([]);
      setRagPickerOpen(false);
      setMetadataFieldsByUri({});
      setMetadataConditionByUri({});
      setMetadataCountByUri({});
      setMetadataFieldValuesByUri({});
      setFollowups([]);
      setFollowupsHidden(false);
      setFollowupsLoading(false);
    },
    [
      metadataConditionByUri,
      onSubmit,
      onStop,
      selectedRagResources,
      status,
    ],
  );

  const handleInsertRagMention = useCallback(
    (resource: RAGResource) => {
      const mentionEntry = getMentionToken(resource);
      const replaced = textInput.value.replace(
        /(?:^|\s)@[\w.-]*$/,
        (match) => {
          const leadingWhitespace = match.startsWith(" ") ? " " : "";
          return `${leadingWhitespace}${mentionEntry} `;
        },
      );

      textInput.setInput(replaced);
      setSelectedRagResources((prev) => {
        if (prev.some((item) => item.uri === resource.uri)) {
          return prev;
        }
        return [...prev, resource];
      });
      setRagPickerOpen(false);
    },
    [textInput],
  );

  const metadataDatasets = useMemo(
    () =>
      selectedRagResources.map((resource) => ({
        uri: resource.uri,
        datasetId: getDatasetId(resource.uri),
        title: resource.title,
        fields: metadataFieldsByUri[resource.uri] ?? [],
        value: metadataConditionByUri[resource.uri] ?? null,
      })),
    [selectedRagResources, metadataFieldsByUri, metadataConditionByUri],
  );

  const ragDialogMetadataDatasets = useMemo(
    () =>
      ragDialogSelectedResources.map((resource) => ({
        uri: resource.uri,
        datasetId: getDatasetId(resource.uri),
        title: resource.title,
        fields: metadataFieldsByUri[resource.uri] ?? [],
        value: ragDialogMetadataConditionByUri[resource.uri] ?? null,
      })),
    [ragDialogSelectedResources, metadataFieldsByUri, ragDialogMetadataConditionByUri],
  );

  const handlePreviewMetadataCount = useCallback(
    async (datasetUri: string, metadataCondition: MetadataCondition | null) => {
      const count = await queryRAGMetadataMatchCount(datasetUri, metadataCondition);
      setMetadataCountByUri((prev) => {
        if (prev[datasetUri] === count) {
          return prev;
        }
        return {
          ...prev,
          [datasetUri]: count,
        };
      });
      return count;
    },
    [],
  );

  const handleApplyMetadataByDataset = useCallback(
    (valueByDataset: Record<string, MetadataCondition | null>) => {
      setMetadataConditionByUri((prev) => {
        let changed = false;
        const next = { ...prev };
        for (const [uri, value] of Object.entries(valueByDataset)) {
          if (!isMetadataConditionEqual(next[uri] ?? null, value)) {
            next[uri] = value;
            changed = true;
          }
        }
        return changed ? next : prev;
      });
    },
    [],
  );

  const handleApplyDialogMetadataByDataset = useCallback(
    (valueByDataset: Record<string, MetadataCondition | null>) => {
      setRagDialogMetadataConditionByUri((prev) => {
        let changed = false;
        const next = { ...prev };
        for (const [uri, value] of Object.entries(valueByDataset)) {
          if (!isMetadataConditionEqual(next[uri] ?? null, value)) {
            next[uri] = value;
            changed = true;
          }
        }
        return changed ? next : prev;
      });
    },
    [],
  );

  const handleLoadMetadataFieldValues = useCallback(
    async (datasetUri: string, fieldName: string) => {
      const existing = metadataFieldValuesByUri[datasetUri]?.[fieldName];
      if (existing !== undefined) {
        return;
      }

      const values = await queryRAGMetadataFieldValues(datasetUri, fieldName, 100);
      setMetadataFieldValuesByUri((prev) => ({
        ...prev,
        [datasetUri]: {
          ...(prev[datasetUri] ?? {}),
          [fieldName]: values,
        },
      }));
    },
    [metadataFieldValuesByUri],
  );

  const handleOpenRagDialog = useCallback(() => {
    setRagDialogSelectedResources(selectedRagResources);
    setRagDialogMetadataConditionByUri(metadataConditionByUri);
    setRagDialogQuery("");
    setRagDialogTab("select");
    setRagDialogOpen(true);
  }, [metadataConditionByUri, selectedRagResources]);

  const handleToggleDialogRagResource = useCallback((resource: RAGResource) => {
    setRagDialogSelectedResources((prev) => {
      const exists = prev.some((item) => item.uri === resource.uri);
      if (exists) {
        return prev.filter((item) => item.uri !== resource.uri);
      }
      return [...prev, resource];
    });
  }, []);

  const handleRemoveSelectedRagResource = useCallback((uri: string) => {
    setSelectedRagResources((prev) => prev.filter((resource) => resource.uri !== uri));
    setMetadataConditionByUri((prev) => {
      if (!(uri in prev)) {
        return prev;
      }
      const next = { ...prev };
      delete next[uri];
      return next;
    });
  }, []);

  const handleApplyRagSelection = useCallback(() => {
    setSelectedRagResources(ragDialogSelectedResources);
    setMetadataConditionByUri(() => {
      const next: Record<string, MetadataCondition | null> = {};
      for (const resource of ragDialogSelectedResources) {
        next[resource.uri] = ragDialogMetadataConditionByUri[resource.uri] ?? null;
      }
      return next;
    });
    setRagDialogOpen(false);
  }, [ragDialogMetadataConditionByUri, ragDialogSelectedResources]);

  const requestFormSubmit = useCallback(() => {
    const form = promptRootRef.current?.querySelector("form");
    form?.requestSubmit();
  }, []);

  const handleFollowupClick = useCallback(
    (suggestion: string) => {
      if (status === "streaming") {
        return;
      }
      const current = (textInput.value ?? "").trim();
      if (current) {
        setPendingSuggestion(suggestion);
        setConfirmOpen(true);
        return;
      }
      textInput.setInput(suggestion);
      setFollowupsHidden(true);
      setTimeout(() => requestFormSubmit(), 0);
    },
    [requestFormSubmit, status, textInput],
  );

  const confirmReplaceAndSend = useCallback(() => {
    if (!pendingSuggestion) {
      setConfirmOpen(false);
      return;
    }
    textInput.setInput(pendingSuggestion);
    setFollowupsHidden(true);
    setConfirmOpen(false);
    setPendingSuggestion(null);
    setTimeout(() => requestFormSubmit(), 0);
  }, [pendingSuggestion, requestFormSubmit, textInput]);

  const confirmAppendAndSend = useCallback(() => {
    if (!pendingSuggestion) {
      setConfirmOpen(false);
      return;
    }
    const current = (textInput.value ?? "").trim();
    const next = current ? `${current}\n${pendingSuggestion}` : pendingSuggestion;
    textInput.setInput(next);
    setFollowupsHidden(true);
    setConfirmOpen(false);
    setPendingSuggestion(null);
    setTimeout(() => requestFormSubmit(), 0);
  }, [pendingSuggestion, requestFormSubmit, textInput]);

  useEffect(() => {
    const streaming = status === "streaming";
    const wasStreaming = wasStreamingRef.current;
    wasStreamingRef.current = streaming;
    if (!wasStreaming || streaming) {
      return;
    }

    if (disabled || isMock) {
      return;
    }

    const lastAi = [...thread.messages].reverse().find((m) => m.type === "ai");
    const lastAiId = lastAi?.id ?? null;
    if (!lastAiId || lastAiId === lastGeneratedForAiIdRef.current) {
      return;
    }
    lastGeneratedForAiIdRef.current = lastAiId;

    const recent = thread.messages
      .filter((m) => m.type === "human" || m.type === "ai")
      .map((m) => {
        const role = m.type === "human" ? "user" : "assistant";
        const content = textOfMessage(m) ?? "";
        return { role, content };
      })
      .filter((m) => m.content.trim().length > 0)
      .slice(-6);

    if (recent.length === 0) {
      return;
    }

    const controller = new AbortController();
    setFollowupsHidden(false);
    setFollowupsLoading(true);
    setFollowups([]);

    fetch(`${getBackendBaseURL()}/api/threads/${threadId}/suggestions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: recent,
        n: 3,
        model_name: context.model_name ?? undefined,
      }),
      signal: controller.signal,
    })
      .then(async (res) => {
        if (!res.ok) {
          return { suggestions: [] as string[] };
        }
        return (await res.json()) as { suggestions?: string[] };
      })
      .then((data) => {
        const suggestions = (data.suggestions ?? [])
          .map((s) => (typeof s === "string" ? s.trim() : ""))
          .filter((s) => s.length > 0)
          .slice(0, 5);
        setFollowups(suggestions);
      })
      .catch(() => {
        setFollowups([]);
      })
      .finally(() => {
        setFollowupsLoading(false);
      });

    return () => controller.abort();
  }, [context.model_name, disabled, isMock, status, thread.messages, threadId]);

  return (
    <div ref={promptRootRef} className="relative w-full">
      {ragPickerOpen && (
        <div className="bg-popover border-border absolute right-0 bottom-full left-0 z-30 mb-2 max-h-52 overflow-auto rounded-md border shadow-md">
          {ragLoading ? (
            <div className="text-muted-foreground px-3 py-2 text-sm">
              {t.common.loading}
            </div>
          ) : ragSuggestions.length > 0 ? (
            ragSuggestions.map((resource) => (
              <button
                key={resource.uri}
                type="button"
                className="hover:bg-accent hover:text-accent-foreground w-full px-3 py-2 text-left text-sm"
                onClick={() => handleInsertRagMention(resource)}
              >
                <div className="font-medium">{resource.title}</div>
                {!!resource.description && (
                  <div className="text-muted-foreground line-clamp-1 text-xs">
                    {resource.description}
                  </div>
                )}
              </button>
            ))
          ) : (
            <div className="text-muted-foreground px-3 py-2 text-sm">
              No matching resources
            </div>
          )}
        </div>
      )}
      <PromptInput
        className={cn(
          "bg-background/85 rounded-2xl backdrop-blur-sm transition-all duration-300 ease-out *:data-[slot='input-group']:rounded-2xl",
          className,
        )}
        disabled={disabled}
        globalDrop
        multiple
        onSubmit={handleSubmit}
        {...props}
      >
        {extraHeader && (
          <div className="absolute top-0 right-0 left-0 z-10">
            <div className="absolute right-0 bottom-0 left-0 flex items-center justify-center">
              {extraHeader}
            </div>
          </div>
        )}
        <PromptInputAttachments>
          {(attachment) => <PromptInputAttachment data={attachment} />}
        </PromptInputAttachments>
        {selectedRagResources.length > 0 && (
          <div
            className={cn(
              "flex w-full flex-wrap items-center gap-2 p-3",
              attachments.files.length > 0 && "pt-0",
            )}
          >
            {selectedRagResources.map((resource) => {
              const hasMetadata = Boolean(
                metadataConditionByUri[resource.uri]?.conditions?.length,
              );
              return (
                <div key={resource.uri} className="max-w-60">
                  <Badge
                    variant="outline"
                    className="bg-background/70 gap-1.5 rounded-full px-2.5 py-1 text-xs"
                  >
                    <DatabaseIcon className="size-3.5" />
                    <span className="max-w-44 truncate">{resource.title}</span>
                    {hasMetadata && (
                      <span className="text-muted-foreground">• meta</span>
                    )}
                    <button
                      type="button"
                      className="text-muted-foreground hover:text-foreground"
                      onClick={() => handleRemoveSelectedRagResource(resource.uri)}
                    >
                      <XIcon className="size-3" />
                    </button>
                  </Badge>
                </div>
              );
            })}
          </div>
        )}
        <PromptInputBody className="absolute top-0 right-0 left-0 z-3">
          <PromptInputTextarea
            className={cn("size-full")}
            disabled={disabled}
            placeholder={
              isRagEnabled
                ? t.inputBox.placeholderWithRag
                : t.inputBox.placeholder
            }
            autoFocus={autoFocus}
            defaultValue={initialValue}
          />
        </PromptInputBody>
        {isRagEnabled && (
          <div className="text-muted-foreground px-4 pb-1 text-xs">
            {t.inputBox.ragHint}
          </div>
        )}
        <PromptInputFooter className="flex">
          <PromptInputTools>
          {/* TODO: Add more connectors here
          <PromptInputActionMenu>
            <PromptInputActionMenuTrigger className="px-2!" />
            <PromptInputActionMenuContent>
              <PromptInputActionAddAttachments
                label={t.inputBox.addAttachments}
              />
            </PromptInputActionMenuContent>
          </PromptInputActionMenu> */}
          <AddAttachmentsButton
            className="px-2!"
            ragEnabled={isRagEnabled}
            onOpenRag={handleOpenRagDialog}
          />
          <PromptInputActionMenu>
            <ModeHoverGuide
              mode={
                context.mode === "flash" ||
                  context.mode === "thinking" ||
                  context.mode === "pro" ||
                  context.mode === "ultra"
                  ? context.mode
                  : "flash"
              }
            >
              <PromptInputActionMenuTrigger className="gap-1! px-2!">
                <div>
                  {context.mode === "flash" && <ZapIcon className="size-3" />}
                  {context.mode === "thinking" && (
                    <LightbulbIcon className="size-3" />
                  )}
                  {context.mode === "pro" && (
                    <GraduationCapIcon className="size-3" />
                  )}
                  {context.mode === "ultra" && (
                    <RocketIcon className="size-3 text-[#dabb5e]" />
                  )}
                </div>
                <div
                  className={cn(
                    "text-xs font-normal",
                    context.mode === "ultra" ? "golden-text" : "",
                  )}
                >
                  {(context.mode === "flash" && t.inputBox.flashMode) ||
                    (context.mode === "thinking" && t.inputBox.reasoningMode) ||
                    (context.mode === "pro" && t.inputBox.proMode) ||
                    (context.mode === "ultra" && t.inputBox.ultraMode)}
                </div>
              </PromptInputActionMenuTrigger>
            </ModeHoverGuide>
            <PromptInputActionMenuContent className="w-80">
              <DropdownMenuGroup>
                <DropdownMenuLabel className="text-muted-foreground text-xs">
                  {t.inputBox.mode}
                </DropdownMenuLabel>
                <PromptInputActionMenu>
                  <PromptInputActionMenuItem
                    className={cn(
                      context.mode === "flash"
                        ? "text-accent-foreground"
                        : "text-muted-foreground/65",
                    )}
                    onSelect={() => handleModeSelect("flash")}
                  >
                    <div className="flex flex-col gap-2">
                      <div className="flex items-center gap-1 font-bold">
                        <ZapIcon
                          className={cn(
                            "mr-2 size-4",
                            context.mode === "flash" &&
                            "text-accent-foreground",
                          )}
                        />
                        {t.inputBox.flashMode}
                      </div>
                      <div className="pl-7 text-xs">
                        {t.inputBox.flashModeDescription}
                      </div>
                    </div>
                    {context.mode === "flash" ? (
                      <CheckIcon className="ml-auto size-4" />
                    ) : (
                      <div className="ml-auto size-4" />
                    )}
                  </PromptInputActionMenuItem>
                  {supportThinking && (
                    <PromptInputActionMenuItem
                      className={cn(
                        context.mode === "thinking"
                          ? "text-accent-foreground"
                          : "text-muted-foreground/65",
                      )}
                      onSelect={() => handleModeSelect("thinking")}
                    >
                      <div className="flex flex-col gap-2">
                        <div className="flex items-center gap-1 font-bold">
                          <LightbulbIcon
                            className={cn(
                              "mr-2 size-4",
                              context.mode === "thinking" &&
                              "text-accent-foreground",
                            )}
                          />
                          {t.inputBox.reasoningMode}
                        </div>
                        <div className="pl-7 text-xs">
                          {t.inputBox.reasoningModeDescription}
                        </div>
                      </div>
                      {context.mode === "thinking" ? (
                        <CheckIcon className="ml-auto size-4" />
                      ) : (
                        <div className="ml-auto size-4" />
                      )}
                    </PromptInputActionMenuItem>
                  )}
                  <PromptInputActionMenuItem
                    className={cn(
                      context.mode === "pro"
                        ? "text-accent-foreground"
                        : "text-muted-foreground/65",
                    )}
                    onSelect={() => handleModeSelect("pro")}
                  >
                    <div className="flex flex-col gap-2">
                      <div className="flex items-center gap-1 font-bold">
                        <GraduationCapIcon
                          className={cn(
                            "mr-2 size-4",
                            context.mode === "pro" && "text-accent-foreground",
                          )}
                        />
                        {t.inputBox.proMode}
                      </div>
                      <div className="pl-7 text-xs">
                        {t.inputBox.proModeDescription}
                      </div>
                    </div>
                    {context.mode === "pro" ? (
                      <CheckIcon className="ml-auto size-4" />
                    ) : (
                      <div className="ml-auto size-4" />
                    )}
                  </PromptInputActionMenuItem>
                  <PromptInputActionMenuItem
                    className={cn(
                      context.mode === "ultra"
                        ? "text-accent-foreground"
                        : "text-muted-foreground/65",
                    )}
                    onSelect={() => handleModeSelect("ultra")}
                  >
                    <div className="flex flex-col gap-2">
                      <div className="flex items-center gap-1 font-bold">
                        <RocketIcon
                          className={cn(
                            "mr-2 size-4",
                            context.mode === "ultra" && "text-[#dabb5e]",
                          )}
                        />
                        <div
                          className={cn(
                            context.mode === "ultra" && "golden-text",
                          )}
                        >
                          {t.inputBox.ultraMode}
                        </div>
                      </div>
                      <div className="pl-7 text-xs">
                        {t.inputBox.ultraModeDescription}
                      </div>
                    </div>
                    {context.mode === "ultra" ? (
                      <CheckIcon className="ml-auto size-4" />
                    ) : (
                      <div className="ml-auto size-4" />
                    )}
                  </PromptInputActionMenuItem>
                </PromptInputActionMenu>
              </DropdownMenuGroup>
            </PromptInputActionMenuContent>
          </PromptInputActionMenu>
          {selectedRagResources.length > 0 && (
            <MetadataFilterDialog
              datasets={metadataDatasets}
              counts={metadataCountByUri}
              fieldValuesByDataset={metadataFieldValuesByUri}
              onLoadFieldValues={handleLoadMetadataFieldValues}
              onPreviewCount={handlePreviewMetadataCount}
              onApply={handleApplyMetadataByDataset}
            />
          )}
          {supportReasoningEffort && context.mode !== "flash" && (
            <PromptInputActionMenu>
              <PromptInputActionMenuTrigger className="gap-1! px-2!">
                <div className="text-xs font-normal">
                  {t.inputBox.reasoningEffort}:
                  {context.reasoning_effort === "minimal" && " " + t.inputBox.reasoningEffortMinimal}
                  {context.reasoning_effort === "low" && " " + t.inputBox.reasoningEffortLow}
                  {context.reasoning_effort === "medium" && " " + t.inputBox.reasoningEffortMedium}
                  {context.reasoning_effort === "high" && " " + t.inputBox.reasoningEffortHigh}
                </div>
              </PromptInputActionMenuTrigger>
              <PromptInputActionMenuContent className="w-70">
                <DropdownMenuGroup>
                  <DropdownMenuLabel className="text-muted-foreground text-xs">
                    {t.inputBox.reasoningEffort}
                  </DropdownMenuLabel>
                  <PromptInputActionMenu>
                    <PromptInputActionMenuItem
                      className={cn(
                        context.reasoning_effort === "minimal"
                          ? "text-accent-foreground"
                          : "text-muted-foreground/65",
                      )}
                      onSelect={() => handleReasoningEffortSelect("minimal")}
                    >
                      <div className="flex flex-col gap-2">
                        <div className="flex items-center gap-1 font-bold">
                          {t.inputBox.reasoningEffortMinimal}
                        </div>
                        <div className="pl-2 text-xs">
                          {t.inputBox.reasoningEffortMinimalDescription}
                        </div>
                      </div>
                      {context.reasoning_effort === "minimal" ? (
                        <CheckIcon className="ml-auto size-4" />
                      ) : (
                        <div className="ml-auto size-4" />
                      )}
                    </PromptInputActionMenuItem>
                    <PromptInputActionMenuItem
                      className={cn(
                        context.reasoning_effort === "low"
                          ? "text-accent-foreground"
                          : "text-muted-foreground/65",
                      )}
                      onSelect={() => handleReasoningEffortSelect("low")}
                    >
                      <div className="flex flex-col gap-2">
                        <div className="flex items-center gap-1 font-bold">
                          {t.inputBox.reasoningEffortLow}
                        </div>
                        <div className="pl-2 text-xs">
                          {t.inputBox.reasoningEffortLowDescription}
                        </div>
                      </div>
                      {context.reasoning_effort === "low" ? (
                        <CheckIcon className="ml-auto size-4" />
                      ) : (
                        <div className="ml-auto size-4" />
                      )}
                    </PromptInputActionMenuItem>
                    <PromptInputActionMenuItem
                      className={cn(
                        context.reasoning_effort === "medium" || !context.reasoning_effort
                          ? "text-accent-foreground"
                          : "text-muted-foreground/65",
                      )}
                      onSelect={() => handleReasoningEffortSelect("medium")}
                    >
                      <div className="flex flex-col gap-2">
                        <div className="flex items-center gap-1 font-bold">
                          {t.inputBox.reasoningEffortMedium}
                        </div>
                        <div className="pl-2 text-xs">
                          {t.inputBox.reasoningEffortMediumDescription}
                        </div>
                      </div>
                      {context.reasoning_effort === "medium" || !context.reasoning_effort ? (
                        <CheckIcon className="ml-auto size-4" />
                      ) : (
                        <div className="ml-auto size-4" />
                      )}
                    </PromptInputActionMenuItem>
                    <PromptInputActionMenuItem
                      className={cn(
                        context.reasoning_effort === "high"
                          ? "text-accent-foreground"
                          : "text-muted-foreground/65",
                      )}
                      onSelect={() => handleReasoningEffortSelect("high")}
                    >
                      <div className="flex flex-col gap-2">
                        <div className="flex items-center gap-1 font-bold">
                          {t.inputBox.reasoningEffortHigh}
                        </div>
                        <div className="pl-2 text-xs">
                          {t.inputBox.reasoningEffortHighDescription}
                        </div>
                      </div>
                      {context.reasoning_effort === "high" ? (
                        <CheckIcon className="ml-auto size-4" />
                      ) : (
                        <div className="ml-auto size-4" />
                      )}
                    </PromptInputActionMenuItem>
                  </PromptInputActionMenu>
                </DropdownMenuGroup>
              </PromptInputActionMenuContent>
            </PromptInputActionMenu>
          )}
        </PromptInputTools>
        <PromptInputTools>
          <ModelSelector
            open={modelDialogOpen}
            onOpenChange={setModelDialogOpen}
          >
            <ModelSelectorTrigger asChild>
              <PromptInputButton>
                <div className="flex min-w-0 flex-col items-start text-left">
                  <ModelSelectorName className="text-xs font-normal">
                    {selectedModel?.display_name}
                  </ModelSelectorName>
                  {selectedModel?.model && (
                    <span className="text-muted-foreground w-full truncate text-[10px] leading-none">
                      {selectedModel.model}
                    </span>
                  )}
                </div>
              </PromptInputButton>
            </ModelSelectorTrigger>
            <ModelSelectorContent>
              <ModelSelectorInput placeholder={t.inputBox.searchModels} />
              <ModelSelectorList>
                {models.map((m) => (
                  <ModelSelectorItem
                    key={m.name}
                    value={m.name}
                    onSelect={() => handleModelSelect(m.name)}
                  >
                    <div className="flex min-w-0 flex-1 flex-col">
                      <ModelSelectorName>{m.display_name}</ModelSelectorName>
                      <span className="text-muted-foreground truncate text-[10px]">
                        {m.model}
                      </span>
                    </div>
                    {m.name === context.model_name ? (
                      <CheckIcon className="ml-auto size-4" />
                    ) : (
                      <div className="ml-auto size-4" />
                    )}
                  </ModelSelectorItem>
                ))}
              </ModelSelectorList>
            </ModelSelectorContent>
          </ModelSelector>
          <PromptInputSubmit
            className="rounded-full"
            disabled={disabled}
            variant="outline"
            status={status}
          />
        </PromptInputTools>
      </PromptInputFooter>
      {isNewThread && searchParams.get("mode") !== "skill" && (
        <div className="absolute right-0 -bottom-20 left-0 z-0 flex items-center justify-center">
          <SuggestionList />
        </div>
      )}
      {!isNewThread && (
        <div className="bg-background absolute right-0 -bottom-[17px] left-0 z-0 h-4"></div>
      )}
      </PromptInput>

      <Dialog open={ragDialogOpen} onOpenChange={setRagDialogOpen}>
        <DialogContent className="sm:max-w-[760px]">
          <DialogHeader>
            <DialogTitle>{t.inputBox.ragDialogTitle}</DialogTitle>
            <DialogDescription>
              {t.inputBox.ragDialogDescription}
            </DialogDescription>
          </DialogHeader>

          <Tabs value={ragDialogTab} onValueChange={(value) => setRagDialogTab(value as "select" | "metadata")}>
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="select">{t.inputBox.ragSelectTab}</TabsTrigger>
              <TabsTrigger value="metadata" disabled={ragDialogSelectedResources.length === 0}>
                {t.inputBox.ragMetadataTab}
              </TabsTrigger>
            </TabsList>

            <TabsContent value="select" className="mt-4 space-y-4">
              <div className="bg-muted/40 flex items-center justify-between rounded-xl border px-4 py-3">
                <div>
                  <div className="text-sm font-medium">{t.inputBox.ragSelectedDatasets}</div>
                  <div className="text-muted-foreground text-xs">
                    {t.inputBox.ragDialogDescription}
                  </div>
                </div>
                <Badge variant="secondary" className="rounded-full px-3 py-1 text-xs">
                  {ragDialogSelectedResources.length}
                </Badge>
              </div>

              <div className="relative">
                <SearchIcon className="text-muted-foreground absolute top-1/2 left-3 size-4 -translate-y-1/2" />
                <Input
                  className="h-10 rounded-xl pl-9"
                  placeholder={t.inputBox.ragSearchPlaceholder}
                  value={ragDialogQuery}
                  onChange={(event) => setRagDialogQuery(event.target.value)}
                />
              </div>

              {ragDialogSelectedResources.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {ragDialogSelectedResources.map((resource) => (
                    <Badge
                      key={resource.uri}
                      variant="outline"
                      className="gap-1.5 rounded-full px-2.5 py-1 text-xs"
                    >
                      <DatabaseIcon className="size-3.5" />
                      <span className="max-w-44 truncate">{resource.title}</span>
                      <button
                        type="button"
                        className="text-muted-foreground hover:text-foreground"
                        onClick={() => handleToggleDialogRagResource(resource)}
                      >
                        <XIcon className="size-3" />
                      </button>
                    </Badge>
                  ))}
                </div>
              )}

              <ScrollArea className="h-[320px] rounded-xl border">
                <div className="grid gap-2 p-3">
                  {ragDialogLoading ? (
                    <div className="text-muted-foreground px-3 py-6 text-center text-sm">
                      {t.common.loading}
                    </div>
                  ) : ragDialogResources.length > 0 ? (
                    ragDialogResources.map((resource) => {
                      const selected = ragDialogSelectedResources.some(
                        (item) => item.uri === resource.uri,
                      );

                      return (
                        <button
                          key={resource.uri}
                          type="button"
                          className={cn(
                            "hover:bg-accent/70 flex w-full items-start gap-3 rounded-xl border px-4 py-3 text-left transition-colors",
                            selected &&
                              "border-primary/40 bg-primary/5 ring-primary/15 ring-1",
                          )}
                          onClick={() => handleToggleDialogRagResource(resource)}
                        >
                          <div
                            className={cn(
                              "mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-lg border",
                              selected
                                ? "border-primary/40 bg-primary/10 text-primary"
                                : "text-muted-foreground bg-background",
                            )}
                          >
                            {selected ? (
                              <CheckIcon className="size-4" />
                            ) : (
                              <DatabaseIcon className="size-4" />
                            )}
                          </div>
                          <div className="min-w-0 flex-1">
                            <div className="flex items-center justify-between gap-3">
                              <div className="truncate text-sm font-medium">
                                {resource.title}
                              </div>
                              {selected && <Badge variant="secondary">Selected</Badge>}
                            </div>
                            <div className="text-muted-foreground mt-1 line-clamp-2 text-xs leading-5">
                              {resource.description || resource.uri}
                            </div>
                          </div>
                        </button>
                      );
                    })
                  ) : (
                    <div className="text-muted-foreground px-3 py-6 text-center text-sm">
                      {t.inputBox.ragSearchEmpty}
                    </div>
                  )}
                </div>
              </ScrollArea>
            </TabsContent>

            <TabsContent value="metadata" className="mt-4">
              {ragDialogMetadataDatasets.length > 0 ? (
                <MetadataFilterEditor
                  datasets={ragDialogMetadataDatasets}
                  counts={metadataCountByUri}
                  fieldValuesByDataset={metadataFieldValuesByUri}
                  onLoadFieldValues={handleLoadMetadataFieldValues}
                  onPreviewCount={handlePreviewMetadataCount}
                  onApply={handleApplyDialogMetadataByDataset}
                />
              ) : (
                <div className="text-muted-foreground flex min-h-56 items-center justify-center rounded-xl border border-dashed text-sm">
                  {t.inputBox.ragNoMetadataDescription}
                </div>
              )}
            </TabsContent>
          </Tabs>

          <DialogFooter className="items-center justify-between sm:justify-between">
            <div className="text-muted-foreground flex items-center gap-2 text-xs">
              <Badge variant="outline">{ragDialogSelectedResources.length}</Badge>
              <span>{t.inputBox.ragSelectedDatasets}</span>
            </div>
            <div className="flex items-center gap-2">
              {ragDialogTab === "select" && ragDialogSelectedResources.length > 0 && (
                <Button
                  variant="outline"
                  onClick={() => setRagDialogTab("metadata")}
                >
                  {t.inputBox.ragConfigureMetadata}
                </Button>
              )}
              <Button variant="outline" onClick={() => setRagDialogOpen(false)}>
                {t.common.cancel}
              </Button>
              <Button onClick={handleApplyRagSelection}>
                {t.inputBox.ragUseSelection}
              </Button>
            </div>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {!disabled &&
        !isNewThread &&
        !followupsHidden &&
        (followupsLoading || followups.length > 0) && (
          <div className="absolute right-0 -top-20 left-0 z-20 flex items-center justify-center">
            <div className="flex items-center gap-2">
              {followupsLoading ? (
                <div className="text-muted-foreground bg-background/80 rounded-full border px-4 py-2 text-xs backdrop-blur-sm">
                  {t.inputBox.followupLoading}
                </div>
              ) : (
                <Suggestions className="min-h-16 w-fit items-start">
                  {followups.map((s) => (
                    <Suggestion
                      key={s}
                      suggestion={s}
                      onClick={() => handleFollowupClick(s)}
                    />
                  ))}
                  <Button
                    aria-label={t.common.close}
                    className="text-muted-foreground cursor-pointer rounded-full px-3 text-xs font-normal"
                    variant="outline"
                    size="sm"
                    type="button"
                    onClick={() => setFollowupsHidden(true)}
                  >
                    <XIcon className="size-4" />
                  </Button>
                </Suggestions>
              )}
            </div>
          </div>
        )}

      <Dialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t.inputBox.followupConfirmTitle}</DialogTitle>
            <DialogDescription>
              {t.inputBox.followupConfirmDescription}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirmOpen(false)}>
              {t.common.cancel}
            </Button>
            <Button variant="secondary" onClick={confirmAppendAndSend}>
              {t.inputBox.followupConfirmAppend}
            </Button>
            <Button onClick={confirmReplaceAndSend}>
              {t.inputBox.followupConfirmReplace}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function sanitizeMetadataCondition(
  value: MetadataCondition | null,
  fields: RAGMetadataField[],
): MetadataCondition | null {
  if (!value?.conditions?.length) {
    return null;
  }
  const allowed = new Set(fields.map((field) => field.name));
  const conditions = value.conditions.filter((item) => allowed.has(item.name));
  if (!conditions.length) {
    return null;
  }
  return {
    logic: value.logic,
    conditions,
  };
}

function isMetadataConditionEqual(
  a: MetadataCondition | null,
  b: MetadataCondition | null,
): boolean {
  if (a === b) {
    return true;
  }
  if (!a || !b) {
    return false;
  }
  if (a.logic !== b.logic) {
    return false;
  }
  if (a.conditions.length !== b.conditions.length) {
    return false;
  }

  for (let index = 0; index < a.conditions.length; index += 1) {
    const left = a.conditions[index];
    const right = b.conditions[index];
    if (!left || !right) {
      return false;
    }
    if (
      left.name !== right.name ||
      left.comparison_operator !== right.comparison_operator ||
      left.value !== right.value
    ) {
      return false;
    }
  }

  return true;
}

function getDatasetId(uri: string): string {
  const match = uri.match(/^rag:\/\/dataset\/([^#/?]+)/);
  return match?.[1] ?? "";
}

function getDatasetUrl(uri: string): string {
  const datasetId = getDatasetId(uri);
  return `http://47.100.52.201:18080/dataset/dataset/${datasetId}`;
}

function getMentionToken(resource: RAGResource): string {
  const safeLabel = (resource.title || "dataset").replace(/\}/g, "");
  return `@{${safeLabel}}`;
}

function escapeRegExp(input: string): string {
  return input.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function expandMentionTokensToLinks(
  text: string,
  resources: RAGResource[],
): string {
  let output = text;
  for (const resource of resources) {
    const token = getMentionToken(resource);
    const link = `[${token}](${getDatasetUrl(resource.uri)})`;
    output = output.replace(new RegExp(escapeRegExp(token), "g"), link);
  }
  return output;
}

function SuggestionList() {
  const { t } = useI18n();
  const { textInput } = usePromptInputController();
  const handleSuggestionClick = useCallback(
    (prompt: string | undefined) => {
      if (!prompt) return;
      textInput.setInput(prompt);
      setTimeout(() => {
        const textarea = document.querySelector<HTMLTextAreaElement>(
          "textarea[name='message']",
        );
        if (textarea) {
          const selStart = prompt.indexOf("[");
          const selEnd = prompt.indexOf("]");
          if (selStart !== -1 && selEnd !== -1) {
            textarea.setSelectionRange(selStart, selEnd + 1);
            textarea.focus();
          }
        }
      }, 500);
    },
    [textInput],
  );
  return (
    <Suggestions className="min-h-16 w-fit items-start">
      <ConfettiButton
        className="text-muted-foreground cursor-pointer rounded-full px-4 text-xs font-normal"
        variant="outline"
        size="sm"
        onClick={() => handleSuggestionClick(t.inputBox.surpriseMePrompt)}
      >
        <SparklesIcon className="size-4" /> {t.inputBox.surpriseMe}
      </ConfettiButton>
      {t.inputBox.suggestions.map((suggestion) => (
        <Suggestion
          key={suggestion.suggestion}
          icon={suggestion.icon}
          suggestion={suggestion.suggestion}
          onClick={() => handleSuggestionClick(suggestion.prompt)}
        />
      ))}
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Suggestion icon={PlusIcon} suggestion={t.common.create} />
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start">
          <DropdownMenuGroup>
            {t.inputBox.suggestionsCreate.map((suggestion, index) =>
              "type" in suggestion && suggestion.type === "separator" ? (
                <DropdownMenuSeparator key={index} />
              ) : (
                !("type" in suggestion) && (
                  <DropdownMenuItem
                    key={suggestion.suggestion}
                    onClick={() => handleSuggestionClick(suggestion.prompt)}
                  >
                    {suggestion.icon && <suggestion.icon className="size-4" />}
                    {suggestion.suggestion}
                  </DropdownMenuItem>
                )
              ),
            )}
          </DropdownMenuGroup>
        </DropdownMenuContent>
      </DropdownMenu>
    </Suggestions>
  );
}

function AddAttachmentsButton({
  className,
  ragEnabled,
  onOpenRag,
}: {
  className?: string;
  ragEnabled: boolean;
  onOpenRag: () => void;
}) {
  const { t } = useI18n();
  const attachments = usePromptInputAttachments();
  return (
    <DropdownMenu>
      <Tooltip content={t.inputBox.addAttachments}>
        <DropdownMenuTrigger asChild>
          <PromptInputButton className={cn("px-2!", className)}>
            <PaperclipIcon className="size-3" />
          </PromptInputButton>
        </DropdownMenuTrigger>
      </Tooltip>
      <DropdownMenuContent align="start" className="w-64 rounded-xl p-2">
        <DropdownMenuLabel className="text-xs">{t.inputBox.addAttachments}</DropdownMenuLabel>
        <DropdownMenuItem onSelect={() => attachments.openFileDialog()}>
          <UploadIcon className="mt-0.5 size-4" />
          <div className="flex min-w-0 flex-col">
            <span>{t.inputBox.uploadSourceLabel}</span>
            <span className="text-muted-foreground text-xs">
              {t.inputBox.uploadSourceDescription}
            </span>
          </div>
        </DropdownMenuItem>
        {ragEnabled && (
          <DropdownMenuItem onSelect={onOpenRag}>
            <DatabaseIcon className="mt-0.5 size-4" />
            <div className="flex min-w-0 flex-col">
              <span>{t.inputBox.ragSourceLabel}</span>
              <span className="text-muted-foreground text-xs">
                {t.inputBox.ragSourceDescription}
              </span>
            </div>
          </DropdownMenuItem>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function uniqueRagResources(resources: RAGResource[]): RAGResource[] {
  const byUri = new Map<string, RAGResource>();
  for (const resource of resources) {
    if (!resource?.uri) {
      continue;
    }
    byUri.set(resource.uri, resource);
  }
  return Array.from(byUri.values());
}
