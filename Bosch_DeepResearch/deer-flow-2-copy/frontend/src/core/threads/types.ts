import type { Message, Thread } from "@langchain/langgraph-sdk";

import type { Todo } from "../todos";

export interface AgentThreadState extends Record<string, unknown> {
  title: string;
  messages: Message[];
  artifacts: string[];
  todos?: Todo[];
}

export interface AgentThread extends Thread<AgentThreadState> {}

export interface AgentThreadContext extends Record<string, unknown> {
  thread_id: string;
  model_name: string | undefined;
  thinking_enabled: boolean;
  is_plan_mode: boolean;
  subagent_enabled: boolean;
  rag_resources?: Array<{
    uri: string;
    title: string;
    description?: string;
  }>;
  metadata_condition?: {
    logic: "and" | "or";
    conditions: Array<{
      name: string;
      comparison_operator: string;
      value: string | number | boolean;
    }>;
  } | {
    per_dataset: Record<
      string,
      {
        logic: "and" | "or";
        conditions: Array<{
          name: string;
          comparison_operator: string;
          value: string | number | boolean;
        }>;
      }
    >;
  };
  reasoning_effort?: "minimal" | "low" | "medium" | "high";
  agent_name?: string;
}
