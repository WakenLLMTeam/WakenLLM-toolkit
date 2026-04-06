"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useI18n } from "@/core/i18n/hooks";
import { useLocalSettings } from "@/core/settings";

import { SettingsSection } from "./settings-section";

export function AgentRuntimeSettingsPage() {
  const { t } = useI18n();
  const [settings, setSettings] = useLocalSettings();
  const engine = settings.context.workflow_engine ?? "langgraph";

  return (
    <div className="space-y-8">
      <SettingsSection
        title={t.settings.agentRuntime.title}
        description={t.settings.agentRuntime.description}
      >
        <div className="space-y-2">
          <div className="text-sm font-medium">
            {t.settings.agentRuntime.engineTitle}
          </div>
          <p className="text-muted-foreground text-sm">
            {t.settings.agentRuntime.engineDescription}
          </p>
          <Select
            value={engine}
            onValueChange={(v) => {
              if (v === "langgraph" || v === "report_orchestrator") {
                setSettings("context", { workflow_engine: v });
              }
            }}
          >
            <SelectTrigger className="mt-2 w-full max-w-md">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="langgraph">
                <div className="flex max-w-sm flex-col items-start gap-0.5 py-0.5">
                  <span className="font-medium">
                    {t.settings.agentRuntime.langgraphLabel}
                  </span>
                  <span className="text-muted-foreground text-xs font-normal">
                    {t.settings.agentRuntime.langgraphHint}
                  </span>
                </div>
              </SelectItem>
              <SelectItem value="report_orchestrator">
                <div className="flex max-w-sm flex-col items-start gap-0.5 py-0.5">
                  <span className="font-medium">
                    {t.settings.agentRuntime.orchestratorLabel}
                  </span>
                  <span className="text-muted-foreground text-xs font-normal">
                    {t.settings.agentRuntime.orchestratorHint}
                  </span>
                </div>
              </SelectItem>
            </SelectContent>
          </Select>
        </div>
      </SettingsSection>
    </div>
  );
}
