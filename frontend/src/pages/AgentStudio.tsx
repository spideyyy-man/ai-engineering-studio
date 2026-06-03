import { useEffect, useState } from "react";
import StubPage from "./_StubPage";
import Card from "@/components/Card";
import { api } from "@/api/client";

interface Catalog {
  agent_types: { key: string; label: string; description: string }[];
  tools: { key: string; label: string }[];
}

export default function AgentStudio() {
  const [cat, setCat] = useState<Catalog | null>(null);

  useEffect(() => {
    api.get<Catalog>("/agents/catalog").then((r) => setCat(r.data)).catch(() => setCat(null));
  }, []);

  return (
    <StubPage
      title="AI Agent Studio"
      subtitle="ReAct, tool-calling, and multi-agent supervisor patterns — with prompt templates and run traces."
      accent="text-studio-agent"
      phase="Phase 4"
      capabilities={[
        "GET /api/v1/agents/catalog — agent types and tool registry",
      ]}
      comingSoon={[
        "POST /api/v1/agents — create / configure agents per project",
        "Tool registry: Search, SQL, RAG, Calculator, HTTP",
        "Workflow builder UI + multi-agent orchestration",
        "Token + cost monitoring per run",
      ]}
      extra={
        cat && (
          <div className="mt-6 grid gap-4 md:grid-cols-2">
            <Card title="Agent types">
              <ul className="space-y-2 text-sm text-slate-200">
                {cat.agent_types.map((a) => (
                  <li key={a.key} className="rounded-lg border border-bg-border bg-bg-panel p-3">
                    <div className="font-medium">{a.label}</div>
                    <div className="mt-0.5 text-xs text-slate-500">{a.description}</div>
                  </li>
                ))}
              </ul>
            </Card>
            <Card title="Tool registry">
              <ul className="space-y-1 text-sm text-slate-200">
                {cat.tools.map((t) => (
                  <li key={t.key} className="flex items-center justify-between">
                    <span>{t.label}</span>
                    <span className="pill">{t.key}</span>
                  </li>
                ))}
              </ul>
            </Card>
          </div>
        )
      }
    />
  );
}
