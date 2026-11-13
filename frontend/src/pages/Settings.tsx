import { useEffect, useState } from "react";
import PageHeader from "@/components/PageHeader";
import Card from "@/components/Card";
import { settingsApi } from "@/api";
import type { AppSettings, ProviderInfo } from "@/api/types";
import MetricBadge from "@/components/MetricBadge";

export default function SettingsPage() {
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [pingResults, setPingResults] = useState<Record<string, boolean | "loading">>({});

  useEffect(() => {
    void settingsApi.get().then(setSettings).catch(() => setSettings(null));
    void settingsApi.providers().then(setProviders).catch(() => setProviders([]));
  }, []);

  async function ping(name: string) {
    setPingResults((p) => ({ ...p, [name]: "loading" }));
    try {
      const r = await settingsApi.testLLM(name);
      setPingResults((p) => ({ ...p, [name]: r.ok }));
    } catch {
      setPingResults((p) => ({ ...p, [name]: false }));
    }
  }

  return (
    <div className="mx-auto max-w-4xl px-8 py-8">
      <PageHeader
        title="Settings"
        subtitle="Defaults are loaded from .env. Edit the file and restart the backend to change them persistently."
        accent="text-studio-settings"
      />

      <div className="space-y-6">
        <Card title="Defaults">
          {settings ? (
            <dl className="grid gap-3 sm:grid-cols-2">
              <Row k="LLM provider" v={`${settings.llm_default_provider} · ${settings.llm_default_model}`} />
              <Row
                k="Embedding provider"
                v={`${settings.embedding_default_provider} · ${settings.embedding_default_model}`}
              />
              <Row k="Vector store" v={settings.vector_store_default} />
              <Row k="Ollama URL" v={settings.ollama_base_url} />
              <Row
                k="OpenAI key"
                v={settings.openai_configured ? "configured" : "not set"}
                ok={settings.openai_configured}
              />
              <Row
                k="Anthropic key"
                v={settings.anthropic_configured ? "configured" : "not set"}
                ok={settings.anthropic_configured}
              />
            </dl>
          ) : (
            <p className="text-sm text-slate-500">Loading…</p>
          )}
        </Card>

        <Card title="Providers" subtitle="Ping each provider to confirm connectivity.">
          <ul className="space-y-2">
            {providers.map((p) => {
              const r = pingResults[p.name];
              return (
                <li
                  key={p.name}
                  className="flex items-center justify-between rounded-lg border border-bg-border bg-bg-panel px-4 py-3"
                >
                  <div>
                    <div className="font-medium text-slate-100">{p.name}</div>
                    <div className="text-xs text-slate-500">
                      default model: {p.default_model || "—"} · key{" "}
                      <MetricBadge severity={p.configured ? "ok" : "warn"}>
                        {p.configured ? "configured" : "not set"}
                      </MetricBadge>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {r === "loading" && <span className="text-xs text-slate-400">…</span>}
                    {r === true && <MetricBadge severity="ok">reachable</MetricBadge>}
                    {r === false && <MetricBadge severity="error">unreachable</MetricBadge>}
                    <button className="btn-ghost" onClick={() => ping(p.name)}>
                      Ping
                    </button>
                  </div>
                </li>
              );
            })}
          </ul>
        </Card>
      </div>
    </div>
  );
}

function Row({ k, v, ok }: { k: string; v: string; ok?: boolean }) {
  return (
    <div className="rounded-lg border border-bg-border bg-bg-panel px-3 py-2">
      <div className="text-xs text-slate-500">{k}</div>
      <div
        className={`mt-0.5 font-mono text-sm ${
          ok === false ? "text-amber-300" : "text-slate-100"
        }`}
      >
        {v}
      </div>
    </div>
  );
}
