import { ReactNode } from "react";
import PageHeader from "@/components/PageHeader";
import Card from "@/components/Card";

interface Props {
  title: string;
  subtitle: string;
  accent: string;
  phase: string;
  capabilities: string[];
  comingSoon: string[];
  extra?: ReactNode;
}

export default function StubPage({
  title,
  subtitle,
  accent,
  phase,
  capabilities,
  comingSoon,
  extra,
}: Props) {
  return (
    <div className="mx-auto max-w-5xl px-8 py-8">
      <PageHeader
        title={title}
        subtitle={subtitle}
        accent={accent}
        action={<span className="pill">{phase}</span>}
      />

      <div className="grid gap-6 md:grid-cols-2">
        <Card title="Available now" subtitle="Live endpoints in this studio.">
          <ul className="space-y-2 text-sm">
            {capabilities.map((c) => (
              <li key={c} className="flex items-start gap-2 text-slate-200">
                <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-400" />
                {c}
              </li>
            ))}
          </ul>
        </Card>

        <Card title="Coming next" subtitle="Planned in this phase.">
          <ul className="space-y-2 text-sm">
            {comingSoon.map((c) => (
              <li key={c} className="flex items-start gap-2 text-slate-400">
                <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-slate-500" />
                {c}
              </li>
            ))}
          </ul>
        </Card>
      </div>

      {extra}
    </div>
  );
}
