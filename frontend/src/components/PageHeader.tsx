import { ReactNode } from "react";

interface Props {
  title: string;
  subtitle?: ReactNode;
  accent?: string;
  action?: ReactNode;
}

export default function PageHeader({ title, subtitle, accent = "text-studio-data", action }: Props) {
  return (
    <header className="mb-6 flex items-start justify-between gap-4">
      <div>
        <h1 className={`text-2xl font-bold tracking-tight ${accent}`}>{title}</h1>
        {subtitle && <p className="mt-1 text-sm text-slate-400">{subtitle}</p>}
      </div>
      {action}
    </header>
  );
}
