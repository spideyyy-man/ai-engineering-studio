import { ReactNode } from "react";

interface Props {
  title: string;
  subtitle?: ReactNode;
  /** Studio text colour class (e.g. "text-studio-data"); drives the accent bar. */
  accent?: string;
  action?: ReactNode;
}

export default function PageHeader({ title, subtitle, accent = "text-studio-data", action }: Props) {
  return (
    <header className="mb-8 flex flex-wrap items-end justify-between gap-4 border-b border-bg-border pb-5">
      <div className="min-w-0">
        <h1 className="flex items-center gap-3 text-[1.75rem] font-semibold leading-tight tracking-tight text-slate-50">
          <span aria-hidden className={`h-6 w-1 shrink-0 rounded-full bg-current ${accent}`} />
          {title}
        </h1>
        {subtitle && <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-400">{subtitle}</p>}
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </header>
  );
}
