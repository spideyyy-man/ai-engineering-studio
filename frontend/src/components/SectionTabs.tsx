import clsx from "clsx";
import { ReactNode } from "react";

export interface SectionTab {
  key: string;
  label: string;
  icon?: ReactNode;
  badge?: ReactNode;
  disabled?: boolean;
}

interface Props {
  tabs: SectionTab[];
  active: string;
  onChange: (key: string) => void;
  accent?: string;
}

export default function SectionTabs({ tabs, active, onChange, accent = "border-studio-data text-studio-data" }: Props) {
  return (
    <div className="flex flex-wrap items-center gap-1 rounded-xl border border-bg-border bg-bg-panel/60 p-1">
      {tabs.map((t) => {
        const isActive = t.key === active;
        return (
          <button
            key={t.key}
            disabled={t.disabled}
            onClick={() => onChange(t.key)}
            className={clsx(
              "flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm font-medium transition",
              isActive
                ? `bg-bg-card border ${accent}`
                : "text-slate-400 hover:bg-bg-card hover:text-slate-100",
              t.disabled && "cursor-not-allowed opacity-40"
            )}
          >
            {t.icon}
            <span>{t.label}</span>
            {t.badge !== undefined && t.badge !== null && (
              <span className="ml-1 rounded-full bg-bg-panel px-1.5 text-xs">{t.badge}</span>
            )}
          </button>
        );
      })}
    </div>
  );
}
