import { ReactNode, useState } from "react";
import clsx from "clsx";
import { ChevronDown, ChevronUp } from "lucide-react";
import TaskBadge from "./TaskBadge";
import PrettySelect from "./PrettySelect";
import type { DatasetSummary, ProfileResponse, QualityReport, TaskDetectionResponse } from "@/api/types";
import { formatBytes, formatNumber } from "@/lib/format";

export interface CockpitTab {
  key: string;
  label: string;
  icon?: ReactNode;
  badge?: ReactNode;
}

interface Props {
  task: TaskDetectionResponse | null;
  taskLoading: boolean;
  summary: DatasetSummary;
  profile: ProfileResponse | null;
  quality: QualityReport | null;
  tabs: CockpitTab[];
  activeSection: string;
  onSectionChange: (k: string) => void;
}

export default function CockpitBar({
  task, taskLoading, summary, profile, quality,
  tabs, activeSection, onSectionChange,
}: Props) {
  const [expanded, setExpanded] = useState(false);

  const errorCount = quality?.summary.error ?? 0;
  const warnCount = quality?.summary.warn ?? 0;
  const issuesTotal = quality?.issues.length ?? 0;
  const duplicates = profile?.duplicates ?? null;


  return (
    <div className="sticky top-0 z-30 -mx-1 mb-4 rounded-xl border border-bg-border bg-bg/85 backdrop-blur shadow-lg">
      {/* Row 1: task chip + inline stats */}
      <div className="flex flex-wrap items-center gap-3 px-3 py-2">
        <button
          onClick={() => setExpanded((v) => !v)}
          className="flex items-center gap-2 rounded-lg border border-bg-border bg-bg-panel/80 px-2.5 py-1.5 text-xs text-slate-100 hover:border-slate-500"
          title="Expand for full task analysis"
        >
          <TaskBadge task={task} loading={taskLoading} compact />
          {expanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        </button>

        <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
          <StatPill label="Rows" value={formatNumber(summary.n_rows)} />
          <StatPill label="Cols" value={String(summary.n_cols)} />
          <StatPill label="Size" value={formatBytes(summary.size_bytes)} />
          {duplicates !== null && (
            <StatPill
              label="Dupes"
              value={formatNumber(duplicates)}
              accent={duplicates > 0 ? "text-amber-300" : undefined}
            />
          )}
          {quality && (
            <StatPill
              label="Issues"
              value={String(issuesTotal)}
              hint={`${errorCount}e · ${warnCount}w`}
              accent={
                errorCount > 0 ? "text-red-300" :
                warnCount > 0 ? "text-amber-300" :
                "text-emerald-300"
              }
            />
          )}
        </div>
      </div>

      {/* Expanded task detail */}
      {expanded && task && (
        <div className="border-t border-bg-border bg-bg-panel/40 px-3 py-3">
          <TaskBadge task={task} loading={taskLoading} />
        </div>
      )}

      {/* Row 2: prominent section dropdown + pill rail */}
      <div className="flex flex-wrap items-center gap-3 border-t border-bg-border bg-bg-panel/60 px-3 py-2.5">
        <PrettySelect
          label="Section"
          value={activeSection}
          size="md"
          options={tabs.map((t) => ({
            value: t.key,
            label: t.label,
            badge: t.badge,
          }))}
          onChange={onSectionChange}
          title="Jump to a section"
        />

        <span className="hidden text-xs text-slate-500 md:inline">or</span>

        <div className="flex flex-wrap gap-1">
          {tabs.map((t) => {
            const isActive = t.key === activeSection;
            return (
              <button
                key={t.key}
                onClick={() => onSectionChange(t.key)}
                className={clsx(
                  "flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium transition",
                  isActive
                    ? "bg-bg-card text-slate-100 ring-1 ring-studio-data"
                    : "text-slate-400 hover:bg-bg-card hover:text-slate-100"
                )}
              >
                {t.icon}
                <span>{t.label}</span>
                {t.badge !== undefined && t.badge !== null && (
                  <span className={clsx(
                    "rounded-full px-1.5 text-[10px]",
                    isActive ? "bg-bg-panel text-slate-200" : "bg-bg-panel/60 text-slate-400"
                  )}>{t.badge}</span>
                )}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function StatPill({
  label, value, hint, accent,
}: { label: string; value: string; hint?: string; accent?: string }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-md border border-bg-border bg-bg-panel/60 px-2 py-1">
      <span className="text-slate-500">{label}</span>
      <span className={clsx("font-mono font-semibold", accent ?? "text-slate-100")}>{value}</span>
      {hint && <span className="text-slate-500">{hint}</span>}
    </span>
  );
}
