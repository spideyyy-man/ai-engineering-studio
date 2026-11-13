import clsx from "clsx";
import { AlertTriangle, Info, ShieldAlert, Wand2, Zap, Star } from "lucide-react";
import type { QualityIssue, RecommendedFix } from "@/api/types";

interface Props {
  issues: QualityIssue[];
  recommendations?: Record<string, RecommendedFix | null>;
  onFix?: (issue: QualityIssue) => void;
  onQuickFix?: (issue: QualityIssue, rec: RecommendedFix) => void;
  busyKey?: string | null;
}

const ICONS = {
  info: Info,
  warn: AlertTriangle,
  error: ShieldAlert,
};

const STYLE: Record<string, { card: string; band: string }> = {
  info:  { card: "border-sky-700/40 bg-sky-900/15",     band: "bg-sky-400" },
  warn:  { card: "border-amber-700/40 bg-amber-900/15", band: "bg-amber-400" },
  error: { card: "border-red-700/40 bg-red-900/15",     band: "bg-red-400" },
};

export function issueKey(i: QualityIssue): string {
  return `${i.code}::${i.column ?? ""}`;
}

export default function QualityIssueList({
  issues,
  recommendations,
  onFix,
  onQuickFix,
  busyKey,
}: Props) {
  if (issues.length === 0) {
    return (
      <div className="rounded-lg border border-emerald-700/40 bg-emerald-900/20 p-4 text-sm text-emerald-200">
        No data-quality issues detected.
      </div>
    );
  }

  return (
    <ul className="space-y-2.5">
      {issues.map((issue, idx) => {
        const Icon = ICONS[issue.severity];
        const sty = STYLE[issue.severity];
        const key = issueKey(issue);
        const rec = recommendations?.[key] ?? null;
        const busy = busyKey === key;

        return (
          <li
            key={idx}
            className={clsx(
              "relative overflow-hidden rounded-lg border pl-4 pr-3 py-3 text-sm transition",
              sty.card
            )}
          >
            <span className={clsx("absolute inset-y-0 left-0 w-1", sty.band)} />
            <div className="flex items-start gap-3">
              <Icon size={18} className="mt-0.5 shrink-0 text-slate-200" />
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-mono text-[10px] text-slate-400">
                    {issue.code}
                  </span>
                  {issue.column && <span className="pill">{issue.column}</span>}
                </div>
                <p className="mt-1 text-slate-100">{issue.message}</p>
                {issue.suggestion && (
                  <p className="mt-0.5 text-xs text-slate-300">→ {issue.suggestion}</p>
                )}
                {rec && (
                  <div className="mt-2 flex items-start gap-1.5 rounded-md border border-emerald-700/30 bg-emerald-900/20 px-2 py-1.5 text-xs text-emerald-100">
                    <Star size={12} className="mt-[2px] shrink-0 text-emerald-300" />
                    <span>
                      <span className="font-semibold">Recommended:</span>{" "}
                      <code className="font-mono">{rec.operation}</code>
                      <span className="ml-1 text-emerald-300/80">— {rec.rationale}</span>
                    </span>
                  </div>
                )}
              </div>
              <div className="ml-2 flex shrink-0 flex-col items-stretch gap-1.5">
                {rec && onQuickFix && (
                  <button
                    className="flex items-center justify-center gap-1.5 rounded-lg border border-emerald-700/40 bg-emerald-700/15 px-2.5 py-1 text-xs font-medium text-emerald-100 hover:bg-emerald-700/30 disabled:cursor-not-allowed disabled:opacity-50"
                    onClick={() => onQuickFix(issue, rec)}
                    title={`Apply ${rec.operation}`}
                    disabled={busy}
                  >
                    <Zap size={12} /> {busy ? "Applying…" : "Quick Fix"}
                  </button>
                )}
                {onFix && (
                  <button
                    className="flex items-center justify-center gap-1.5 rounded-lg border border-bg-border bg-bg-card px-2.5 py-1 text-xs font-medium text-slate-200 hover:border-studio-data hover:text-studio-data"
                    onClick={() => onFix(issue)}
                    title="Configure a fix manually"
                  >
                    <Wand2 size={12} /> Fix
                  </button>
                )}
              </div>
            </div>
          </li>
        );
      })}
    </ul>
  );
}
