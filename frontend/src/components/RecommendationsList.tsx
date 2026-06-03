import { Check, Sparkles } from "lucide-react";
import clsx from "clsx";
import type { Recommendation } from "@/api/types";

interface Props {
  recommendations: Recommendation[];
  selected: Set<number>;
  onToggle: (idx: number) => void;
}

const STEP_COLOR: Record<string, string> = {
  imputation: "text-sky-400",
  scaling: "text-violet-400",
  encoding: "text-emerald-400",
  balancing: "text-amber-400",
  drop: "text-red-400",
  deduplicate: "text-slate-300",
};

export default function RecommendationsList({ recommendations, selected, onToggle }: Props) {
  if (recommendations.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-bg-border bg-bg-panel/50 p-6 text-center text-sm text-slate-400">
        <Sparkles className="mx-auto mb-2 text-slate-500" size={20} />
        No preprocessing issues detected — your dataset looks clean.
      </div>
    );
  }

  return (
    <ul className="space-y-2">
      {recommendations.map((rec, idx) => {
        const isOn = selected.has(idx);
        return (
          <li
            key={idx}
            className={clsx(
              "cursor-pointer rounded-lg border bg-bg-panel p-4 transition",
              isOn ? "border-studio-data" : "border-bg-border hover:border-slate-500"
            )}
            onClick={() => onToggle(idx)}
          >
            <div className="flex items-start gap-3">
              <span
                className={clsx(
                  "mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded border",
                  isOn ? "border-studio-data bg-studio-data text-white" : "border-bg-border"
                )}
              >
                {isOn && <Check size={12} />}
              </span>
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span
                    className={clsx(
                      "font-mono text-xs uppercase tracking-wide",
                      STEP_COLOR[rec.step] ?? "text-slate-300"
                    )}
                  >
                    {rec.step}
                  </span>
                  {rec.columns.length > 0 && (
                    <span className="pill">{rec.columns.length} col{rec.columns.length > 1 ? "s" : ""}</span>
                  )}
                </div>
                <p className="mt-1 text-sm text-slate-200">{rec.rationale}</p>
                <code className="mt-1 block truncate font-mono text-xs text-slate-400">
                  {rec.code_hint}
                </code>
                {rec.columns.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {rec.columns.slice(0, 12).map((c) => (
                      <span key={c} className="pill">{c}</span>
                    ))}
                    {rec.columns.length > 12 && (
                      <span className="pill">+{rec.columns.length - 12}</span>
                    )}
                  </div>
                )}
              </div>
            </div>
          </li>
        );
      })}
    </ul>
  );
}
