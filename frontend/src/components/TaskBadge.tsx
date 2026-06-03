import clsx from "clsx";
import {
  BrainCircuit, LineChart, Clock4, BookOpen, MessageSquare,
  Layers, HelpCircle,
} from "lucide-react";
import type { TaskDetectionResponse, TaskType } from "@/api/types";

interface Props {
  task: TaskDetectionResponse | null;
  loading?: boolean;
  compact?: boolean;
}

const META: Record<TaskType, { label: string; icon: typeof BrainCircuit; band: string; chip: string }> = {
  classification: {
    label: "Classification",
    icon: BrainCircuit,
    band: "from-emerald-500/40 to-emerald-700/10",
    chip: "border-emerald-700/40 bg-emerald-900/30 text-emerald-100",
  },
  regression: {
    label: "Regression",
    icon: LineChart,
    band: "from-sky-500/40 to-sky-700/10",
    chip: "border-sky-700/40 bg-sky-900/30 text-sky-100",
  },
  time_series_regression: {
    label: "Time-series forecasting",
    icon: Clock4,
    band: "from-violet-500/40 to-violet-700/10",
    chip: "border-violet-700/40 bg-violet-900/30 text-violet-100",
  },
  time_series: {
    label: "Time series (unsup.)",
    icon: Clock4,
    band: "from-violet-500/40 to-violet-700/10",
    chip: "border-violet-700/40 bg-violet-900/30 text-violet-100",
  },
  rag_corpus: {
    label: "RAG corpus",
    icon: BookOpen,
    band: "from-amber-500/40 to-amber-700/10",
    chip: "border-amber-700/40 bg-amber-900/30 text-amber-100",
  },
  nlp: {
    label: "NLP",
    icon: MessageSquare,
    band: "from-rose-500/40 to-rose-700/10",
    chip: "border-rose-700/40 bg-rose-900/30 text-rose-100",
  },
  clustering: {
    label: "Clustering",
    icon: Layers,
    band: "from-cyan-500/40 to-cyan-700/10",
    chip: "border-cyan-700/40 bg-cyan-900/30 text-cyan-100",
  },
  unknown: {
    label: "Unknown",
    icon: HelpCircle,
    band: "from-slate-500/30 to-slate-700/10",
    chip: "border-bg-border bg-bg-panel text-slate-200",
  },
};

export default function TaskBadge({ task, loading, compact }: Props) {
  if (loading || !task) {
    return (
      <div className="flex items-center gap-2 rounded-lg border border-bg-border bg-bg-panel/60 px-3 py-2 text-xs text-slate-400">
        <HelpCircle size={14} />
        {loading ? "Detecting task…" : "Task: not detected yet"}
      </div>
    );
  }
  const m = META[task.task] ?? META.unknown;
  const Icon = m.icon;

  if (compact) {
    return (
      <span
        className={clsx(
          "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium",
          m.chip
        )}
      >
        <Icon size={12} /> {m.label}
      </span>
    );
  }

  return (
    <div
      className={clsx(
        "relative overflow-hidden rounded-xl border border-bg-border bg-gradient-to-r p-4 shadow-inner",
        m.band
      )}
    >
      <div className="flex items-start gap-3">
        <div className={clsx("grid h-10 w-10 shrink-0 place-items-center rounded-lg border", m.chip)}>
          <Icon size={20} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs uppercase tracking-wide text-slate-300">Detected task</span>
            <span className={clsx("rounded-full border px-2 py-0.5 text-xs font-medium", m.chip)}>
              {m.label}
            </span>
            <span className="text-xs text-slate-400">
              · {task.n_samples} samples × {task.n_features} features
            </span>
          </div>
          <p className="mt-1 text-sm text-slate-100">{task.rationale}</p>
          {task.signals.length > 0 && (
            <ul className="mt-1.5 flex flex-wrap gap-1.5">
              {task.signals.map((s, i) => (
                <li key={i} className="rounded-full border border-bg-border bg-bg-panel/60 px-2 py-0.5 text-[11px] text-slate-300">
                  {s}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}
