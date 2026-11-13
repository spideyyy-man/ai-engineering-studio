import clsx from "clsx";
import { Activity, TrendingUp } from "lucide-react";
import type { BaselineResponse } from "@/api/types";

interface Props {
  baseline: BaselineResponse | null;
  loading?: boolean;
  onRefresh?: () => void;
}

function fmt(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return v.toFixed(3);
}

export default function BaselineCard({ baseline, loading, onRefresh }: Props) {
  if (loading || !baseline) {
    return (
      <div className="rounded-xl border border-bg-border bg-bg-panel/60 p-4 text-xs text-slate-400">
        <div className="flex items-center gap-2">
          <Activity size={14} />
          {loading ? "Training baseline…" : "Baseline not computed yet."}
        </div>
        {!loading && onRefresh && (
          <button className="mt-2 btn-ghost py-1 text-xs" onClick={onRefresh}>
            Compute now
          </button>
        )}
      </div>
    );
  }

  const delta = baseline.delta ?? 0;
  const sign = delta >= 0 ? "+" : "";
  const goodDelta = delta >= 0.1;
  const okDelta = delta >= 0.02;
  const accent = goodDelta ? "text-emerald-300" : okDelta ? "text-amber-300" : "text-red-300";

  // Bar widths — both as positive ratios from 0..1 for visual; for regression r2 can be negative
  const maxScore = Math.max(0.001, baseline.tree_score ?? 0, baseline.dummy_score ?? 0);
  const dummyPct = clamp01((baseline.dummy_score ?? 0) / maxScore) * 100;
  const treePct = clamp01((baseline.tree_score ?? 0) / maxScore) * 100;

  return (
    <div className="rounded-xl border border-bg-border bg-bg-panel/60 p-4">
      <header className="mb-3 flex items-start justify-between gap-2">
        <div>
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <Activity size={12} /> Baseline · {baseline.task}
          </div>
          <div className="mt-0.5 text-sm text-slate-200">
            How much signal does this dataset carry?
          </div>
        </div>
        <div className={clsx("flex items-center gap-1 text-sm font-semibold", accent)}>
          <TrendingUp size={14} /> {sign}{fmt(baseline.delta)}
        </div>
      </header>

      <div className="space-y-2">
        <Row label="Dummy" pct={dummyPct} value={fmt(baseline.dummy_score)}
             metric={baseline.primary_metric} accent="bg-slate-500" />
        <Row label="Decision Tree (d=3)" pct={treePct} value={fmt(baseline.tree_score)}
             metric={baseline.primary_metric} accent="bg-emerald-500" />
      </div>

      <p className="mt-3 rounded-md border border-bg-border bg-bg-card/60 px-3 py-2 text-xs text-slate-300">
        {baseline.interpretation}
      </p>
    </div>
  );
}

function Row({
  label, pct, value, metric, accent,
}: { label: string; pct: number; value: string; metric: string; accent: string }) {
  return (
    <div>
      <div className="flex items-baseline justify-between text-xs">
        <span className="text-slate-300">{label}</span>
        <span className="font-mono text-slate-100">
          {value} <span className="ml-1 text-slate-500">({metric})</span>
        </span>
      </div>
      <div className="mt-1 h-2 w-full rounded-full bg-bg-card">
        <div
          className={clsx("h-2 rounded-full transition-all", accent)}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

function clamp01(x: number): number {
  if (!Number.isFinite(x)) return 0;
  return Math.max(0, Math.min(1, x));
}
