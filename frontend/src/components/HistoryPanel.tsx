import { useState } from "react";
import { History, Undo2, ChevronDown, ChevronRight, ArrowRight } from "lucide-react";
import StatsDiffPanel from "./StatsDiffPanel";
import type { HistoryEntry } from "@/api/types";

interface Props {
  entries: HistoryEntry[];
  onUndo: () => Promise<void> | void;
  busy?: boolean;
}

export default function HistoryPanel({ entries, onUndo, busy }: Props) {
  const [expanded, setExpanded] = useState<number | null>(null);

  if (entries.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-bg-border bg-bg-panel/50 p-4 text-center text-xs text-slate-400">
        <History size={16} className="mx-auto mb-1.5 text-slate-500" />
        No transforms applied yet. Use Quick Fix or Auto-fix all to start tracking history.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="text-xs uppercase tracking-wide text-slate-400">
          {entries.length} step{entries.length === 1 ? "" : "s"}
        </div>
        <button
          className="flex items-center gap-1.5 rounded-lg border border-amber-700/40 bg-amber-700/15 px-2.5 py-1 text-xs font-medium text-amber-100 hover:bg-amber-700/30 disabled:cursor-not-allowed disabled:opacity-50"
          disabled={busy}
          onClick={() => void onUndo()}
        >
          <Undo2 size={12} /> {busy ? "Undoing…" : "Undo last"}
        </button>
      </div>

      <ol className="space-y-1.5">
        {entries.map((e, i) => {
          const isOpen = expanded === e.step;
          const diff = e.stats_diff_json ?? {};
          return (
            <li
              key={e.id}
              className="rounded-lg border border-bg-border bg-bg-panel/60"
            >
              <button
                onClick={() => setExpanded(isOpen ? null : e.step)}
                className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs hover:bg-bg-card"
              >
                {isOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                <span className="grid h-5 w-5 shrink-0 place-items-center rounded-full border border-bg-border bg-bg-card font-mono text-[10px] text-slate-300">
                  {e.step}
                </span>
                <span className="font-mono text-emerald-300">{e.operation}</span>
                <span className="ml-1 truncate text-slate-300">{e.summary}</span>
                <span className="ml-auto whitespace-nowrap font-mono text-[10px] text-slate-500">
                  {e.n_rows_before}x{e.n_cols_before} <ArrowRight size={9} className="inline -mt-0.5" /> {e.n_rows_after}x{e.n_cols_after}
                </span>
              </button>
              {isOpen && Object.keys(diff).length > 0 && (
                <div className="border-t border-bg-border px-3 py-2">
                  <StatsDiffPanel diff={diff} title="" />
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
