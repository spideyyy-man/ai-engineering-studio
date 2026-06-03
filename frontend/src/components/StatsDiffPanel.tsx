import { ArrowRight } from "lucide-react";
import type { ColumnStat, StatsDiff } from "@/api/types";

interface Props {
  diff: StatsDiff;
  title?: string;
}

function fmt(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return String(v);
    return Number.isInteger(v) ? v.toString() : v.toFixed(3);
  }
  return String(v);
}

const NUMERIC_KEYS: (keyof ColumnStat)[] = ["missing", "nunique", "mean", "median", "std", "min", "max"];
const CATEGORICAL_KEYS: (keyof ColumnStat)[] = ["missing", "nunique", "top_value", "top_freq"];

function changed(b: unknown, a: unknown): boolean {
  if (b === a) return false;
  if (typeof b === "number" && typeof a === "number") {
    return Math.abs(b - a) > 1e-9;
  }
  return true;
}

export default function StatsDiffPanel({ diff, title = "Column stats: before → after" }: Props) {
  const entries = Object.entries(diff);
  if (entries.length === 0) return null;

  return (
    <div className="rounded-lg border border-bg-border bg-bg-panel/60 p-3 text-xs">
      <div className="mb-2 flex items-center gap-2 text-slate-300">
        <ArrowRight size={12} className="text-emerald-300" />
        <span className="font-semibold">{title}</span>
      </div>
      <div className="space-y-2">
        {entries.map(([col, { before, after }]) => {
          const isNumeric =
            before.dtype?.match(/int|float|number/i) || after.dtype?.match(/int|float|number/i);
          const keys = isNumeric ? NUMERIC_KEYS : CATEGORICAL_KEYS;

          const dropped = before.present && !after.present;
          const added = !before.present && after.present;

          return (
            <div key={col} className="rounded border border-bg-border bg-bg-card/60 p-2">
              <div className="mb-1 flex items-center gap-2">
                <span className="font-mono text-sm text-slate-100">{col}</span>
                <span className="pill">{after.dtype ?? before.dtype ?? "?"}</span>
                {dropped && <span className="rounded bg-red-900/50 px-1.5 text-[10px] text-red-200">dropped</span>}
                {added && <span className="rounded bg-emerald-900/50 px-1.5 text-[10px] text-emerald-200">added</span>}
              </div>
              {dropped ? (
                <p className="text-slate-400">Column removed by the transform.</p>
              ) : added ? (
                <p className="text-slate-400">New column created by the transform.</p>
              ) : (
                <table className="w-full">
                  <thead>
                    <tr className="text-left text-[10px] uppercase tracking-wide text-slate-500">
                      <th className="py-0.5">stat</th>
                      <th className="py-0.5">before</th>
                      <th className="py-0.5"></th>
                      <th className="py-0.5">after</th>
                    </tr>
                  </thead>
                  <tbody>
                    {keys.map((k) => {
                      const b = before[k] as unknown;
                      const a = after[k] as unknown;
                      const c = changed(b, a);
                      return (
                        <tr key={String(k)} className={c ? "text-slate-100" : "text-slate-500"}>
                          <td className="py-0.5 font-mono text-[11px]">{String(k)}</td>
                          <td className="py-0.5 font-mono">{fmt(b)}</td>
                          <td className="py-0.5 text-slate-500">→</td>
                          <td className="py-0.5 font-mono">
                            {fmt(a)}{" "}
                            {c && typeof b === "number" && typeof a === "number" && (
                              <span className="ml-1 text-[10px] text-emerald-300">
                                ({a - b >= 0 ? "+" : ""}
                                {(a - b).toFixed(3)})
                              </span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
