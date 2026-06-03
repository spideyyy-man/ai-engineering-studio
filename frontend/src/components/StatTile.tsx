import clsx from "clsx";
import { ReactNode } from "react";

interface Props {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  accent?: string;
}

export default function StatTile({ label, value, hint, accent = "text-studio-data" }: Props) {
  return (
    <div className="rounded-lg border border-bg-border bg-bg-panel p-4">
      <div className="label">{label}</div>
      <div className={clsx("mt-1 text-2xl font-semibold", accent)}>{value}</div>
      {hint && <div className="mt-1 text-xs text-slate-500">{hint}</div>}
    </div>
  );
}
