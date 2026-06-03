import clsx from "clsx";

interface Props {
  severity?: "info" | "warn" | "error" | "ok";
  children: React.ReactNode;
}

const STYLES: Record<string, string> = {
  info: "border-sky-700/40 bg-sky-900/30 text-sky-200",
  warn: "border-amber-700/40 bg-amber-900/30 text-amber-200",
  error: "border-red-700/40 bg-red-900/30 text-red-200",
  ok: "border-emerald-700/40 bg-emerald-900/30 text-emerald-200",
};

export default function MetricBadge({ severity = "info", children }: Props) {
  return (
    <span
      className={clsx(
        "inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium",
        STYLES[severity]
      )}
    >
      {children}
    </span>
  );
}
