import { ReactNode } from "react";
import clsx from "clsx";
import { ChevronDown } from "lucide-react";

export interface PrettyOption {
  value: string;
  label: string;
  badge?: ReactNode;
  hint?: string;
}

interface Props {
  label?: ReactNode;
  value: string;
  options: PrettyOption[];
  onChange: (value: string) => void;
  /** Visual size — sm | md | lg */
  size?: "sm" | "md" | "lg";
  /** Stretch the trigger to fill its container */
  block?: boolean;
  /** Optional accent color for active border */
  accent?: string;
  className?: string;
  title?: string;
}

/**
 * A native `<select>` styled as an unmistakable dropdown button.
 * Renders the trigger as a button with a visible chevron, so users can't
 * miss it on dark backgrounds where the native arrow is barely legible.
 *
 * Keeps accessibility + keyboard nav of native <select> while looking
 * like a designed control.
 */
export default function PrettySelect({
  label,
  value,
  options,
  onChange,
  size = "md",
  block = false,
  accent = "focus-within:border-studio-data",
  className,
  title,
}: Props) {
  const active = options.find((o) => o.value === value);

  const sizeCls = {
    sm: "h-8 px-2.5 text-xs",
    md: "h-9 px-3 text-sm",
    lg: "h-11 px-4 text-base",
  }[size];

  return (
    <div className={clsx("inline-flex items-center gap-2", block && "w-full", className)}>
      {label && (
        <span className="shrink-0 text-[11px] font-semibold text-slate-400">
          {label}
        </span>
      )}
      <div
        className={clsx(
          "relative inline-flex items-center rounded-lg border border-bg-border bg-bg-card shadow-sm transition",
          "hover:border-slate-500",
          accent,
          block && "w-full",
        )}
        title={title}
      >
        {/* Visible button face */}
        <div className={clsx("flex items-center gap-2 pr-8 font-medium text-slate-100", sizeCls)}>
          <span className="truncate">{active?.label ?? value}</span>
          {active?.badge !== undefined && active?.badge !== null && (
            <span className="rounded-full bg-bg-panel px-1.5 py-0.5 text-[10px] font-semibold text-slate-300">
              {active.badge}
            </span>
          )}
        </div>
        <ChevronDown
          size={14}
          className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-400"
        />
        {/* Native <select> sits on top, fully transparent, captures clicks + keyboard */}
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className={clsx(
            "absolute inset-0 cursor-pointer appearance-none bg-transparent text-transparent outline-none",
            "[&>option]:bg-bg-card [&>option]:text-slate-100",
          )}
        >
          {options.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
              {o.badge !== undefined && o.badge !== null ? `  (${o.badge})` : ""}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
