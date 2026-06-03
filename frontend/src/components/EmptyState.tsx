import { ReactNode } from "react";
import clsx from "clsx";

interface Props {
  icon?: ReactNode;
  title: string;
  description?: ReactNode;
  action?: ReactNode;
  className?: string;
}

export default function EmptyState({ icon, title, description, action, className }: Props) {
  return (
    <div
      className={clsx(
        "flex flex-col items-center justify-center rounded-xl border border-dashed border-bg-border bg-bg-panel/50 p-10 text-center",
        className
      )}
    >
      {icon && <div className="mb-3 text-slate-500">{icon}</div>}
      <h4 className="text-base font-semibold text-slate-100">{title}</h4>
      {description && <p className="mt-1 max-w-md text-sm text-slate-400">{description}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}
