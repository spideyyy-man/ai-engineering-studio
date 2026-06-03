import { ReactNode, useEffect } from "react";
import { X } from "lucide-react";

interface Props {
  open: boolean;
  title?: ReactNode;
  subtitle?: ReactNode;
  onClose: () => void;
  width?: string;
  children: ReactNode;
  footer?: ReactNode;
}

export default function Modal({
  open,
  title,
  subtitle,
  onClose,
  width = "max-w-2xl",
  children,
  footer,
}: Props) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className={`relative max-h-[85vh] w-full ${width} overflow-hidden rounded-2xl border border-bg-border bg-bg-card shadow-2xl`}
        onClick={(e) => e.stopPropagation()}
      >
        {(title || subtitle) && (
          <header className="flex items-start justify-between gap-3 border-b border-bg-border bg-bg-panel px-5 py-3">
            <div>
              {title && <h3 className="text-base font-semibold text-slate-100">{title}</h3>}
              {subtitle && <p className="mt-0.5 text-xs text-slate-400">{subtitle}</p>}
            </div>
            <button
              className="rounded p-1 text-slate-400 hover:bg-bg-card hover:text-slate-100"
              onClick={onClose}
            >
              <X size={16} />
            </button>
          </header>
        )}
        <div className="max-h-[65vh] overflow-y-auto p-5">{children}</div>
        {footer && (
          <footer className="flex items-center justify-end gap-2 border-t border-bg-border bg-bg-panel px-5 py-3">
            {footer}
          </footer>
        )}
      </div>
    </div>
  );
}
