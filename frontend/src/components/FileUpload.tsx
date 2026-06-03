import { useState, useRef, DragEvent } from "react";
import clsx from "clsx";
import { Upload } from "lucide-react";

interface Props {
  accept?: string;
  disabled?: boolean;
  onFile: (file: File) => Promise<void> | void;
  hint?: string;
}

export default function FileUpload({
  accept = ".csv,.tsv,.xlsx,.xls,.parquet,.json",
  disabled = false,
  onFile,
  hint,
}: Props) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handle(file: File) {
    setError(null);
    setUploading(true);
    try {
      await onFile(file);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(false);
    }
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) void handle(f);
  }

  return (
    <div>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => !disabled && !uploading && inputRef.current?.click()}
        className={clsx(
          "cursor-pointer rounded-xl border-2 border-dashed p-8 text-center transition",
          dragging
            ? "border-studio-data bg-studio-data/10"
            : "border-bg-border bg-bg-panel/40 hover:border-slate-500",
          disabled && "cursor-not-allowed opacity-50"
        )}
      >
        <Upload className="mx-auto mb-2 text-slate-400" size={28} />
        <div className="text-sm font-medium text-slate-200">
          {uploading ? "Uploading…" : "Drop a file or click to browse"}
        </div>
        <div className="mt-1 text-xs text-slate-500">
          {hint ?? "CSV, TSV, XLSX, Parquet, JSON"}
        </div>
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          disabled={disabled || uploading}
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void handle(f);
            e.target.value = "";
          }}
        />
      </div>
      {error && <p className="mt-2 text-sm text-red-400">{error}</p>}
    </div>
  );
}
