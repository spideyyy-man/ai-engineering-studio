import { useState } from "react";
import { Copy, Check, Download } from "lucide-react";

interface Props {
  code: string;
  language?: string;
  filename?: string;
}

export default function CodeBlock({ code, language = "python", filename }: Props) {
  const [copied, setCopied] = useState(false);

  function copy() {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }

  function download() {
    if (!filename) return;
    const blob = new Blob([code], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="overflow-hidden rounded-lg border border-bg-border bg-bg-panel">
      <div className="flex items-center justify-between border-b border-bg-border bg-bg-card px-3 py-2 text-xs text-slate-400">
        <span className="font-mono">
          {filename ?? language}
        </span>
        <div className="flex gap-2">
          <button
            className="inline-flex items-center gap-1 rounded px-2 py-1 text-xs hover:bg-bg-panel"
            onClick={copy}
            title="Copy"
          >
            {copied ? <Check size={14} /> : <Copy size={14} />}
            {copied ? "Copied" : "Copy"}
          </button>
          {filename && (
            <button
              className="inline-flex items-center gap-1 rounded px-2 py-1 text-xs hover:bg-bg-panel"
              onClick={download}
              title="Download"
            >
              <Download size={14} /> Download
            </button>
          )}
        </div>
      </div>
      <pre className="max-h-[500px] overflow-auto p-4 font-mono text-xs leading-relaxed text-slate-200">
        {code}
      </pre>
    </div>
  );
}
