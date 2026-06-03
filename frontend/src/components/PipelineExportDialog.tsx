import { useEffect, useState } from "react";
import { Code2, Download } from "lucide-react";

import Modal from "./Modal";
import CodeBlock from "./CodeBlock";
import { dataApi } from "@/api";

interface Props {
  open: boolean;
  onClose: () => void;
  datasetId: string;
}

export default function PipelineExportDialog({ open, onClose, datasetId }: Props) {
  const [filename, setFilename] = useState<string>("preprocess.py");
  const [code, setCode] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open || !datasetId) return;
    setError(null);
    setCode("");
    setLoading(true);
    dataApi
      .pipelineExport(datasetId)
      .then((r) => {
        setFilename(r.filename);
        setCode(r.code);
      })
      .catch((e) => setError((e as Error).message))
      .finally(() => setLoading(false));
  }, [open, datasetId]);

  return (
    <Modal
      open={open}
      onClose={onClose}
      width="max-w-4xl"
      title={
        <span className="flex items-center gap-2">
          <Code2 size={16} className="text-studio-data" /> Pipeline export
        </span>
      }
      subtitle="A runnable preprocess.py that replays every transform applied to this dataset."
      footer={
        <>
          <button className="btn-ghost" onClick={onClose}>Close</button>
          {code && (
            <a
              className="btn-primary"
              href={`data:text/x-python;charset=utf-8,${encodeURIComponent(code)}`}
              download={filename}
            >
              <Download size={14} /> Download {filename}
            </a>
          )}
        </>
      }
    >
      {error && (
        <div className="mb-3 rounded-lg border border-red-700/40 bg-red-900/20 px-3 py-2 text-sm text-red-200">
          {error}
        </div>
      )}
      {loading ? (
        <p className="text-sm text-slate-400">Generating…</p>
      ) : code ? (
        <CodeBlock code={code} filename={filename} />
      ) : (
        <p className="text-sm text-slate-500">
          No transforms applied yet — apply at least one to generate a pipeline.
        </p>
      )}
    </Modal>
  );
}
