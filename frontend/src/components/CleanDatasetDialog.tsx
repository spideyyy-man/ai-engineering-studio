import { useState, useEffect } from "react";
import { Sparkles, AlertCircle } from "lucide-react";

import Modal from "./Modal";
import { dataApi } from "@/api";
import type { AutoFixResponse } from "@/api/types";

interface Props {
  open: boolean;
  onClose: () => void;
  datasetId: string;
  datasetName: string;
  targetCol: string | null;
  onApplied: (result: AutoFixResponse) => void;
}

export default function CleanDatasetDialog({
  open, onClose, datasetId, datasetName, targetCol, onApplied,
}: Props) {
  const [newName, setNewName] = useState<string>("");
  const [maxRounds, setMaxRounds] = useState<number>(5);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setNewName(`${datasetName}_clean`);
      setMaxRounds(5);
      setError(null);
    }
  }, [open, datasetName]);

  async function run() {
    if (!datasetId) return;
    setRunning(true);
    setError(null);
    try {
      const r = await dataApi.cleanDataset(datasetId, {
        new_name: newName.trim() || undefined,
        target_col: targetCol,
        max_rounds: maxRounds,
      });
      onApplied(r);
      onClose();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setRunning(false);
    }
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      width="max-w-lg"
      title={
        <span className="flex items-center gap-2">
          <Sparkles size={16} className="text-emerald-300" /> Create clean dataset
        </span>
      }
      subtitle="Iteratively apply the recommended fix to every problem column until no fixable issues remain. Produces a brand-new dataset; the original is untouched."
      footer={
        <>
          <button className="btn-ghost" onClick={onClose} disabled={running}>
            Cancel
          </button>
          <button
            className="btn-primary bg-emerald-400 hover:bg-emerald-300"
            onClick={run}
            disabled={running}
          >
            <Sparkles size={14} /> {running ? "Cleaning… (can take 10–30s)" : "Run clean"}
          </button>
        </>
      }
    >
      {error && (
        <div className="mb-3 flex items-start gap-2 rounded-lg border border-red-700/40 bg-red-900/20 px-3 py-2 text-sm text-red-200">
          <AlertCircle size={16} className="mt-0.5 shrink-0" />
          {error}
        </div>
      )}

      <div className="space-y-3">
        <div>
          <label className="label">New dataset name</label>
          <input
            className="input"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder={`${datasetName}_clean`}
          />
        </div>
        <div>
          <label className="label flex items-center justify-between">
            <span>Max rounds</span>
            <span className="font-mono text-xs text-slate-500">{maxRounds}</span>
          </label>
          <input
            type="range"
            min={1}
            max={10}
            step={1}
            value={maxRounds}
            onChange={(e) => setMaxRounds(parseInt(e.target.value, 10))}
            className="w-full"
          />
          <p className="mt-1 text-xs text-slate-500">
            The loop stops early as soon as no further fixable issues remain. 5 rounds is plenty for nearly any dataset.
          </p>
        </div>

        <div className="rounded-lg border border-bg-border bg-bg-panel/60 p-3 text-xs text-slate-400">
          <div className="mb-1 font-semibold text-slate-200">What this does</div>
          <ul className="space-y-1">
            <li>• Re-scans quality after each fix so newly-revealed issues get caught.</li>
            <li>• Picks the recommended method for every issue (median for numeric NaN, mode for categorical, drop for IDs/constants, log1p replace for skew, IQR clip for outliers, etc.).</li>
            <li>• <span className="font-mono">class_imbalance</span> is left as-is — it's handled at training time (class_weight / SMOTE in the trainer).</li>
            <li>• Original dataset is never overwritten.</li>
          </ul>
        </div>
      </div>
    </Modal>
  );
}
