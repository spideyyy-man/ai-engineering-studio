import { useEffect, useMemo, useState } from "react";
import { Sparkles, Wand2, Check, AlertCircle, Star } from "lucide-react";

import Modal from "./Modal";
import { dataApi } from "@/api";
import type {
  DatasetSummary,
  QualityIssue,
  RecommendedFix,
  TransformOperation,
  TransformResponse,
} from "@/api/types";

interface Props {
  open: boolean;
  onClose: () => void;
  issue: QualityIssue | null;
  datasetId: string;
  summary: DatasetSummary | null;
  onApplied: (result: TransformResponse) => void;
}

export default function FixDialog({
  open,
  onClose,
  issue,
  datasetId,
  summary,
  onApplied,
}: Props) {
  const [ops, setOps] = useState<TransformOperation[]>([]);
  const [opKey, setOpKey] = useState<string | null>(null);
  const [params, setParams] = useState<Record<string, unknown>>({});
  const [inplace, setInplace] = useState<boolean>(false);
  const [newName, setNewName] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [recommendation, setRecommendation] = useState<RecommendedFix | null>(null);
  // Track if user manually changed the op — we won't overwrite their choice with the recommendation.
  const [userChose, setUserChose] = useState(false);

  useEffect(() => {
    if (!open || !issue) return;
    setError(null);
    setOpKey(null);
    setParams({});
    setNewName("");
    setUserChose(false);
    setRecommendation(null);
    setLoading(true);

    // Fetch fixes + recommendation in parallel
    Promise.all([
      dataApi.fixesFor(issue.code),
      datasetId
        ? dataApi.recommendedFix(datasetId, issue.code, issue.column).catch(() => null)
        : Promise.resolve(null),
    ])
      .then(([fixes, rec]) => {
        setOps(fixes.operations);
        setRecommendation(rec);
        // Pre-select the recommended op (and pre-fill its params)
        if (rec && fixes.operations.find((o) => o.key === rec.operation)) {
          setOpKey(rec.operation);
          setParams(rec.params ?? {});
        }
      })
      .catch((e) => setError((e as Error).message))
      .finally(() => setLoading(false));
  }, [open, issue, datasetId]);

  // When user picks a DIFFERENT op (not the recommended one), pre-fill its params from defaults.
  useEffect(() => {
    if (!opKey || !userChose) return;
    const op = ops.find((o) => o.key === opKey);
    if (!op) return;
    const next: Record<string, unknown> = {};
    for (const [pk, ps] of Object.entries(op.params)) {
      if (ps.type === "columns") {
        next[pk] = issue?.column ? [issue.column] : [];
      } else if (ps.default !== undefined && ps.default !== null) {
        next[pk] = ps.default;
      } else {
        next[pk] = null;
      }
    }
    setParams(next);
  }, [opKey, ops, issue, userChose]);

  const selected = useMemo(
    () => (opKey ? ops.find((o) => o.key === opKey) ?? null : null),
    [opKey, ops]
  );

  const columnOptions = summary?.columns ?? [];

  async function apply() {
    if (!opKey || !issue) return;
    setError(null);
    setLoading(true);
    try {
      const result = await dataApi.transform(datasetId, {
        operation: opKey,
        params: coerceParams(params),
        inplace,
        new_name: !inplace ? newName.trim() || undefined : undefined,
      });
      onApplied(result);
      onClose();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      width="max-w-3xl"
      title={
        <span className="flex items-center gap-2">
          <Wand2 size={16} className="text-studio-data" />
          Fix: {issue?.code}
        </span>
      }
      subtitle={issue?.message}
      footer={
        <>
          <label className="mr-auto flex items-center gap-2 text-xs text-slate-300">
            <input
              type="checkbox"
              checked={inplace}
              onChange={(e) => setInplace(e.target.checked)}
            />
            <span className="font-mono">inplace = {inplace ? "true" : "false"}</span>
            <span className="text-slate-500">
              {inplace
                ? "(overwrite the original file)"
                : "(save as a new dataset row)"}
            </span>
          </label>
          {!inplace && (
            <input
              className="input max-w-[180px] py-1 text-xs"
              placeholder="New dataset name (optional)"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
            />
          )}
          <button className="btn-ghost" onClick={onClose} disabled={loading}>
            Cancel
          </button>
          <button
            className="btn-primary"
            onClick={apply}
            disabled={!opKey || loading}
          >
            {loading ? "Applying…" : "Apply"}
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

      {!issue ? null : ops.length === 0 ? (
        <p className="text-sm text-slate-500">
          No automated fixes registered for <span className="font-mono">{issue.code}</span> yet.
        </p>
      ) : (
        <>
          {recommendation && (
            <div className="mb-3 flex items-start gap-2 rounded-lg border border-emerald-700/30 bg-emerald-900/20 px-3 py-2 text-sm text-emerald-100">
              <Star size={14} className="mt-0.5 shrink-0 text-emerald-300" />
              <div>
                <span className="font-semibold">Recommended:</span>{" "}
                <code className="font-mono">{recommendation.operation}</code>
                <p className="text-xs text-emerald-300/90">{recommendation.rationale}</p>
              </div>
            </div>
          )}
          <div className="mb-2 text-xs text-slate-500">
            Pick a method
          </div>
          <ul className="grid gap-2 sm:grid-cols-2">
            {ops.map((op) => {
              const active = op.key === opKey;
              const isRec = recommendation?.operation === op.key;
              return (
                <li key={op.key}>
                  <button
                    type="button"
                    onClick={() => {
                      setUserChose(true);
                      setOpKey(op.key);
                    }}
                    className={`relative w-full rounded-lg border p-3 text-left transition ${
                      active
                        ? "border-studio-data bg-studio-data/5"
                        : "border-bg-border bg-bg-panel hover:border-slate-500"
                    }`}
                  >
                    {isRec && (
                      <span className="absolute right-2 top-2 inline-flex items-center gap-0.5 rounded-full border border-emerald-700/40 bg-emerald-900/40 px-1.5 py-0.5 text-[10px] font-medium text-emerald-200">
                        <Star size={9} /> recommended
                      </span>
                    )}
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-slate-100">{op.label}</span>
                      {active && (
                        <span className="grid h-4 w-4 place-items-center rounded-full bg-studio-data text-on-accent">
                          <Check size={11} />
                        </span>
                      )}
                    </div>
                    <p className="mt-1 text-xs text-slate-400">{op.description}</p>
                  </button>
                </li>
              );
            })}
          </ul>

          {selected && (
            <div className="mt-5 space-y-3 rounded-lg border border-bg-border bg-bg-panel/60 p-4">
              <div className="flex items-center gap-2 text-xs text-slate-500">
                <Sparkles size={12} /> Configure
              </div>

              {Object.entries(selected.params).map(([pk, ps]) => (
                <div key={pk}>
                  <label className="label flex items-center justify-between">
                    <span>
                      {pk}{" "}
                      <span className="font-mono text-[10px] text-slate-500">
                        {ps.type}
                      </span>
                    </span>
                    {ps.required && <span className="text-[10px] text-amber-400">required</span>}
                  </label>

                  {ps.type === "columns" && (
                    <ColumnPicker
                      scope={ps.scope}
                      columns={columnOptions}
                      value={(params[pk] as string[]) ?? []}
                      onChange={(v) => setParams((s) => ({ ...s, [pk]: v }))}
                      hint={ps.help}
                    />
                  )}
                  {(ps.type === "string" || ps.type === "mapping") && (
                    <input
                      className="input"
                      value={String(params[pk] ?? ps.default ?? "")}
                      onChange={(e) => setParams((s) => ({ ...s, [pk]: e.target.value }))}
                      placeholder={ps.help}
                    />
                  )}
                  {(ps.type === "float" || ps.type === "int") && (
                    <input
                      type="number"
                      className="input"
                      value={String(params[pk] ?? ps.default ?? "")}
                      min={ps.min}
                      max={ps.max}
                      step={ps.step ?? (ps.type === "int" ? 1 : 0.1)}
                      onChange={(e) =>
                        setParams((s) => ({
                          ...s,
                          [pk]:
                            ps.type === "int"
                              ? parseInt(e.target.value, 10)
                              : parseFloat(e.target.value),
                        }))
                      }
                    />
                  )}
                  {ps.type === "bool" && (
                    <label className="flex items-center gap-2 text-sm text-slate-200">
                      <input
                        type="checkbox"
                        checked={Boolean(params[pk] ?? ps.default)}
                        onChange={(e) =>
                          setParams((s) => ({ ...s, [pk]: e.target.checked }))
                        }
                      />
                      {ps.help ?? "enabled"}
                    </label>
                  )}

                  {ps.help && ps.type !== "string" && (
                    <p className="mt-1 text-xs text-slate-500">{ps.help}</p>
                  )}
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </Modal>
  );
}

// ---------------------------------------------------------------- helpers

function coerceParams(p: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(p)) {
    if (typeof v === "string" && /^-?\d+(\.\d+)?$/.test(v.trim())) {
      // string-typed param that LOOKS numeric — preserve as string for fill_constant; backend handles
      out[k] = v;
    } else {
      out[k] = v;
    }
  }
  return out;
}

function ColumnPicker({
  scope,
  columns,
  value,
  onChange,
  hint,
}: {
  scope?: string;
  columns: { name: string; dtype: string }[];
  value: string[];
  onChange: (v: string[]) => void;
  hint?: string;
}) {
  const filtered = columns.filter((c) => {
    if (!scope || scope === "any") return true;
    const isNum = /int|float|number|decimal/i.test(c.dtype);
    if (scope === "numeric") return isNum;
    if (scope === "categorical") return !isNum;
    return true;
  });

  return (
    <div>
      <div className="flex flex-wrap gap-1.5">
        {filtered.length === 0 ? (
          <span className="text-xs text-slate-500">No matching columns.</span>
        ) : (
          filtered.map((c) => {
            const on = value.includes(c.name);
            return (
              <button
                key={c.name}
                type="button"
                onClick={() => {
                  if (on) onChange(value.filter((v) => v !== c.name));
                  else onChange([...value, c.name]);
                }}
                className={`rounded-full border px-2.5 py-0.5 text-xs ${
                  on
                    ? "border-studio-data bg-studio-data/15 text-studio-data"
                    : "border-bg-border bg-bg-card text-slate-300 hover:border-slate-500"
                }`}
                title={c.dtype}
              >
                {c.name}
              </button>
            );
          })
        )}
      </div>
      {hint && <p className="mt-1 text-xs text-slate-500">{hint}</p>}
    </div>
  );
}
