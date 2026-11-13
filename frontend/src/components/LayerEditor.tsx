import { useMemo, useState } from "react";
import { ChevronDown, ChevronUp, Plus, Trash2, AlertTriangle } from "lucide-react";

import HyperparamForm from "@/components/HyperparamForm";
import { useDLStore } from "@/store/dlStore";
import type { LayerCatalogItem, LayerShapeRow } from "@/api/dl";

const CATEGORY_ORDER = ["core", "conv", "recurrent", "embedding", "attention", "regularize", "shape"];
const CATEGORY_LABEL: Record<string, string> = {
  core: "Core",
  conv: "Convolution / Pooling",
  recurrent: "Recurrent",
  embedding: "Embedding",
  attention: "Attention",
  regularize: "Normalization / Dropout",
  shape: "Shape ops",
};

export default function LayerEditor() {
  const spec = useDLStore((s) => s.spec);
  const validation = useDLStore((s) => s.validation);
  const catalog = useDLStore((s) => s.catalog);
  const dataType = useDLStore((s) => s.dataType);

  const addLayer = useDLStore((s) => s.addLayer);
  const removeLayer = useDLStore((s) => s.removeLayer);
  const moveLayer = useDLStore((s) => s.moveLayer);
  const updateLayerParam = useDLStore((s) => s.updateLayerParam);

  const [openId, setOpenId] = useState<string | null>(null);
  const [paletteOpen, setPaletteOpen] = useState(false);

  const rowByIndex = useMemo(() => {
    const m = new Map<number, LayerShapeRow>();
    validation?.layers.forEach((r) => m.set(r.index, r));
    return m;
  }, [validation]);

  const palette = useMemo(() => {
    const items = (catalog?.layers ?? []).filter((l) => l.data_types.includes(dataType));
    const groups: Record<string, LayerCatalogItem[]> = {};
    for (const it of items) (groups[it.category] ??= []).push(it);
    return groups;
  }, [catalog, dataType]);

  if (!spec) return null;

  return (
    <div className="space-y-3">
      {/* Input chip */}
      <div className="flex items-center gap-2 rounded-lg border border-dashed border-bg-border bg-bg-panel/40 px-3 py-2 text-xs">
        <span className="font-mono text-slate-500">input</span>
        <span className="font-mono text-studio-dl">({spec.input_shape.join(", ")})</span>
        <span className="text-slate-500">— fed into layer 1</span>
      </div>

      {/* Layers */}
      <ul className="space-y-2">
        {spec.layers.map((layer, i) => {
          const row = rowByIndex.get(i);
          const open = openId === layer._id;
          const catItem = catalog?.layers.find((l) => l.key === layer.type);
          const hasErr = Boolean(row?.error);
          const isLast = i === spec.layers.length - 1;
          return (
            <li
              key={layer._id}
              className={`rounded-lg border ${
                hasErr ? "border-red-700/50 bg-red-900/10" : "border-bg-border bg-bg-panel"
              }`}
            >
              <div className="flex items-center gap-2 px-3 py-2">
                <span className="grid h-6 w-6 shrink-0 place-items-center rounded bg-bg-card text-[11px] font-mono text-slate-400">
                  {i + 1}
                </span>
                <button
                  className="flex min-w-0 flex-1 items-center gap-2 text-left"
                  onClick={() => setOpenId(open ? null : layer._id ?? null)}
                >
                  <span className="font-medium text-slate-100">{catItem?.label ?? layer.type}</span>
                  {isLast && <span className="pill">output</span>}
                  {row?.output_shape && (
                    <span className="font-mono text-xs text-studio-dl">
                      → ({row.output_shape.join(", ")})
                    </span>
                  )}
                  {row && row.params > 0 && (
                    <span className="ml-auto hidden font-mono text-[10px] text-slate-500 sm:inline">
                      {row.params.toLocaleString()} params
                    </span>
                  )}
                </button>

                {/* reorder + delete */}
                <div className="flex items-center gap-0.5">
                  <button
                    className="rounded p-1 text-slate-500 hover:bg-bg-card hover:text-slate-200 disabled:opacity-30"
                    disabled={i === 0}
                    title="Move up"
                    onClick={() => moveLayer(layer._id!, -1)}
                  >
                    <ChevronUp size={15} />
                  </button>
                  <button
                    className="rounded p-1 text-slate-500 hover:bg-bg-card hover:text-slate-200 disabled:opacity-30"
                    disabled={i === spec.layers.length - 1}
                    title="Move down"
                    onClick={() => moveLayer(layer._id!, 1)}
                  >
                    <ChevronDown size={15} />
                  </button>
                  <button
                    className="rounded p-1 text-slate-500 hover:bg-red-900/40 hover:text-red-300"
                    title="Delete layer"
                    onClick={() => removeLayer(layer._id!)}
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              </div>

              {hasErr && (
                <div className="flex items-start gap-2 border-t border-red-700/30 px-3 py-2 text-xs text-red-200">
                  <AlertTriangle size={14} className="mt-0.5 shrink-0" />
                  <span>{row?.error}</span>
                </div>
              )}

              {open && catItem && (
                <div className="border-t border-bg-border px-3 py-3">
                  <p className="mb-2 text-xs text-slate-400">{catItem.description}</p>
                  {Object.keys(catItem.params).length === 0 ? (
                    <p className="text-xs text-slate-500">No tunable parameters.</p>
                  ) : (
                    <HyperparamForm
                      spec={catItem.params}
                      values={layer.params}
                      onChange={(k, v) => updateLayerParam(layer._id!, k, v)}
                    />
                  )}
                </div>
              )}
            </li>
          );
        })}
      </ul>

      {/* Add layer */}
      <div className="rounded-lg border border-dashed border-bg-border bg-bg-panel/40">
        <button
          className="flex w-full items-center gap-2 px-3 py-2 text-sm text-slate-300 hover:text-studio-dl"
          onClick={() => setPaletteOpen((o) => !o)}
        >
          <Plus size={15} /> Add a layer
          {paletteOpen ? <ChevronUp size={14} className="ml-auto" /> : <ChevronDown size={14} className="ml-auto" />}
        </button>
        {paletteOpen && (
          <div className="space-y-3 border-t border-bg-border px-3 py-3">
            {CATEGORY_ORDER.filter((c) => palette[c]?.length).map((cat) => (
              <div key={cat}>
                <div className="mb-1 text-[10px] font-semibold text-slate-500">
                  {CATEGORY_LABEL[cat] ?? cat}
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {palette[cat].map((it) => (
                    <button
                      key={it.key}
                      title={it.description}
                      className="rounded-md border border-bg-border bg-bg-card px-2 py-1 text-xs text-slate-200 hover:border-studio-dl hover:text-studio-dl"
                      onClick={() => addLayer(it.key)}
                    >
                      {it.label}
                    </button>
                  ))}
                </div>
              </div>
            ))}
            <p className="text-[11px] text-slate-500">
              New layers are inserted just before the output layer. Reorder with the arrows.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
