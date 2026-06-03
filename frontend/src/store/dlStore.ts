import { create } from "zustand";
import { dlApi } from "@/api/dl";
import type {
  ArchitectureInfo,
  DLCatalog,
  DLTrainResponse,
  LayerCatalogItem,
  LayerSpec,
  ModelSpec,
  ValidationReport,
} from "@/api/dl";

let _seq = 0;
const newId = () => `l${Date.now().toString(36)}_${_seq++}`;

function withIds(spec: ModelSpec): ModelSpec {
  return { ...spec, layers: spec.layers.map((l) => ({ ...l, _id: l._id ?? newId() })) };
}

function defaultsFor(item: LayerCatalogItem): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, s] of Object.entries(item.params)) out[k] = s.default ?? null;
  return out;
}

interface DLState {
  catalog: DLCatalog | null;
  architectures: ArchitectureInfo[];

  dataType: "tabular" | "image" | "text" | "timeseries";
  architecture: string | null;
  isRegression: boolean;

  spec: ModelSpec | null;
  validation: ValidationReport | null;
  code: string;
  validating: boolean;

  training: boolean;
  error: string | null;
  result: DLTrainResponse | null;

  fetchCatalog: () => Promise<void>;
  fetchArchitectures: (dataType: string) => Promise<void>;
  setDataType: (dt: DLState["dataType"]) => void;
  setRegression: (v: boolean) => void;
  scaffold: (architecture: string, inputShape: number[], nClasses: number | null) => Promise<void>;

  // layer editing
  addLayer: (kind: string) => void;
  removeLayer: (id: string) => void;
  moveLayer: (id: string, dir: -1 | 1) => void;
  updateLayerParam: (id: string, key: string, value: unknown) => void;

  // compile / training / api
  setApi: (api: "sequential" | "functional") => void;
  setCompile: <K extends keyof ModelSpec["compile"]>(k: K, v: ModelSpec["compile"][K]) => void;
  setTraining: <K extends keyof ModelSpec["training"]>(k: K, v: ModelSpec["training"][K]) => void;
  setInputShape: (shape: number[]) => void;
  setNClasses: (n: number | null) => void;

  revalidate: () => Promise<void>;
  train: (projectId: string, datasetId: string, targetCol: string | null) => Promise<void>;
  reset: () => void;
}

export const useDLStore = create<DLState>((set, get) => ({
  catalog: null,
  architectures: [],

  dataType: "tabular",
  architecture: null,
  isRegression: false,

  spec: null,
  validation: null,
  code: "",
  validating: false,

  training: false,
  error: null,
  result: null,

  fetchCatalog: async () => {
    try {
      set({ catalog: await dlApi.catalog() });
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  fetchArchitectures: async (dataType) => {
    try {
      set({ architectures: await dlApi.architectures(dataType) });
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  setDataType: (dt) => {
    set({ dataType: dt, architecture: null, spec: null, validation: null, code: "", result: null });
    void get().fetchArchitectures(dt);
  },
  setRegression: (v) => set({ isRegression: v }),

  scaffold: async (architecture, inputShape, nClasses) => {
    const { dataType, isRegression } = get();
    set({ error: null, architecture });
    try {
      const raw = await dlApi.scaffold({
        architecture,
        data_type: dataType,
        input_shape: inputShape,
        n_classes: nClasses,
        is_regression: isRegression,
      });
      set({ spec: withIds(raw), result: null });
      await get().revalidate();
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  addLayer: (kind) => {
    const { spec, catalog } = get();
    if (!spec || !catalog) return;
    const item = catalog.layers.find((l) => l.key === kind);
    if (!item) return;
    const layer: LayerSpec = { type: kind, params: defaultsFor(item), name: null, _id: newId() };
    // insert before the final (output) layer when there is one
    const layers = [...spec.layers];
    const insertAt = Math.max(0, layers.length - 1);
    layers.splice(insertAt, 0, layer);
    set({ spec: { ...spec, layers } });
    void get().revalidate();
  },

  removeLayer: (id) => {
    const { spec } = get();
    if (!spec) return;
    set({ spec: { ...spec, layers: spec.layers.filter((l) => l._id !== id) } });
    void get().revalidate();
  },

  moveLayer: (id, dir) => {
    const { spec } = get();
    if (!spec) return;
    const layers = [...spec.layers];
    const i = layers.findIndex((l) => l._id === id);
    const j = i + dir;
    if (i < 0 || j < 0 || j >= layers.length) return;
    [layers[i], layers[j]] = [layers[j], layers[i]];
    set({ spec: { ...spec, layers } });
    void get().revalidate();
  },

  updateLayerParam: (id, key, value) => {
    const { spec } = get();
    if (!spec) return;
    const layers = spec.layers.map((l) =>
      l._id === id ? { ...l, params: { ...l.params, [key]: value } } : l,
    );
    set({ spec: { ...spec, layers } });
    void get().revalidate();
  },

  setApi: (api) => {
    const { spec } = get();
    if (!spec) return;
    set({ spec: { ...spec, api } });
    void get().revalidate();
  },

  setCompile: (k, v) => {
    const { spec } = get();
    if (!spec) return;
    set({ spec: { ...spec, compile: { ...spec.compile, [k]: v } } });
    void get().revalidate();
  },

  setTraining: (k, v) => {
    const { spec } = get();
    if (!spec) return;
    set({ spec: { ...spec, training: { ...spec.training, [k]: v } } });
  },

  setInputShape: (shape) => {
    const { spec } = get();
    if (!spec) return;
    set({ spec: { ...spec, input_shape: shape } });
    void get().revalidate();
  },

  setNClasses: (n) => {
    const { spec } = get();
    if (!spec) return;
    set({ spec: { ...spec, n_classes: n } });
    void get().revalidate();
  },

  revalidate: async () => {
    const { spec } = get();
    if (!spec) return;
    set({ validating: true });
    try {
      const [validation, code] = await Promise.all([
        dlApi.validate(spec),
        dlApi.codegen(spec).then((r) => r.code),
      ]);
      set({ validation, code, validating: false });
    } catch (e) {
      set({ validating: false, error: (e as Error).message });
    }
  },

  train: async (projectId, datasetId, targetCol) => {
    const { spec } = get();
    if (!spec) {
      set({ error: "Scaffold or build a model first." });
      return;
    }
    set({ training: true, error: null, result: null });
    try {
      const r = await dlApi.train({
        project_id: projectId,
        dataset_id: datasetId,
        target_col: targetCol,
        spec,
      });
      set({ training: false, result: r, validation: r.validation ?? get().validation });
    } catch (e) {
      set({ training: false, error: (e as Error).message });
    }
  },

  reset: () =>
    set({ spec: null, validation: null, code: "", result: null, error: null, architecture: null }),
}));
