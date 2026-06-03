import { create } from "zustand";
import { mlApi } from "@/api/ml";
import type {
  AlgorithmInfo,
  CVStrategy,
  DetectTaskResponse,
  HPTMode,
  MLTrainResponse,
} from "@/api/ml";

interface MLState {
  algorithms: AlgorithmInfo[];
  loadingAlgorithms: boolean;

  task: "classification" | "regression" | "clustering" | "unknown";
  taskRationale: string;

  algoKey: string | null;
  hparams: Record<string, unknown>;

  testSize: number;
  randomState: number;

  // Slice A — CV + HPT controls
  cvStrategy: CVStrategy;
  cvFolds: number;
  hptMode: HPTMode;
  hptNIter: number;

  training: boolean;
  error: string | null;
  result: MLTrainResponse | null;

  fetchAlgorithms: () => Promise<void>;
  detectTask: (datasetId: string, targetCol: string | null) => Promise<void>;
  selectAlgo: (key: string | null) => void;
  setHparam: (k: string, v: unknown) => void;
  setTestSize: (v: number) => void;
  setRandomState: (v: number) => void;
  setCVStrategy: (v: CVStrategy) => void;
  setCVFolds: (v: number) => void;
  setHPTMode: (v: HPTMode) => void;
  setHPTNIter: (v: number) => void;
  train: (datasetId: string, projectId: string, targetCol: string | null) => Promise<void>;
  reset: () => void;
}

function defaultsFor(algo: AlgorithmInfo | undefined): Record<string, unknown> {
  if (!algo) return {};
  const out: Record<string, unknown> = {};
  for (const [k, spec] of Object.entries(algo.params)) {
    out[k] = spec.default ?? null;
  }
  return out;
}

export const useMLStore = create<MLState>((set, get) => ({
  algorithms: [],
  loadingAlgorithms: false,

  task: "unknown",
  taskRationale: "",

  algoKey: null,
  hparams: {},

  testSize: 0.2,
  randomState: 42,

  cvStrategy: "none",
  cvFolds: 5,
  hptMode: "manual",
  hptNIter: 20,

  training: false,
  error: null,
  result: null,

  fetchAlgorithms: async () => {
    set({ loadingAlgorithms: true });
    try {
      const algos = await mlApi.algorithms();
      set({ algorithms: algos, loadingAlgorithms: false });
    } catch (e) {
      set({ loadingAlgorithms: false, error: (e as Error).message });
    }
  },

  detectTask: async (datasetId, targetCol) => {
    set({ error: null });
    try {
      const r: DetectTaskResponse = await mlApi.detectTask(datasetId, targetCol);
      set({ task: r.task_type, taskRationale: r.rationale });
      // If currently-selected algorithm doesn't match the new task, clear it
      const { algoKey, algorithms } = get();
      if (algoKey) {
        const cur = algorithms.find((a) => a.key === algoKey);
        if (cur && cur.task !== r.task_type) {
          set({ algoKey: null, hparams: {} });
        }
      }
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  selectAlgo: (key) => {
    const { algorithms } = get();
    const algo = algorithms.find((a) => a.key === key);
    set({ algoKey: key, hparams: defaultsFor(algo), result: null, error: null });
  },

  setHparam: (k, v) => set((s) => ({ hparams: { ...s.hparams, [k]: v } })),
  setTestSize: (v) => set({ testSize: v }),
  setRandomState: (v) => set({ randomState: v }),
  setCVStrategy: (v) => set({ cvStrategy: v }),
  setCVFolds: (v) => set({ cvFolds: v }),
  setHPTMode: (v) => set({ hptMode: v }),
  setHPTNIter: (v) => set({ hptNIter: v }),

  train: async (datasetId, projectId, targetCol) => {
    const { algoKey, hparams, testSize, randomState, task,
            cvStrategy, cvFolds, hptMode, hptNIter } = get();
    if (!algoKey) {
      set({ error: "Pick an algorithm first." });
      return;
    }
    set({ training: true, error: null, result: null });
    try {
      const r = await mlApi.train({
        project_id: projectId,
        dataset_id: datasetId,
        target_col: task === "clustering" ? null : targetCol,
        algorithm: algoKey,
        hyperparams: hparams,
        test_size: testSize,
        random_state: randomState,
        cv_strategy: cvStrategy === "none" ? null : cvStrategy,
        cv_folds: cvFolds,
        hpt_mode: hptMode,
        hpt_n_iter: hptNIter,
      });
      set({ training: false, result: r });
    } catch (e) {
      set({ training: false, error: (e as Error).message });
    }
  },

  reset: () => set({ result: null, error: null, hparams: {}, algoKey: null }),
}));
