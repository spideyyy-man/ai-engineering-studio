import { create } from "zustand";
import type {
  Dataset,
  DatasetSummary,
  ProfileResponse,
  QualityReport,
  Recommendation,
  RecommendationsResponse,
} from "@/api/types";
import { dataApi } from "@/api";

interface DataState {
  datasets: Dataset[];
  currentDatasetId: string | null;
  summary: DatasetSummary | null;
  profile: ProfileResponse | null;
  quality: QualityReport | null;
  recommendations: Recommendation[];
  targetCol: string | null;

  loading: boolean;
  error: string | null;

  setTarget: (col: string | null) => void;
  setCurrent: (id: string | null) => void;
  fetchDatasets: (projectId: string) => Promise<void>;
  upload: (projectId: string, file: File) => Promise<Dataset>;
  loadAll: (datasetId: string, targetCol?: string) => Promise<void>;
  remove: (id: string) => Promise<void>;
  reset: () => void;
}

export const useDataStore = create<DataState>((set, get) => ({
  datasets: [],
  currentDatasetId: null,
  summary: null,
  profile: null,
  quality: null,
  recommendations: [],
  targetCol: null,

  loading: false,
  error: null,

  setTarget: (col) => set({ targetCol: col }),

  setCurrent: (id) => set({ currentDatasetId: id }),

  fetchDatasets: async (projectId) => {
    set({ loading: true, error: null });
    try {
      const datasets = await dataApi.list(projectId);
      set({
        datasets,
        loading: false,
        currentDatasetId: get().currentDatasetId ?? datasets[0]?.id ?? null,
      });
    } catch (e) {
      set({ loading: false, error: (e as Error).message });
    }
  },

  upload: async (projectId, file) => {
    set({ loading: true, error: null });
    try {
      const ds = await dataApi.upload(projectId, file);
      set((s) => ({
        datasets: [ds, ...s.datasets],
        currentDatasetId: ds.id,
        loading: false,
      }));
      return ds;
    } catch (e) {
      set({ loading: false, error: (e as Error).message });
      throw e;
    }
  },

  loadAll: async (datasetId, targetCol) => {
    set({ loading: true, error: null });
    try {
      const [summary, profile, quality, recsResp]: [
        DatasetSummary,
        ProfileResponse,
        QualityReport,
        RecommendationsResponse
      ] = await Promise.all([
        dataApi.get(datasetId),
        dataApi.profile(datasetId),
        dataApi.quality(datasetId, targetCol),
        dataApi.recommendations(datasetId, targetCol),
      ]);
      set({
        summary,
        profile,
        quality,
        recommendations: recsResp.recommendations,
        loading: false,
      });
    } catch (e) {
      set({ loading: false, error: (e as Error).message });
    }
  },

  remove: async (id) => {
    await dataApi.remove(id);
    set((s) => ({
      datasets: s.datasets.filter((d) => d.id !== id),
      currentDatasetId: s.currentDatasetId === id ? null : s.currentDatasetId,
      summary: s.currentDatasetId === id ? null : s.summary,
      profile: s.currentDatasetId === id ? null : s.profile,
      quality: s.currentDatasetId === id ? null : s.quality,
      recommendations: s.currentDatasetId === id ? [] : s.recommendations,
    }));
  },

  reset: () =>
    set({
      summary: null,
      profile: null,
      quality: null,
      recommendations: [],
      targetCol: null,
    }),
}));
