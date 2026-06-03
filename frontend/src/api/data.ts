import { api } from "./client";
import type {
  AutoFixResponse,
  BaselineResponse,
  CodegenResponse,
  Dataset,
  DatasetSummary,
  EmbeddingVizResponse,
  HistoryResponse,
  PipelineExportResponse,
  ProfileResponse,
  QualityReport,
  Recommendation,
  RecommendationsResponse,
  RecommendedFix,
  SuggestedFixResponse,
  TaskDetectionResponse,
  TransformOperation,
  TransformRequest,
  TransformResponse,
  UndoResponse,
} from "./types";

export const dataApi = {
  upload: async (projectId: string, file: File, name?: string) => {
    const fd = new FormData();
    fd.append("project_id", projectId);
    fd.append("file", file);
    if (name) fd.append("name", name);
    const r = await api.post<Dataset>("/data/upload", fd, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return r.data;
  },
  list: (projectId?: string) =>
    api
      .get<Dataset[]>("/data/datasets", {
        params: projectId ? { project_id: projectId } : {},
      })
      .then((r) => r.data),
  get: (id: string) => api.get<DatasetSummary>(`/data/datasets/${id}`).then((r) => r.data),
  profile: (id: string) =>
    api.get<ProfileResponse>(`/data/datasets/${id}/profile`).then((r) => r.data),
  quality: (id: string, targetCol?: string) =>
    api
      .get<QualityReport>(`/data/datasets/${id}/quality`, {
        params: targetCol ? { target_col: targetCol } : {},
      })
      .then((r) => r.data),
  recommendations: (id: string, targetCol?: string) =>
    api
      .get<RecommendationsResponse>(`/data/datasets/${id}/recommendations`, {
        params: targetCol ? { target_col: targetCol } : {},
      })
      .then((r) => r.data),
  codegen: (id: string, payload: { target?: string; steps: Recommendation[]; framework?: string }) =>
    api.post<CodegenResponse>(`/data/datasets/${id}/codegen`, payload).then((r) => r.data),
  remove: (id: string) => api.delete(`/data/datasets/${id}`).then((r) => r.data),

  // Transformations
  listTransforms: () =>
    api.get<TransformOperation[]>("/data/transformations").then((r) => r.data),
  fixesFor: (issueCode: string) =>
    api
      .get<SuggestedFixResponse>(`/data/transformations/for-issue/${issueCode}`)
      .then((r) => r.data),
  transform: (id: string, payload: TransformRequest) =>
    api
      .post<TransformResponse>(`/data/datasets/${id}/transform`, payload)
      .then((r) => r.data),

  recommendedFix: (id: string, issue_code: string, column: string | null) =>
    api
      .get<RecommendedFix | null>(`/data/datasets/${id}/recommended-fix`, {
        params: { issue_code, ...(column ? { column } : {}) },
      })
      .then((r) => r.data),

  autoFix: (id: string, inplace: boolean, target_col: string | null, new_name?: string) =>
    api
      .post<AutoFixResponse>(
        `/data/datasets/${id}/auto-fix`,
        { inplace, new_name },
        { params: target_col ? { target_col } : {} }
      )
      .then((r) => r.data),

  cleanDataset: (
    id: string,
    body: { new_name?: string; target_col?: string | null; max_rounds?: number }
  ) =>
    api
      .post<AutoFixResponse>(`/data/datasets/${id}/clean-dataset`, body, {
        timeout: 180_000,
      })
      .then((r) => r.data),

  // ---- History / Undo ----
  history: (id: string) =>
    api.get<HistoryResponse>(`/data/datasets/${id}/history`).then((r) => r.data),
  undo: (id: string) =>
    api.post<UndoResponse>(`/data/datasets/${id}/undo`).then((r) => r.data),

  // ---- Pipeline export ----
  pipelineExport: (id: string) =>
    api
      .get<PipelineExportResponse>(`/data/datasets/${id}/pipeline-export`)
      .then((r) => r.data),

  // ---- Task detection (data layer; ML studio has its own) ----
  detectTask: (id: string, target_col: string | null) =>
    api
      .get<TaskDetectionResponse>(`/data/datasets/${id}/detect-task`, {
        params: target_col ? { target_col } : {},
      })
      .then((r) => r.data),

  // ---- Baseline ----
  baseline: (id: string, target_col: string) =>
    api
      .get<BaselineResponse>(`/data/datasets/${id}/baseline`, {
        params: { target_col },
      })
      .then((r) => r.data),

  // ---- Embedding viz (t-SNE) ----
  embeddingViz: (id: string, color_col: string | null, sample_rows = 2000) =>
    api
      .get<EmbeddingVizResponse>(`/data/datasets/${id}/embedding-viz`, {
        params: {
          method: "tsne",
          sample_rows,
          ...(color_col ? { color_col } : {}),
        },
        timeout: 120_000,
      })
      .then((r) => r.data),
};
