import { api } from "./client";

export interface HyperparamSpec {
  type: "int" | "int_or_none" | "float" | "select" | "csv_int" | "bool" | string;
  default: unknown;
  min?: number;
  max?: number;
  step?: number;
  options?: unknown[];
  label?: string;
  fmt?: string;
}

export interface AlgorithmInfo {
  key: string;
  label: string;
  task: "classification" | "regression" | "clustering";
  description: string;
  params: Record<string, HyperparamSpec>;
  fixed_params: Record<string, unknown>;
  needs_scaling?: boolean;
}

export interface PreprocessingStep {
  kind: "impute_numeric" | "scale_numeric" | "impute_categorical" | "encode_categorical" | string;
  applied: boolean;
  method: string;
  rationale: string;
}

export interface PreprocessingApplied {
  numeric_columns: number;
  categorical_columns: number;
  steps: PreprocessingStep[];
}

export interface DetectTaskResponse {
  task_type: "classification" | "regression" | "clustering" | "unknown";
  rationale: string;
}

export type CVStrategy =
  | "none"
  | "kfold"
  | "stratified_kfold"
  | "time_series"
  | "group_kfold"
  | "leave_one_out";

export type HPTMode = "manual" | "grid" | "random";

export interface MLTrainRequest {
  project_id: string;
  dataset_id: string;
  target_col?: string | null;
  algorithm: string;
  hyperparams?: Record<string, unknown>;
  test_size?: number;
  random_state?: number;
  // Slice A
  cv_strategy?: CVStrategy | null;
  cv_folds?: number;
  hpt_mode?: HPTMode;
  hpt_n_iter?: number;
  hpt_scoring?: string | null;
}

export interface HPTResultRow {
  rank: number;
  mean_score: number | null;
  std_score: number | null;
  params: Record<string, unknown>;
}

export interface MLTrainResponse {
  model_id: string;
  task: "classification" | "regression" | "clustering";
  algorithm: string;
  algorithm_label: string;
  metrics: Record<string, unknown>;
  figures: Record<string, unknown>;
  hparams_effective: Record<string, unknown>;
  feature_columns: string[];
  target_column: string | null;
  n_train: number;
  n_test: number;
  pickle_filename: string;
  pickle_download_url: string;
  sample_filename: string;
  sample_code: string;
  sample_download_url: string;
  preprocessing_applied?: PreprocessingApplied | null;
  // Slice A
  cv_used?: boolean;
  cv_strategy?: string | null;
  cv_folds?: number | null;
  cv_mean_score?: number | null;
  cv_std_score?: number | null;
  cv_scoring?: string | null;
  hpt_mode?: string | null;
  hpt_best_params?: Record<string, unknown> | null;
  hpt_best_score?: number | null;
  hpt_cv_results?: HPTResultRow[] | null;
}

export const mlApi = {
  algorithms: (task?: string) =>
    api
      .get<AlgorithmInfo[]>("/ml/algorithms", { params: task ? { task } : {} })
      .then((r) => r.data),
  detectTask: (dataset_id: string, target_col: string | null) =>
    api
      .post<DetectTaskResponse>("/ml/detect-task", { dataset_id, target_col })
      .then((r) => r.data),
  train: (payload: MLTrainRequest) =>
    api.post<MLTrainResponse>("/ml/train", payload).then((r) => r.data),
  models: (project_id?: string) =>
    api
      .get("/ml/models", { params: project_id ? { project_id } : {} })
      .then((r) => r.data),
};
