import { api } from "./client";
import type { HyperparamSpec } from "./ml";

// ---------------------------------------------------------------- catalog types

export interface LayerCatalogItem {
  key: string;
  label: string;
  category: string;
  data_types: string[];
  description: string;
  params: Record<string, HyperparamSpec>;
}

export interface OptimizerInfo {
  key: string;
  label: string;
  keras: string;
  description: string;
}

export interface DLCatalog {
  layers: LayerCatalogItem[];
  optimizers: OptimizerInfo[];
  losses_by_task: Record<string, { key: string; keras: string; label: string }[]>;
  metrics_by_task: Record<string, string[]>;
  tf_available: boolean;
}

export interface ArchitectureInfo {
  key: string;
  label: string;
  data_type: "tabular" | "image" | "text" | "timeseries";
  api: "sequential" | "functional";
  description: string;
  trainable_here: boolean;
}

// ---------------------------------------------------------------- model spec

export interface LayerSpec {
  type: string;
  params: Record<string, unknown>;
  name?: string | null;
  // client-only id for stable list keys (stripped before sending)
  _id?: string;
}

export interface CompileSpec {
  optimizer: string;
  learning_rate: number;
  loss: string;
  metrics: string[];
}

export interface TrainingSpec {
  epochs: number;
  batch_size: number;
  validation_split: number;
  early_stopping: boolean;
  reduce_lr_on_plateau: boolean;
}

export interface ModelSpec {
  api: "sequential" | "functional";
  task: string;
  data_type: string;
  architecture?: string | null;
  input_shape: number[];
  n_classes?: number | null;
  layers: LayerSpec[];
  compile: CompileSpec;
  training: TrainingSpec;
}

// ---------------------------------------------------------------- validation

export interface LayerShapeRow {
  index: number;
  type: string;
  label: string;
  input_shape: number[] | null;
  output_shape: number[] | null;
  params: number;
  error: string | null;
}

export interface ValidationReport {
  ok: boolean;
  api: string;
  input_shape: number[];
  output_shape: number[] | null;
  total_params: number;
  total_params_human: string;
  layers: LayerShapeRow[];
  errors: string[];
  warnings: string[];
}

// ---------------------------------------------------------------- train

export interface DLTrainResponse {
  model_id: string;
  task: string;
  api: string;
  architecture?: string | null;
  tf_available: boolean;
  trained: boolean;
  message?: string | null;
  metrics: Record<string, unknown>;
  figures: Record<string, unknown>;
  history: Record<string, number[]>;
  validation?: ValidationReport | null;
  n_train: number;
  n_test: number;
  n_features?: number | null;
  n_classes?: number | null;
  epochs_ran: number;
  feature_columns: string[];
  target_column?: string | null;
  class_names?: string[] | null;
  code: string;
  code_filename: string;
  model_download_url?: string | null;
  code_download_url?: string | null;
}

// strip client-only fields before sending the spec to the backend
function cleanSpec(spec: ModelSpec): ModelSpec {
  return {
    ...spec,
    layers: spec.layers.map(({ _id, ...rest }) => rest),
  };
}

export const dlApi = {
  architectures: (data_type?: string) =>
    api
      .get<ArchitectureInfo[]>("/dl/architectures", { params: data_type ? { data_type } : {} })
      .then((r) => r.data),
  catalog: () => api.get<DLCatalog>("/dl/catalog").then((r) => r.data),
  scaffold: (body: {
    architecture: string;
    data_type: string;
    input_shape: number[];
    n_classes?: number | null;
    is_regression?: boolean;
  }) => api.post<ModelSpec>("/dl/scaffold", body).then((r) => r.data),
  validate: (spec: ModelSpec) =>
    api.post<ValidationReport>("/dl/validate", { spec: cleanSpec(spec) }).then((r) => r.data),
  codegen: (spec: ModelSpec) =>
    api
      .post<{ filename: string; code: string }>("/dl/codegen", { spec: cleanSpec(spec) })
      .then((r) => r.data),
  train: (body: {
    project_id: string;
    dataset_id: string;
    target_col: string | null;
    spec: ModelSpec;
    random_state?: number;
  }) =>
    api
      .post<DLTrainResponse>(
        "/dl/train",
        { ...body, spec: cleanSpec(body.spec) },
        { timeout: 600_000 }, // training can be slow — allow up to 10 min
      )
      .then((r) => r.data),
  models: (project_id?: string) =>
    api.get("/dl/models", { params: project_id ? { project_id } : {} }).then((r) => r.data),
};
