// Mirror of backend Pydantic schemas — keep in sync.

export interface Project {
  id: string;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
}

export interface Dataset {
  id: string;
  project_id: string;
  name: string;
  source_type: string;
  storage_path: string;
  n_rows: number;
  n_cols: number;
  size_bytes: number;
  dtypes_json: Record<string, string>;
  created_at: string;
}

export interface DatasetSummary {
  id: string;
  name: string;
  n_rows: number;
  n_cols: number;
  size_bytes: number;
  sample: Record<string, unknown>[];
  columns: { name: string; dtype: string }[];
}

export interface ProfileResponse {
  shape: { rows: number; cols: number };
  memory_usage_bytes: number;
  dtypes: Record<string, string>;
  missing: Record<string, { count: number; pct: number }>;
  duplicates: number;
  describe_numeric: Record<string, Record<string, number | null>>;
  describe_categorical: Record<string, { unique: number; top: string | null; freq: number; top_values: Record<string, number> }>;
  correlations: Record<string, Record<string, number | null>> | null;
  figures: Record<string, unknown>;
}

export interface QualityIssue {
  severity: "info" | "warn" | "error";
  code: string;
  column: string | null;
  message: string;
  suggestion: string | null;
}

export interface QualityReport {
  issues: QualityIssue[];
  summary: Record<string, number>;
}

export interface Recommendation {
  step: string;
  columns: string[];
  rationale: string;
  code_hint: string;
}

export interface RecommendationsResponse {
  recommendations: Recommendation[];
}

export interface CodegenResponse {
  filename: string;
  code: string;
  download_url: string;
}

export interface TransformParamSpec {
  type: "columns" | "string" | "float" | "int" | "bool" | "mapping" | string;
  scope?: "numeric" | "categorical" | "any" | string;
  default?: unknown;
  min?: number;
  max?: number;
  step?: number;
  required?: boolean;
  help?: string;
}

export interface TransformOperation {
  key: string;
  label: string;
  description: string;
  fits: string[];
  params: Record<string, TransformParamSpec>;
}

export interface SuggestedFixResponse {
  issue_code: string;
  operations: TransformOperation[];
}

export interface TransformRequest {
  operation: string;
  params: Record<string, unknown>;
  inplace: boolean;
  new_name?: string;
}

export interface ColumnStat {
  present: boolean;
  dtype?: string;
  n?: number;
  missing?: number;
  nunique?: number;
  mean?: number | null;
  median?: number | null;
  std?: number | null;
  min?: number | null;
  max?: number | null;
  top_value?: string | null;
  top_freq?: number;
}

export type StatsDiff = Record<string, { before: ColumnStat; after: ColumnStat }>;

export interface TransformResponse {
  dataset_id: string;
  inplace: boolean;
  summary: string;
  n_rows_before: number;
  n_rows_after: number;
  n_cols_before: number;
  n_cols_after: number;
  new_dataset: Dataset | null;
  stats_diff: StatsDiff;
  history_step: number | null;
}

export interface RecommendedFix {
  issue_code: string;
  column: string | null;
  operation: string;
  params: Record<string, unknown>;
  rationale: string;
}

export interface AutoFixStep {
  issue_code: string;
  column: string | null;
  operation: string | null;
  status: "applied" | "skipped" | "failed";
  summary: string | null;
  rationale: string | null;
  error: string | null;
}

export interface AutoFixResponse {
  dataset_id: string;
  inplace: boolean;
  steps: AutoFixStep[];
  n_applied: number;
  n_skipped: number;
  n_failed: number;
  n_rows_before: number;
  n_rows_after: number;
  n_cols_before: number;
  n_cols_after: number;
  new_dataset: Dataset | null;
  rounds_run?: number;
  final_issues_count?: number | null;
  unfixable?: string[];
}

export interface CleanDatasetRequest {
  new_name?: string;
  target_col?: string | null;
  max_rounds?: number;
}

// ---------- History / Undo ----------

export interface HistoryEntry {
  id: string;
  dataset_id: string;
  step: number;
  operation: string;
  params_json: Record<string, unknown>;
  summary: string;
  n_rows_before: number;
  n_rows_after: number;
  n_cols_before: number;
  n_cols_after: number;
  stats_diff_json: StatsDiff;
  created_at: string;
}

export interface HistoryResponse {
  dataset_id: string;
  entries: HistoryEntry[];
}

export interface UndoResponse {
  dataset_id: string;
  undone_step: number;
  operation: string;
  n_rows_after: number;
  n_cols_after: number;
  history_remaining: number;
}

// ---------- Pipeline export ----------

export interface PipelineExportResponse {
  filename: string;
  code: string;
}

// ---------- Task detection ----------

export type TaskType =
  | "classification"
  | "regression"
  | "time_series_regression"
  | "time_series"
  | "rag_corpus"
  | "nlp"
  | "clustering"
  | "unknown";

export interface TaskDetectionResponse {
  task: TaskType;
  rationale: string;
  signals: string[];
  is_supervised: boolean;
  n_features: number;
  n_samples: number;
  ml_friendly: boolean;
}

// ---------- Baseline ----------

export interface BaselineResponse {
  task: "classification" | "regression";
  primary_metric: string;
  dummy_score: number | null;
  tree_score: number | null;
  delta: number | null;
  extra: Record<string, unknown>;
  interpretation: string;
}

// ---------- Embedding viz ----------

export interface EmbeddingVizResponse {
  method: string;
  n_points: number;
  perplexity: number;
  figure: unknown;
}

export interface AppSettings {
  llm_default_provider: string;
  llm_default_model: string;
  embedding_default_provider: string;
  embedding_default_model: string;
  vector_store_default: string;
  ollama_base_url: string;
  openai_configured: boolean;
  anthropic_configured: boolean;
}

export interface ProviderInfo {
  name: string;
  configured: boolean;
  default_model: string;
}
