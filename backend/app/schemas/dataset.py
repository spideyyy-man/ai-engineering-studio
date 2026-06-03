from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DatasetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    name: str
    source_type: str
    storage_path: str
    n_rows: int
    n_cols: int
    size_bytes: int
    dtypes_json: dict[str, str] = {}
    created_at: datetime


class DatasetSummary(BaseModel):
    id: str
    name: str
    n_rows: int
    n_cols: int
    size_bytes: int
    sample: list[dict[str, Any]] = []
    columns: list[dict[str, Any]] = []


class ProfileResponse(BaseModel):
    """Full EDA report payload — Plotly figures returned as JSON dicts."""

    shape: dict[str, int]
    memory_usage_bytes: int
    dtypes: dict[str, str]
    missing: dict[str, dict[str, Any]]
    duplicates: int
    describe_numeric: dict[str, dict[str, float | None]]
    describe_categorical: dict[str, dict[str, Any]]
    correlations: dict[str, dict[str, float | None]] | None = None
    figures: dict[str, Any]  # name -> Plotly figure JSON


class QualityIssue(BaseModel):
    severity: str  # info | warn | error
    code: str
    column: str | None = None
    message: str
    suggestion: str | None = None


class QualityReport(BaseModel):
    issues: list[QualityIssue]
    summary: dict[str, int]   # counts by severity


class Recommendation(BaseModel):
    step: str        # scaling, encoding, imputation, balancing, drop, ...
    columns: list[str] = []
    rationale: str
    code_hint: str


class RecommendationsResponse(BaseModel):
    recommendations: list[Recommendation]


class CodegenRequest(BaseModel):
    target: str | None = Field(default=None, description="Target column if supervised")
    steps: list[Recommendation]
    framework: str = "sklearn"


class CodegenResponse(BaseModel):
    filename: str
    code: str
    download_url: str


# ---------- Transformations (apply operations to data) ----------

class TransformParamSpec(BaseModel):
    type: str                # columns | string | float | int | bool | mapping
    scope: str | None = None  # numeric | categorical | any
    default: Any = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    required: bool = False
    help: str | None = None


class TransformOperation(BaseModel):
    key: str
    label: str
    description: str
    fits: list[str] = []
    params: dict[str, TransformParamSpec] = {}


class TransformRequest(BaseModel):
    operation: str
    params: dict[str, Any] = {}
    inplace: bool = False
    new_name: str | None = None  # used when inplace=False


class TransformResponse(BaseModel):
    dataset_id: str
    inplace: bool
    summary: str
    n_rows_before: int
    n_rows_after: int
    n_cols_before: int
    n_cols_after: int
    new_dataset: DatasetRead | None = None
    stats_diff: dict[str, dict[str, dict[str, Any]]] = {}
    history_step: int | None = None


class SuggestedFixResponse(BaseModel):
    issue_code: str
    operations: list[TransformOperation]


class RecommendedFix(BaseModel):
    """A single recommended action for one (issue, column) pair."""
    issue_code: str
    column: str | None = None
    operation: str
    params: dict[str, Any] = {}
    rationale: str


class AutoFixStep(BaseModel):
    issue_code: str
    column: str | None
    operation: str | None
    status: str  # applied | skipped | failed
    summary: str | None = None
    rationale: str | None = None
    error: str | None = None


class AutoFixRequest(BaseModel):
    inplace: bool = False
    new_name: str | None = None


class AutoFixResponse(BaseModel):
    dataset_id: str
    inplace: bool
    steps: list[AutoFixStep]
    n_applied: int
    n_skipped: int
    n_failed: int
    n_rows_before: int
    n_rows_after: int
    n_cols_before: int
    n_cols_after: int
    new_dataset: DatasetRead | None = None
    # populated only by /clean-dataset (iterative)
    rounds_run: int = 1
    final_issues_count: int | None = None
    unfixable: list[str] = []


class CleanDatasetRequest(BaseModel):
    new_name: str | None = None
    target_col: str | None = None
    max_rounds: int = 5


# ---------- History / undo ----------

class HistoryEntry(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_id: str
    step: int
    operation: str
    params_json: dict[str, Any] = {}
    summary: str
    n_rows_before: int
    n_rows_after: int
    n_cols_before: int
    n_cols_after: int
    stats_diff_json: dict[str, Any] = {}
    created_at: datetime


class HistoryResponse(BaseModel):
    dataset_id: str
    entries: list[HistoryEntry]


class UndoResponse(BaseModel):
    dataset_id: str
    undone_step: int
    operation: str
    n_rows_after: int
    n_cols_after: int
    history_remaining: int


# ---------- Pipeline export ----------

class PipelineExportResponse(BaseModel):
    filename: str
    code: str


# ---------- Task detection ----------

class TaskDetectionResponse(BaseModel):
    task: str
    rationale: str
    signals: list[str] = []
    is_supervised: bool
    n_features: int
    n_samples: int
    ml_friendly: bool


# ---------- Baseline ----------

class BaselineResponse(BaseModel):
    task: str
    primary_metric: str
    dummy_score: float | None
    tree_score: float | None
    delta: float | None
    extra: dict[str, Any] = {}
    interpretation: str
