from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class HyperparamSpec(BaseModel):
    type: str
    default: Any | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: list[Any] | None = None
    label: str | None = None
    fmt: str | None = None


class AlgorithmInfo(BaseModel):
    key: str
    label: str
    task: str
    description: str
    params: dict[str, HyperparamSpec] = {}
    fixed_params: dict[str, Any] = {}
    needs_scaling: bool = True


class DetectTaskRequest(BaseModel):
    dataset_id: str
    target_col: str | None = None


class DetectTaskResponse(BaseModel):
    task_type: str
    rationale: str


class MLTrainRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    project_id: str
    dataset_id: str
    target_col: str | None = None
    algorithm: str
    hyperparams: dict[str, Any] = {}
    test_size: float = 0.2
    random_state: int = 42

    # Slice A: CV + HPT
    cv_strategy: str | None = None        # none | kfold | stratified_kfold | time_series | group_kfold | leave_one_out
    cv_folds: int = 5
    hpt_mode: str = "manual"               # manual | grid | random
    hpt_n_iter: int = 20                   # for RandomizedSearchCV
    hpt_scoring: str | None = None         # override; defaults by task


class HPTResultRow(BaseModel):
    rank: int
    mean_score: float | None = None
    std_score: float | None = None
    params: dict[str, Any] = {}


class MLTrainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    task: str
    algorithm: str
    algorithm_label: str
    metrics: dict[str, Any]
    figures: dict[str, Any]
    hparams_effective: dict[str, Any]
    feature_columns: list[str]
    target_column: str | None
    n_train: int
    n_test: int
    pickle_filename: str
    pickle_download_url: str
    sample_filename: str
    sample_code: str
    sample_download_url: str

    # Auto preprocessing summary — what got baked into the pickle
    preprocessing_applied: dict[str, Any] | None = None

    # CV / HPT extras (supervised only)
    cv_used: bool = False
    cv_strategy: str | None = None
    cv_folds: int | None = None
    cv_mean_score: float | None = None
    cv_std_score: float | None = None
    cv_scoring: str | None = None
    hpt_mode: str | None = None
    hpt_best_params: dict[str, Any] | None = None
    hpt_best_score: float | None = None
    hpt_cv_results: list[HPTResultRow] | None = None


class MLRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: str
    project_id: str
    task_type: str
    algorithm: str
    metrics_json: dict[str, Any] = {}
    artifact_path: str | None = None
    code_path: str | None = None
    status: str
    created_at: datetime
