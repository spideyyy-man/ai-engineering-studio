from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------- model spec

class LayerSpec(BaseModel):
    type: str                              # dense, conv2d, lstm, ...
    params: dict[str, Any] = {}
    name: str | None = None


class CompileSpec(BaseModel):
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    loss: str = "mse"
    metrics: list[str] = []


class TrainingSpec(BaseModel):
    epochs: int = 20
    batch_size: int = 32
    validation_split: float = 0.2
    early_stopping: bool = True
    reduce_lr_on_plateau: bool = True


class ModelSpec(BaseModel):
    api: str = "sequential"                # sequential | functional
    task: str = "tabular_classification"   # tabular_classification|tabular_regression|image_classification|text_classification|timeseries_forecast
    data_type: str = "tabular"             # tabular|image|text|timeseries
    architecture: str | None = None
    input_shape: list[int] = []
    n_classes: int | None = None
    layers: list[LayerSpec] = []
    compile: CompileSpec = Field(default_factory=CompileSpec)
    training: TrainingSpec = Field(default_factory=TrainingSpec)


# ---------------------------------------------------------------- requests

class ScaffoldRequest(BaseModel):
    architecture: str
    data_type: str
    input_shape: list[int]
    n_classes: int | None = None
    is_regression: bool = False


class ValidateRequest(BaseModel):
    spec: ModelSpec


class CodegenRequest(BaseModel):
    spec: ModelSpec


class DLTrainRequest(BaseModel):
    project_id: str
    dataset_id: str
    target_col: str | None = None
    spec: ModelSpec
    random_state: int = 42


# legacy shape kept so old imports don't break
class DLBuildRequest(BaseModel):
    architecture: str
    data_type: str
    input_shape: list[int]
    config: dict[str, Any] = {}


# ---------------------------------------------------------------- responses

class LayerShapeRow(BaseModel):
    index: int
    type: str
    label: str
    input_shape: list[int] | None = None
    output_shape: list[int] | None = None
    params: int = 0
    error: str | None = None


class ValidationReport(BaseModel):
    ok: bool
    api: str
    input_shape: list[int]
    output_shape: list[int] | None = None
    total_params: int
    total_params_human: str
    layers: list[LayerShapeRow]
    errors: list[str]
    warnings: list[str]


class CodegenResponse(BaseModel):
    filename: str
    code: str


class DLTrainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    task: str
    api: str
    architecture: str | None = None
    tf_available: bool
    trained: bool
    message: str | None = None
    metrics: dict[str, Any] = {}
    figures: dict[str, Any] = {}
    history: dict[str, Any] = {}
    validation: ValidationReport | None = None
    n_train: int = 0
    n_test: int = 0
    n_features: int | None = None
    n_classes: int | None = None
    epochs_ran: int = 0
    feature_columns: list[str] = []
    target_column: str | None = None
    class_names: list[str] | None = None
    code: str = ""
    code_filename: str = "model.py"
    model_download_url: str | None = None
    code_download_url: str | None = None


class DLRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    architecture: str
    data_type: str
    epochs: int
    metrics_json: dict[str, Any] = {}
    history_json: dict[str, Any] = {}
    artifact_path: str | None = None
    code_path: str | None = None
    status: str
    created_at: datetime
