"""Path helpers for per-project artifact directories."""
from __future__ import annotations

from pathlib import Path
from uuid import UUID

from ..config import get_settings


def project_dir(project_id: UUID | str) -> Path:
    """Returns per-project storage subdir; created if missing."""
    s = get_settings()
    p = s.storage_root / "projects" / str(project_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dataset_path(project_id: UUID | str, filename: str) -> Path:
    p = project_dir(project_id) / "datasets"
    p.mkdir(parents=True, exist_ok=True)
    return p / filename


def model_path(project_id: UUID | str, model_id: UUID | str) -> Path:
    p = project_dir(project_id) / "models" / str(model_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


def vectorstore_path(project_id: UUID | str, index_id: UUID | str) -> Path:
    p = project_dir(project_id) / "vectorstores" / str(index_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


def codegen_path(project_id: UUID | str, kind: str, name: str) -> Path:
    p = project_dir(project_id) / "code" / kind
    p.mkdir(parents=True, exist_ok=True)
    return p / name
