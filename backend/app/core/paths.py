"""Path helpers for per-project artifact directories."""
from __future__ import annotations

import re
from pathlib import Path
from uuid import UUID

from ..config import get_settings
from .errors import ValidationError


_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._ -]+")


def safe_filename(name: str, default: str = "file") -> str:
    """Reduce a client-supplied filename to a safe bare name.

    Drops any directory part (both / and \\ separators), replaces unusual
    characters with "_", and strips leading dots so the result can never
    point outside the folder it is joined to.
    """
    base = (name or "").replace("\\", "/").split("/")[-1]
    base = _UNSAFE_CHARS.sub("_", base).strip().lstrip(".")
    return base[:150] or default


def _inside(child: Path, parent: Path) -> Path:
    """Return ``child`` if it resolves inside ``parent``, else raise."""
    resolved = child.resolve()
    if not resolved.is_relative_to(parent.resolve()):
        raise ValidationError(f"refusing to use a path outside the project folder: {child.name}")
    return child


def project_dir(project_id: UUID | str) -> Path:
    """Returns per-project storage subdir; created if missing."""
    s = get_settings()
    p = s.storage_root / "projects" / str(project_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dataset_path(project_id: UUID | str, filename: str) -> Path:
    p = project_dir(project_id) / "datasets"
    p.mkdir(parents=True, exist_ok=True)
    return _inside(p / filename, p)


def model_path(project_id: UUID | str, model_id: UUID | str) -> Path:
    p = project_dir(project_id) / "models" / str(model_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


def vectorstore_path(project_id: UUID | str, index_id: UUID | str) -> Path:
    p = project_dir(project_id) / "vectorstores" / str(index_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


def codegen_path(project_id: UUID | str, kind: str, name: str) -> Path:
    root = project_dir(project_id) / "code"
    p = _inside(root / kind, root)
    p.mkdir(parents=True, exist_ok=True)
    return _inside(p / name, p)
