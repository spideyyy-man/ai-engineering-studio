"""Snapshot store for dataset transforms — feeds undo + replay."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...core.paths import project_dir


def history_dir(project_id: str, dataset_id: str) -> Path:
    p = project_dir(project_id) / "history" / dataset_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def snapshot_path(project_id: str, dataset_id: str, step: int) -> Path:
    return history_dir(project_id, dataset_id) / f"step_{step:04d}.parquet"


def save_snapshot(df: pd.DataFrame, path: Path) -> int:
    """Snapshot the dataframe to disk as parquet. Returns bytes written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return int(path.stat().st_size) if path.exists() else 0


def load_snapshot(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
