"""Dataset loading + persisting of uploaded files."""
from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pandas as pd

from ...core.errors import ValidationError


SUPPORTED_EXT = {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json"}


def save_upload(stream: BinaryIO, dest: Path) -> int:
    """Stream upload to disk; returns bytes written."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with dest.open("wb") as f:
        while chunk := stream.read(1024 * 1024):
            f.write(chunk)
            total += len(chunk)
    return total


def load_dataframe(path: str | Path, *, nrows: int | None = None) -> pd.DataFrame:
    """Read a dataset file into a DataFrame, dispatching by extension."""
    p = Path(path)
    if not p.exists():
        raise ValidationError(f"file not found: {p}")

    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXT:
        raise ValidationError(f"unsupported extension {ext}; supported: {sorted(SUPPORTED_EXT)}")

    if ext == ".csv":
        return pd.read_csv(p, nrows=nrows)
    if ext == ".tsv":
        return pd.read_csv(p, sep="\t", nrows=nrows)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(p, nrows=nrows)
    if ext == ".parquet":
        df = pd.read_parquet(p)
        return df.head(nrows) if nrows else df
    if ext == ".json":
        df = pd.read_json(p)
        return df.head(nrows) if nrows else df

    raise ValidationError(f"unsupported extension {ext}")  # pragma: no cover
