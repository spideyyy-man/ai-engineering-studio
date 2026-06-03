"""Per-column stats diff (before vs after a transform)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _safe_float(x) -> float | None:
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def column_stats(df: pd.DataFrame, col: str) -> dict[str, Any]:
    if col not in df.columns:
        return {"present": False}
    s = df[col]
    stats: dict[str, Any] = {
        "present": True,
        "dtype": str(s.dtype),
        "n": int(len(s)),
        "missing": int(s.isna().sum()),
        "nunique": int(s.nunique(dropna=True)),
    }
    if pd.api.types.is_numeric_dtype(s):
        s_clean = s.dropna()
        stats["mean"] = _safe_float(s_clean.mean()) if len(s_clean) else None
        stats["median"] = _safe_float(s_clean.median()) if len(s_clean) else None
        stats["std"] = _safe_float(s_clean.std()) if len(s_clean) > 1 else None
        stats["min"] = _safe_float(s_clean.min()) if len(s_clean) else None
        stats["max"] = _safe_float(s_clean.max()) if len(s_clean) else None
    else:
        vc = s.value_counts(dropna=True)
        stats["top_value"] = str(vc.index[0]) if len(vc) else None
        stats["top_freq"] = int(vc.iloc[0]) if len(vc) else 0
    return stats


def stats_diff(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    columns: list[str] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """For each named column, return {col: {before: {...}, after: {...}}}.

    If `columns` is None: compute the union of dropped + added + value-changed columns.
    """
    if columns is None:
        before_cols = set(df_before.columns)
        after_cols = set(df_after.columns)
        columns = sorted(before_cols | after_cols)

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for col in columns:
        out[col] = {
            "before": column_stats(df_before, col),
            "after": column_stats(df_after, col),
        }
    return out
