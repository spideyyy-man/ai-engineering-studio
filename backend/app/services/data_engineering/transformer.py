"""Concrete dataset transformations: drop, fill, encode, etc.

Each operation is a pure function `(df, params) -> (new_df, summary_text)`.
The router decides whether to overwrite the original file (inplace=True) or
clone the dataset to a new file + DB row (inplace=False).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from ...core.errors import ValidationError


# ---------------------------------------------------------------- ops

def _numeric_only(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def _existing(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def op_drop_rows_missing(df: pd.DataFrame, p: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    cols = _existing(df, p.get("columns") or list(df.columns))
    before = len(df)
    out = df.dropna(subset=cols).reset_index(drop=True)
    return out, f"Dropped {before - len(out)} rows with missing values in {cols or 'any column'}."


def op_drop_column(df: pd.DataFrame, p: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    cols = _existing(df, p.get("columns") or [])
    if not cols:
        raise ValidationError("drop_column requires `columns`")
    out = df.drop(columns=cols)
    return out, f"Dropped column(s): {', '.join(cols)}."


def op_fill_mean(df: pd.DataFrame, p: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    cols = _numeric_only(df, p.get("columns") or [])
    if not cols:
        raise ValidationError("fill_mean requires numeric `columns`")
    out = df.copy()
    for c in cols:
        out[c] = out[c].fillna(out[c].mean())
    return out, f"Filled NaN with column mean in {cols}."


def op_fill_median(df: pd.DataFrame, p: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    cols = _numeric_only(df, p.get("columns") or [])
    if not cols:
        raise ValidationError("fill_median requires numeric `columns`")
    out = df.copy()
    for c in cols:
        out[c] = out[c].fillna(out[c].median())
    return out, f"Filled NaN with column median in {cols}."


def op_fill_mode(df: pd.DataFrame, p: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    cols = _existing(df, p.get("columns") or [])
    if not cols:
        raise ValidationError("fill_mode requires `columns`")
    out = df.copy()
    for c in cols:
        m = out[c].mode(dropna=True)
        if len(m):
            out[c] = out[c].fillna(m.iloc[0])
    return out, f"Filled NaN with column mode in {cols}."


def op_fill_constant(df: pd.DataFrame, p: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    cols = _existing(df, p.get("columns") or [])
    value = p.get("value", 0)
    if not cols:
        raise ValidationError("fill_constant requires `columns`")
    out = df.copy()
    for c in cols:
        out[c] = out[c].fillna(value)
    return out, f"Filled NaN with {value!r} in {cols}."


def op_forward_fill(df: pd.DataFrame, p: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    cols = _existing(df, p.get("columns") or list(df.columns))
    out = df.copy()
    out[cols] = out[cols].ffill()
    return out, f"Forward-filled NaN in {cols}."


def op_backward_fill(df: pd.DataFrame, p: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    cols = _existing(df, p.get("columns") or list(df.columns))
    out = df.copy()
    out[cols] = out[cols].bfill()
    return out, f"Backward-filled NaN in {cols}."


def op_drop_duplicates(df: pd.DataFrame, p: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    before = len(df)
    cols = _existing(df, p.get("columns") or [])
    subset = cols or None
    out = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    return out, f"Dropped {before - len(out)} duplicate rows."


def op_clip_outliers_iqr(df: pd.DataFrame, p: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    cols = _numeric_only(df, p.get("columns") or [])
    k = float(p.get("k", 1.5))
    if not cols:
        raise ValidationError("clip_outliers_iqr requires numeric `columns`")
    out = df.copy()
    total_clipped = 0
    for c in cols:
        q1, q3 = out[c].quantile(0.25), out[c].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - k * iqr, q3 + k * iqr
        mask = (out[c] < low) | (out[c] > high)
        total_clipped += int(mask.sum())
        out[c] = out[c].clip(lower=low, upper=high)
    return out, f"Clipped {total_clipped} outliers (k={k}·IQR) in {cols}."


def op_drop_outliers_iqr(df: pd.DataFrame, p: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    cols = _numeric_only(df, p.get("columns") or [])
    k = float(p.get("k", 1.5))
    if not cols:
        raise ValidationError("drop_outliers_iqr requires numeric `columns`")
    out = df.copy()
    mask = pd.Series(False, index=out.index)
    for c in cols:
        q1, q3 = out[c].quantile(0.25), out[c].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - k * iqr, q3 + k * iqr
        mask = mask | (out[c] < low) | (out[c] > high)
    out = out[~mask].reset_index(drop=True)
    return out, f"Dropped {int(mask.sum())} outlier rows (k={k}·IQR) across {cols}."


def op_log_transform(df: pd.DataFrame, p: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    cols = _numeric_only(df, p.get("columns") or [])
    if not cols:
        raise ValidationError("log_transform requires numeric `columns`")
    replace = bool(p.get("replace", False))
    out = df.copy()
    affected: list[str] = []
    for c in cols:
        if replace:
            out[c] = np.log1p(out[c].clip(lower=-0.999))
            affected.append(c)
        else:
            new_name = p.get("rename") or f"log_{c}"
            out[new_name] = np.log1p(out[c].clip(lower=-0.999))
            affected.append(new_name)
    verb = "Replaced with log1p" if replace else "Added log1p columns"
    return out, f"{verb}: {affected}."


def op_one_hot_encode(df: pd.DataFrame, p: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    cols = _existing(df, p.get("columns") or [])
    if not cols:
        raise ValidationError("one_hot_encode requires `columns`")
    drop_first = bool(p.get("drop_first", False))
    out = pd.get_dummies(df, columns=cols, drop_first=drop_first, dtype=int)
    added = [c for c in out.columns if c not in df.columns]
    return out, f"One-hot encoded {cols} → {len(added)} new dummy columns."


def op_label_encode(df: pd.DataFrame, p: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    cols = _existing(df, p.get("columns") or [])
    if not cols:
        raise ValidationError("label_encode requires `columns`")
    out = df.copy()
    for c in cols:
        out[c] = pd.Categorical(out[c]).codes
    return out, f"Label-encoded {cols} (categories → integers)."


def op_strip_whitespace(df: pd.DataFrame, p: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    cols = [c for c in _existing(df, p.get("columns") or []) if df[c].dtype == object]
    if not cols:
        raise ValidationError("strip_whitespace requires text `columns`")
    out = df.copy()
    for c in cols:
        out[c] = out[c].astype(str).str.strip()
    return out, f"Stripped whitespace in {cols}."


def op_lowercase(df: pd.DataFrame, p: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    cols = [c for c in _existing(df, p.get("columns") or []) if df[c].dtype == object]
    if not cols:
        raise ValidationError("lowercase requires text `columns`")
    out = df.copy()
    for c in cols:
        out[c] = out[c].astype(str).str.lower()
    return out, f"Lowercased text in {cols}."


def op_rename(df: pd.DataFrame, p: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    mapping = p.get("mapping") or {}
    if not mapping:
        raise ValidationError("rename requires `mapping`: {old: new}")
    out = df.rename(columns=mapping)
    return out, f"Renamed columns: {mapping}."


# ---------------------------------------------------------------- registry

OPERATIONS: dict[str, dict[str, Any]] = {
    "drop_rows_missing": {
        "label": "Drop rows with missing values",
        "description": "Remove every row containing NaN in the selected columns.",
        "fits": ["missing_high", "missing_moderate", "missing"],
        "apply": op_drop_rows_missing,
        "params": {
            "columns": {"type": "columns", "scope": "any", "required": False,
                        "help": "Leave empty to drop on ANY column."},
        },
    },
    "drop_column": {
        "label": "Drop column(s)",
        "description": "Remove the column entirely from the dataset.",
        "fits": ["missing_high", "constant_column", "leakage_id_like", "high_cardinality"],
        "apply": op_drop_column,
        "params": {"columns": {"type": "columns", "scope": "any", "required": True}},
    },
    "fill_mean": {
        "label": "Fill NaN with column mean",
        "description": "Mean imputation — fastest for normally-distributed numeric features.",
        "fits": ["missing_high", "missing_moderate", "missing"],
        "apply": op_fill_mean,
        "params": {"columns": {"type": "columns", "scope": "numeric", "required": True}},
    },
    "fill_median": {
        "label": "Fill NaN with column median",
        "description": "Median imputation — robust to outliers.",
        "fits": ["missing_high", "missing_moderate", "missing"],
        "apply": op_fill_median,
        "params": {"columns": {"type": "columns", "scope": "numeric", "required": True}},
    },
    "fill_mode": {
        "label": "Fill NaN with column mode",
        "description": "Most-frequent imputation — works for any dtype, preferred for categorical.",
        "fits": ["missing_high", "missing_moderate", "missing"],
        "apply": op_fill_mode,
        "params": {"columns": {"type": "columns", "scope": "any", "required": True}},
    },
    "fill_constant": {
        "label": "Fill NaN with a custom value",
        "description": "Replace NaN with a value you choose (e.g., 0, \"unknown\").",
        "fits": ["missing_high", "missing_moderate", "missing"],
        "apply": op_fill_constant,
        "params": {
            "columns": {"type": "columns", "scope": "any", "required": True},
            "value": {"type": "string", "default": "0", "required": True,
                      "help": "Numbers parsed automatically; strings preserved."},
        },
    },
    "forward_fill": {
        "label": "Forward-fill (carry last value down)",
        "description": "Propagate the last valid observation forward — common for time series.",
        "fits": ["missing_high", "missing_moderate", "missing"],
        "apply": op_forward_fill,
        "params": {"columns": {"type": "columns", "scope": "any", "required": False}},
    },
    "backward_fill": {
        "label": "Backward-fill (carry next value up)",
        "description": "Propagate the next valid observation backward.",
        "fits": ["missing_high", "missing_moderate", "missing"],
        "apply": op_backward_fill,
        "params": {"columns": {"type": "columns", "scope": "any", "required": False}},
    },
    "drop_duplicates": {
        "label": "Drop duplicate rows",
        "description": "Remove exact duplicates; optionally check only a subset of columns.",
        "fits": ["duplicates"],
        "apply": op_drop_duplicates,
        "params": {
            "columns": {"type": "columns", "scope": "any", "required": False,
                        "help": "Leave empty to dedupe across ALL columns."},
        },
    },
    "clip_outliers_iqr": {
        "label": "Clip outliers to IQR fences",
        "description": "Values outside [Q1 − k·IQR, Q3 + k·IQR] are clipped to the boundary.",
        "fits": ["outliers"],
        "apply": op_clip_outliers_iqr,
        "params": {
            "columns": {"type": "columns", "scope": "numeric", "required": True},
            "k": {"type": "float", "default": 1.5, "min": 0.5, "max": 5.0, "step": 0.1},
        },
    },
    "drop_outliers_iqr": {
        "label": "Drop outlier rows (IQR rule)",
        "description": "Remove rows where any selected column lies outside k·IQR fences.",
        "fits": ["outliers"],
        "apply": op_drop_outliers_iqr,
        "params": {
            "columns": {"type": "columns", "scope": "numeric", "required": True},
            "k": {"type": "float", "default": 1.5, "min": 0.5, "max": 5.0, "step": 0.1},
        },
    },
    "log_transform": {
        "label": "Log1p transform",
        "description": "Apply log(1 + x). Toggle 'replace' to overwrite the column in place (recommended for skew); leave off to add log_<col> alongside the original.",
        "fits": ["skewed"],
        "apply": op_log_transform,
        "params": {
            "columns": {"type": "columns", "scope": "numeric", "required": True},
            "replace": {
                "type": "bool", "default": True,
                "help": "Overwrite the source column. Disable to keep the original alongside a new log_<col>.",
            },
        },
    },
    "one_hot_encode": {
        "label": "One-hot encode",
        "description": "Replace each category with a 0/1 indicator column.",
        "fits": ["high_cardinality"],
        "apply": op_one_hot_encode,
        "params": {
            "columns": {"type": "columns", "scope": "categorical", "required": True},
            "drop_first": {"type": "bool", "default": False,
                            "help": "Drop the first category to avoid the dummy-variable trap."},
        },
    },
    "label_encode": {
        "label": "Label encode",
        "description": "Map each category to an integer code. Best for high-cardinality categoricals.",
        "fits": ["high_cardinality"],
        "apply": op_label_encode,
        "params": {"columns": {"type": "columns", "scope": "categorical", "required": True}},
    },
    "strip_whitespace": {
        "label": "Strip whitespace",
        "description": "Trim leading/trailing whitespace in text columns.",
        "fits": [],
        "apply": op_strip_whitespace,
        "params": {"columns": {"type": "columns", "scope": "categorical", "required": True}},
    },
    "lowercase": {
        "label": "Lowercase text",
        "description": "Convert text values to lowercase — useful before deduplication.",
        "fits": [],
        "apply": op_lowercase,
        "params": {"columns": {"type": "columns", "scope": "categorical", "required": True}},
    },
    "rename": {
        "label": "Rename columns",
        "description": "Change column names — handy when raw uploads have ugly headers.",
        "fits": [],
        "apply": op_rename,
        "params": {"mapping": {"type": "mapping", "required": True}},
    },
}


def list_operations() -> list[dict[str, Any]]:
    """JSON-safe catalog for the frontend."""
    out: list[dict[str, Any]] = []
    for key, op in OPERATIONS.items():
        out.append({
            "key": key,
            "label": op["label"],
            "description": op["description"],
            "fits": op.get("fits", []),
            "params": op.get("params", {}),
        })
    return out


def suggest_operations(issue_code: str) -> list[str]:
    return [key for key, op in OPERATIONS.items() if issue_code in op.get("fits", [])]


def apply_operation(df: pd.DataFrame, key: str, params: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    if key not in OPERATIONS:
        raise ValidationError(f"Unknown operation: {key}")
    fn: Callable[..., tuple[pd.DataFrame, str]] = OPERATIONS[key]["apply"]
    out, summary = fn(df, params or {})
    return out, summary


# ---------------------------------------------------------------- recommendation

def recommend_fix(
    df: pd.DataFrame,
    issue_code: str,
    column: str | None = None,
) -> dict[str, Any] | None:
    """Pick a sensible default fix for one diagnosed issue.

    Returns {operation, params, rationale} or None if there is no sensible
    automated transform (e.g., class_imbalance — handled at training time).
    """

    def _kind(col: str | None) -> str:
        if not col or col not in df.columns:
            return "unknown"
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            return "datetime"
        if pd.api.types.is_numeric_dtype(s):
            return "numeric"
        return "categorical"

    kind = _kind(column)

    # ---- missing values
    if issue_code == "missing_high":
        return {
            "operation": "drop_column",
            "params": {"columns": [column]} if column else {},
            "rationale": "≥30% of values are missing — too sparse to impute reliably; drop the column.",
        }
    if issue_code == "missing_moderate":
        if kind == "numeric":
            return {
                "operation": "fill_median",
                "params": {"columns": [column]},
                "rationale": "Numeric column — median imputation is robust to outliers.",
            }
        if kind == "datetime":
            return {
                "operation": "forward_fill",
                "params": {"columns": [column]},
                "rationale": "Datetime column — forward-fill propagates the last observed value.",
            }
        if kind == "categorical":
            return {
                "operation": "fill_mode",
                "params": {"columns": [column]},
                "rationale": "Categorical column — most-frequent value preserves the dominant category.",
            }
        return None

    # ---- duplicates
    if issue_code == "duplicates":
        return {
            "operation": "drop_duplicates",
            "params": {},
            "rationale": "Drop exact duplicate rows — they inflate training samples.",
        }

    # ---- outliers
    if issue_code == "outliers":
        return {
            "operation": "clip_outliers_iqr",
            "params": {"columns": [column], "k": 1.5} if column else {"k": 1.5},
            "rationale": "Clip to IQR fences — caps extremes without dropping rows.",
        }

    # ---- skew
    if issue_code == "skewed":
        return {
            "operation": "log_transform",
            "params": ({"columns": [column], "replace": True} if column
                       else {"replace": True}),
            "rationale": "Heavy-tailed numeric column — log1p applied in place to truly resolve the skew.",
        }

    # ---- high cardinality categorical
    if issue_code == "high_cardinality":
        if column and column in df.columns:
            n_unique = int(df[column].nunique(dropna=True))
            if n_unique <= 20:
                return {
                    "operation": "one_hot_encode",
                    "params": {"columns": [column], "drop_first": False},
                    "rationale": f"Only {n_unique} unique values — one-hot encoding stays manageable.",
                }
            return {
                "operation": "label_encode",
                "params": {"columns": [column]},
                "rationale": f"{n_unique} unique values — label-encoding avoids dimensionality blow-up.",
            }
        return None

    # ---- constant
    if issue_code == "constant_column":
        return {
            "operation": "drop_column",
            "params": {"columns": [column]} if column else {},
            "rationale": "Single unique value — provides no signal; drop the column.",
        }

    # ---- ID-like leakage
    if issue_code == "leakage_id_like":
        return {
            "operation": "drop_column",
            "params": {"columns": [column]} if column else {},
            "rationale": "Identifier-like column — exclude to prevent leakage and overfit.",
        }

    # ---- class imbalance (no transform; handled at training time)
    if issue_code == "class_imbalance":
        return None

    return None
