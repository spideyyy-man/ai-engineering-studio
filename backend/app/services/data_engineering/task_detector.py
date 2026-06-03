"""Heuristic task-type detection.

Inspects the schema + (optional) target column to decide whether this dataset
is best suited to classification, regression, time-series, NLP, RAG corpus,
or generic clustering. Returns the primary task + a list of signals so the UI
can show *why* a given task was chosen.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


TEXT_COLUMN_AVG_CHARS_RAG = 200
TEXT_COLUMN_AVG_CHARS_NLP = 40
CLASSIFICATION_NUMERIC_UNIQUE_MAX = 20
DATETIME_PARSE_SAMPLE = 50


def _avg_str_len(s: pd.Series) -> float:
    sample = s.dropna().astype(str)
    if sample.empty:
        return 0.0
    if len(sample) > 1000:
        sample = sample.sample(1000, random_state=0)
    return float(sample.str.len().mean())


def _looks_datetime(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if s.dtype != object:
        return False
    sample = s.dropna().astype(str).head(DATETIME_PARSE_SAMPLE)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().sum() / len(sample) > 0.8


def detect_task_type(df: pd.DataFrame, target_col: str | None = None) -> dict[str, Any]:
    n_rows = len(df)
    cols = list(df.columns)
    signals: list[str] = []

    # ---- Datetime / time-series check
    datetime_cols = [c for c in cols if _looks_datetime(df[c])]
    if datetime_cols:
        signals.append(f"Datetime-like column(s): {datetime_cols[:3]}")

    # ---- Text characteristics
    text_cols = [c for c in cols if df[c].dtype == object]
    long_text_cols = [c for c in text_cols if _avg_str_len(df[c]) >= TEXT_COLUMN_AVG_CHARS_RAG]
    medium_text_cols = [
        c for c in text_cols
        if TEXT_COLUMN_AVG_CHARS_NLP <= _avg_str_len(df[c]) < TEXT_COLUMN_AVG_CHARS_RAG
    ]

    if long_text_cols:
        signals.append(f"Long text column(s) (avg ≥ {TEXT_COLUMN_AVG_CHARS_RAG} chars): {long_text_cols[:3]}")
    if medium_text_cols:
        signals.append(f"Medium text column(s): {medium_text_cols[:3]}")

    # ---- Target-driven branch
    if target_col and target_col in df.columns:
        s = df[target_col]
        n_unique = int(s.nunique(dropna=True))
        if s.dtype == object or pd.api.types.is_bool_dtype(s):
            task = "classification"
            rationale = f"Target '{target_col}' is non-numeric with {n_unique} unique values."
        elif pd.api.types.is_integer_dtype(s) and n_unique <= CLASSIFICATION_NUMERIC_UNIQUE_MAX:
            task = "classification"
            rationale = f"Target '{target_col}' is integer with only {n_unique} unique values — treated as classes."
        elif n_unique <= 10:
            task = "classification"
            rationale = f"Target '{target_col}' has only {n_unique} unique values — treated as classification."
        else:
            task = "regression"
            rationale = f"Target '{target_col}' is continuous numeric — regression."
        # Override if we also see a strong datetime signal (likely supervised time-series forecasting)
        if datetime_cols and task == "regression":
            task = "time_series_regression"
            rationale += " A datetime column is present — likely time-series forecasting."
        return {
            "task": task,
            "rationale": rationale,
            "signals": signals,
            "is_supervised": True,
            "n_features": int(df.shape[1] - 1),
            "n_samples": int(n_rows),
            "ml_friendly": True,
        }

    # ---- Unsupervised heuristics
    if long_text_cols and len(long_text_cols) >= max(1, len(cols) // 2):
        return {
            "task": "rag_corpus",
            "rationale": "Most columns are long-form text — looks like a RAG corpus.",
            "signals": signals + ["Most columns are long text"],
            "is_supervised": False,
            "n_features": int(df.shape[1]),
            "n_samples": int(n_rows),
            "ml_friendly": True,
        }

    if medium_text_cols and not df.select_dtypes(include=[np.number]).columns.tolist():
        return {
            "task": "nlp",
            "rationale": "Text-heavy without numeric features — NLP task.",
            "signals": signals,
            "is_supervised": False,
            "n_features": int(df.shape[1]),
            "n_samples": int(n_rows),
            "ml_friendly": True,
        }

    if datetime_cols and df.select_dtypes(include=[np.number]).shape[1] >= 1:
        return {
            "task": "time_series",
            "rationale": "Datetime + numeric columns and no target — unsupervised time-series.",
            "signals": signals,
            "is_supervised": False,
            "n_features": int(df.shape[1]),
            "n_samples": int(n_rows),
            "ml_friendly": True,
        }

    if df.select_dtypes(include=[np.number]).shape[1] >= 2:
        return {
            "task": "clustering",
            "rationale": "Mostly numeric, no target — clustering.",
            "signals": signals,
            "is_supervised": False,
            "n_features": int(df.shape[1]),
            "n_samples": int(n_rows),
            "ml_friendly": True,
        }

    return {
        "task": "unknown",
        "rationale": "Couldn't infer a clear task — set a target column or upload more rows.",
        "signals": signals,
        "is_supervised": False,
        "n_features": int(df.shape[1]),
        "n_samples": int(n_rows),
        "ml_friendly": False,
    }
