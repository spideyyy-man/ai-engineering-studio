"""Data-quality analyzer: missing, outliers, skew, imbalance, leakage hints."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

MISSING_WARN_PCT = 5.0
MISSING_ERROR_PCT = 30.0
OUTLIER_WARN_PCT = 1.0
SKEW_WARN = 1.0
IMBALANCE_WARN_RATIO = 10.0   # majority/minority ratio
HIGH_CARDINALITY_THRESHOLD = 0.5  # nunique / n_rows
ID_LIKE_PATTERNS = ("id", "uuid", "guid", "email", "phone")


def _iqr_outliers(s: pd.Series) -> int:
    s = s.dropna()
    if len(s) < 4:
        return 0
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((s < low) | (s > high)).sum())


def build_quality_report(df: pd.DataFrame, target_col: str | None = None) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    n_rows = len(df)

    # --- Missing values
    for col in df.columns:
        pct = float(df[col].isna().mean() * 100) if n_rows else 0.0
        if pct >= MISSING_ERROR_PCT:
            issues.append({
                "severity": "error", "code": "missing_high", "column": str(col),
                "message": f"{pct:.1f}% missing in '{col}'",
                "suggestion": "Consider dropping the column, or using a strong imputation strategy.",
            })
        elif pct >= MISSING_WARN_PCT:
            issues.append({
                "severity": "warn", "code": "missing_moderate", "column": str(col),
                "message": f"{pct:.1f}% missing in '{col}'",
                "suggestion": "Impute with median (numeric) or mode (categorical).",
            })

    # --- Duplicates
    dup = int(df.duplicated().sum())
    if dup:
        issues.append({
            "severity": "warn", "code": "duplicates", "column": None,
            "message": f"{dup} duplicate rows ({dup / n_rows * 100:.2f}%)" if n_rows else f"{dup} duplicates",
            "suggestion": "Run df.drop_duplicates() before training.",
        })

    numeric_df = df.select_dtypes(include=[np.number])

    # --- Outliers (IQR)
    for col in numeric_df.columns:
        n_out = _iqr_outliers(numeric_df[col])
        pct = (n_out / n_rows * 100) if n_rows else 0.0
        if pct >= OUTLIER_WARN_PCT:
            issues.append({
                "severity": "warn", "code": "outliers", "column": str(col),
                "message": f"{n_out} outliers ({pct:.1f}%) in '{col}'",
                "suggestion": "Consider winsorising, log-transforming, or using a robust scaler.",
            })

    # --- Skew
    for col in numeric_df.columns:
        try:
            sk = float(numeric_df[col].skew())
        except Exception:
            continue
        if abs(sk) >= SKEW_WARN:
            issues.append({
                "severity": "info", "code": "skewed", "column": str(col),
                "message": f"Skewness {sk:.2f} in '{col}' — distribution is heavy-tailed.",
                "suggestion": "Apply np.log1p, Box-Cox, or Yeo-Johnson transform.",
            })

    # --- Class imbalance (for target)
    if target_col and target_col in df.columns:
        vc = df[target_col].value_counts(dropna=True)
        if len(vc) >= 2 and vc.iloc[-1] > 0:
            ratio = float(vc.iloc[0] / vc.iloc[-1])
            if ratio >= IMBALANCE_WARN_RATIO:
                issues.append({
                    "severity": "warn", "code": "class_imbalance", "column": str(target_col),
                    "message": f"Target '{target_col}' majority/minority ratio = {ratio:.1f}",
                    "suggestion": "Use SMOTE, class_weight='balanced', or stratified sampling.",
                })

    # --- Leakage hints: ID-like or high-cardinality categorical columns
    for col in df.columns:
        name = str(col).lower()
        if any(p in name for p in ID_LIKE_PATTERNS):
            issues.append({
                "severity": "warn", "code": "leakage_id_like", "column": str(col),
                "message": f"'{col}' looks like an identifier — likely leaks/overfits.",
                "suggestion": "Exclude from features before training.",
            })
        if df[col].dtype == "object" and n_rows:
            unique_ratio = df[col].nunique(dropna=True) / n_rows
            if unique_ratio > HIGH_CARDINALITY_THRESHOLD:
                issues.append({
                    "severity": "info", "code": "high_cardinality", "column": str(col),
                    "message": f"'{col}' has very high cardinality ({unique_ratio:.0%} unique).",
                    "suggestion": "Use target/frequency encoding, or drop if not predictive.",
                })

    # --- Constant columns
    for col in df.columns:
        if df[col].nunique(dropna=False) <= 1:
            issues.append({
                "severity": "warn", "code": "constant_column", "column": str(col),
                "message": f"'{col}' has a single unique value — provides no signal.",
                "suggestion": "Drop this column.",
            })

    summary = {"error": 0, "warn": 0, "info": 0}
    for i in issues:
        summary[i["severity"]] = summary.get(i["severity"], 0) + 1

    return {"issues": issues, "summary": summary}
