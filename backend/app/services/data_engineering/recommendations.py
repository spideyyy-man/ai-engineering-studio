"""Rule-based preprocessing recommendations."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_recommendations(df: pd.DataFrame, target_col: str | None = None) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []

    numeric_df = df.select_dtypes(include=[np.number])
    cat_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    if target_col:
        cat_cols = [c for c in cat_cols if c != target_col]

    # --- Imputation
    num_with_missing = [str(c) for c in numeric_df.columns if df[c].isna().any()]
    if num_with_missing:
        recs.append({
            "step": "imputation",
            "columns": num_with_missing,
            "rationale": "Numeric columns contain missing values; median imputation is robust to outliers.",
            "code_hint": "SimpleImputer(strategy='median')",
        })
    cat_with_missing = [str(c) for c in cat_cols if df[c].isna().any()]
    if cat_with_missing:
        recs.append({
            "step": "imputation",
            "columns": cat_with_missing,
            "rationale": "Categorical columns contain missing values; mode (most-frequent) imputation preserves category structure.",
            "code_hint": "SimpleImputer(strategy='most_frequent')",
        })

    # --- Scaling
    scale_cols: list[str] = []
    for col in numeric_df.columns:
        s = numeric_df[col].dropna()
        if len(s) < 2:
            continue
        if s.std() == 0:
            continue
        scale_cols.append(str(col))
    if scale_cols:
        # Pick StandardScaler unless skew is large → RobustScaler
        heavy_skew = any(abs(numeric_df[c].skew()) > 1.5 for c in scale_cols if c in numeric_df.columns)
        recs.append({
            "step": "scaling",
            "columns": scale_cols,
            "rationale": (
                "Heavy-tailed distributions detected; RobustScaler is less sensitive to outliers."
                if heavy_skew else
                "Numeric features have different magnitudes; StandardScaler equalizes scale for linear / NN models."
            ),
            "code_hint": "RobustScaler()" if heavy_skew else "StandardScaler()",
        })

    # --- Encoding
    if cat_cols:
        low_card = [c for c in cat_cols if df[c].nunique(dropna=True) <= 20]
        high_card = [c for c in cat_cols if df[c].nunique(dropna=True) > 20]
        if low_card:
            recs.append({
                "step": "encoding",
                "columns": [str(c) for c in low_card],
                "rationale": "Low-cardinality categorical features — one-hot encoding is safe and interpretable.",
                "code_hint": "OneHotEncoder(handle_unknown='ignore', sparse_output=False)",
            })
        if high_card:
            recs.append({
                "step": "encoding",
                "columns": [str(c) for c in high_card],
                "rationale": "High-cardinality categorical features — target/ordinal encoding avoids dimensionality explosion.",
                "code_hint": "category_encoders.TargetEncoder()  # or OrdinalEncoder",
            })

    # --- Class balancing
    if target_col and target_col in df.columns:
        vc = df[target_col].value_counts(dropna=True)
        if len(vc) >= 2 and vc.iloc[-1] > 0 and vc.iloc[0] / vc.iloc[-1] >= 5:
            recs.append({
                "step": "balancing",
                "columns": [str(target_col)],
                "rationale": "Significant class imbalance detected; SMOTE or class_weight='balanced' helps minority recall.",
                "code_hint": "imblearn.over_sampling.SMOTE(random_state=42)",
            })

    # --- Drop constant / id-like columns
    drop_cols: list[str] = []
    for col in df.columns:
        if df[col].nunique(dropna=False) <= 1:
            drop_cols.append(str(col))
        elif str(col).lower() in {"id", "uuid", "guid"} or str(col).lower().endswith("_id"):
            drop_cols.append(str(col))
    if drop_cols:
        recs.append({
            "step": "drop",
            "columns": sorted(set(drop_cols)),
            "rationale": "Constant or identifier-like columns provide no signal and may leak.",
            "code_hint": "df.drop(columns=[...])",
        })

    # --- Duplicates
    if df.duplicated().any():
        recs.append({
            "step": "deduplicate",
            "columns": [],
            "rationale": f"{int(df.duplicated().sum())} duplicate rows inflate training samples.",
            "code_hint": "df = df.drop_duplicates()",
        })

    return recs
