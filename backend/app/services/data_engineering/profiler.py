"""EDA profiling: shape, dtypes, missing, describe, correlations, plots."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Limit chart columns for performance
MAX_NUMERIC_PLOTS = 8
MAX_CATEGORICAL_PLOTS = 6
MAX_CORR_COLS = 25


def _safe_float(x) -> float | None:
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _fig_to_json(fig: go.Figure) -> dict[str, Any]:
    return json.loads(fig.to_json())


def summarize_for_card(df: pd.DataFrame) -> dict[str, Any]:
    """Lightweight summary for dataset list cards."""
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "size_bytes": int(df.memory_usage(deep=True).sum()),
        "columns": [
            {"name": str(c), "dtype": str(df[c].dtype)} for c in df.columns
        ],
        "sample": df.head(10).fillna("").to_dict(orient="records"),
    }


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Return the full EDA payload — matches schemas.dataset.ProfileResponse."""

    n_rows, n_cols = df.shape
    memory = int(df.memory_usage(deep=True).sum())
    dtypes = {str(c): str(t) for c, t in df.dtypes.items()}

    # ---- Missing values ----
    missing = {}
    for col in df.columns:
        n_miss = int(df[col].isna().sum())
        missing[str(col)] = {
            "count": n_miss,
            "pct": _safe_float(n_miss / n_rows * 100) if n_rows else 0.0,
        }

    duplicates = int(df.duplicated().sum())

    numeric_df = df.select_dtypes(include=[np.number])
    categorical_df = df.select_dtypes(exclude=[np.number])

    # ---- describe() ----
    describe_numeric: dict[str, dict[str, float | None]] = {}
    if not numeric_df.empty:
        nd = numeric_df.describe().to_dict()
        describe_numeric = {
            str(col): {str(stat): _safe_float(v) for stat, v in stats.items()}
            for col, stats in nd.items()
        }

    describe_categorical: dict[str, dict[str, Any]] = {}
    for col in categorical_df.columns:
        s = categorical_df[col]
        vc = s.value_counts(dropna=True)
        describe_categorical[str(col)] = {
            "unique": int(s.nunique(dropna=True)),
            "top": str(vc.index[0]) if len(vc) else None,
            "freq": int(vc.iloc[0]) if len(vc) else 0,
            "top_values": vc.head(10).to_dict(),
        }

    # ---- Correlations ----
    correlations: dict[str, dict[str, float | None]] | None = None
    if numeric_df.shape[1] >= 2:
        cols = numeric_df.columns[:MAX_CORR_COLS]
        corr = numeric_df[cols].corr(numeric_only=True)
        correlations = {
            str(c): {str(c2): _safe_float(corr.loc[c, c2]) for c2 in corr.columns}
            for c in corr.index
        }

    # ---- Figures (Plotly JSON) ----
    figures: dict[str, Any] = {}

    # Histograms for numeric
    for col in numeric_df.columns[:MAX_NUMERIC_PLOTS]:
        fig = px.histogram(numeric_df, x=col, nbins=30, title=f"Distribution: {col}")
        fig.update_layout(margin=dict(l=40, r=20, t=40, b=40), height=300)
        figures[f"hist::{col}"] = _fig_to_json(fig)

    # Boxplots for numeric
    if not numeric_df.empty:
        melt_cols = list(numeric_df.columns[:MAX_NUMERIC_PLOTS])
        if melt_cols:
            melted = numeric_df[melt_cols].melt(var_name="column", value_name="value")
            fig = px.box(melted, x="column", y="value", title="Numeric boxplots")
            fig.update_layout(margin=dict(l=40, r=20, t=40, b=40), height=350)
            figures["boxplots"] = _fig_to_json(fig)

    # Bar chart for top categorical
    for col in list(categorical_df.columns)[:MAX_CATEGORICAL_PLOTS]:
        vc = categorical_df[col].value_counts(dropna=True).head(15)
        if len(vc) == 0:
            continue
        fig = px.bar(x=vc.index.astype(str), y=vc.values, title=f"Top categories: {col}",
                     labels={"x": col, "y": "count"})
        fig.update_layout(margin=dict(l=40, r=20, t=40, b=40), height=300)
        figures[f"cats::{col}"] = _fig_to_json(fig)

    # Missingness bar
    miss_series = df.isna().sum().sort_values(ascending=False)
    miss_series = miss_series[miss_series > 0]
    if len(miss_series):
        fig = px.bar(x=miss_series.index.astype(str), y=miss_series.values,
                     title="Missing values per column",
                     labels={"x": "column", "y": "missing"})
        fig.update_layout(margin=dict(l=40, r=20, t=40, b=40), height=300)
        figures["missing_bar"] = _fig_to_json(fig)

    # Correlation heatmap
    if correlations is not None:
        cols = list(correlations.keys())
        z = [[correlations[r].get(c) for c in cols] for r in cols]
        fig = go.Figure(go.Heatmap(
            z=z, x=cols, y=cols, colorscale="RdBu", zmin=-1, zmax=1, colorbar=dict(title="ρ"),
        ))
        fig.update_layout(title="Correlation matrix", margin=dict(l=40, r=20, t=40, b=40), height=400)
        figures["corr_heatmap"] = _fig_to_json(fig)

    return {
        "shape": {"rows": int(n_rows), "cols": int(n_cols)},
        "memory_usage_bytes": memory,
        "dtypes": dtypes,
        "missing": missing,
        "duplicates": duplicates,
        "describe_numeric": describe_numeric,
        "describe_categorical": describe_categorical,
        "correlations": correlations,
        "figures": figures,
    }
