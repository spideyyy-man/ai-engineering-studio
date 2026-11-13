"""Convert the transform history of a dataset into a runnable preprocess.py.

The exporter emits TWO pieces:
  - A pandas `preprocess(df)` function that replays the history deterministically.
  - A `build_pipeline()` factory that returns an equivalent sklearn ColumnTransformer
    composed of imputers / encoders / scalers — for use in a real training script.

Some transforms (drop_duplicates, ffill/bfill, rename) don't have a sklearn
analogue and only show up in the pandas function. The two stages run in order:
pandas function first, sklearn pipeline second.
"""
from __future__ import annotations

from typing import Any

# --- ops that go in the pandas replay step (df-level)
DF_LEVEL_OPS = {
    "drop_rows_missing", "drop_column", "drop_duplicates",
    "forward_fill", "backward_fill",
    "drop_outliers_iqr", "clip_outliers_iqr",
    "log_transform",
    "strip_whitespace", "lowercase", "rename",
    # Imputation/encoding can also be done in pandas if user prefers — but we
    # prefer to route those into the sklearn pipeline.
}

# --- ops that get translated into a sklearn pipeline step keyed by op kind
SKLEARN_OPS = {
    "fill_mean":      ("impute_numeric",     "SimpleImputer(strategy='mean')"),
    "fill_median":    ("impute_numeric",     "SimpleImputer(strategy='median')"),
    "fill_mode":      ("impute_categorical", "SimpleImputer(strategy='most_frequent')"),
    "fill_constant":  ("impute_any",         None),  # special: needs value
    "one_hot_encode": ("encode",             "OneHotEncoder(handle_unknown='ignore', sparse_output=False)"),
    "label_encode":   ("encode_ordinal",     "OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)"),
}


def _py_lit(v: Any) -> str:
    """Best-effort Python literal for primitive params."""
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, (int, float, bool)) or v is None:
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_py_lit(x) for x in v) + "]"
    if isinstance(v, dict):
        items = ", ".join(f"{_py_lit(k)}: {_py_lit(x)}" for k, x in v.items())
        return "{" + items + "}"
    return repr(v)


def _df_line(op: str, params: dict[str, Any]) -> str:
    cols = params.get("columns") or []
    if op == "drop_rows_missing":
        if cols:
            return f"    df = df.dropna(subset={_py_lit(cols)}).reset_index(drop=True)"
        return "    df = df.dropna().reset_index(drop=True)"
    if op == "drop_column":
        return f"    df = df.drop(columns={_py_lit(cols)})"
    if op == "drop_duplicates":
        subset = f", subset={_py_lit(cols)}" if cols else ""
        return f"    df = df.drop_duplicates(keep='first'{subset}).reset_index(drop=True)"
    if op == "forward_fill":
        return f"    df[{_py_lit(cols)}] = df[{_py_lit(cols)}].ffill()" if cols else "    df = df.ffill()"
    if op == "backward_fill":
        return f"    df[{_py_lit(cols)}] = df[{_py_lit(cols)}].bfill()" if cols else "    df = df.bfill()"
    if op == "clip_outliers_iqr":
        k = float(params.get("k", 1.5))
        return (
            f"    for _c in {_py_lit(cols)}:\n"
            f"        _q1, _q3 = df[_c].quantile(0.25), df[_c].quantile(0.75)\n"
            f"        _iqr = _q3 - _q1\n"
            f"        df[_c] = df[_c].clip(lower=_q1 - {k}*_iqr, upper=_q3 + {k}*_iqr)"
        )
    if op == "drop_outliers_iqr":
        k = float(params.get("k", 1.5))
        return (
            f"    _mask = pd.Series(False, index=df.index)\n"
            f"    for _c in {_py_lit(cols)}:\n"
            f"        _q1, _q3 = df[_c].quantile(0.25), df[_c].quantile(0.75)\n"
            f"        _iqr = _q3 - _q1\n"
            f"        _mask = _mask | (df[_c] < _q1 - {k}*_iqr) | (df[_c] > _q3 + {k}*_iqr)\n"
            f"    df = df[~_mask].reset_index(drop=True)"
        )
    if op == "log_transform":
        return (
            f"    for _c in {_py_lit(cols)}:\n"
            f"        df['log_' + _c] = np.log1p(df[_c].clip(lower=-0.999))"
        )
    if op == "strip_whitespace":
        return f"    for _c in {_py_lit(cols)}:\n        df[_c] = df[_c].astype(str).str.strip()"
    if op == "lowercase":
        return f"    for _c in {_py_lit(cols)}:\n        df[_c] = df[_c].astype(str).str.lower()"
    if op == "rename":
        return f"    df = df.rename(columns={_py_lit(params.get('mapping') or {})})"
    # Fallthrough: pandas implementations for imputers (when user prefers in-replay)
    if op == "fill_mean":
        return f"    for _c in {_py_lit(cols)}:\n        df[_c] = df[_c].fillna(df[_c].mean())"
    if op == "fill_median":
        return f"    for _c in {_py_lit(cols)}:\n        df[_c] = df[_c].fillna(df[_c].median())"
    if op == "fill_mode":
        return (
            f"    for _c in {_py_lit(cols)}:\n"
            f"        _m = df[_c].mode(dropna=True)\n"
            f"        if len(_m): df[_c] = df[_c].fillna(_m.iloc[0])"
        )
    if op == "fill_constant":
        v = params.get("value", 0)
        return f"    for _c in {_py_lit(cols)}:\n        df[_c] = df[_c].fillna({_py_lit(v)})"
    if op == "one_hot_encode":
        drop_first = params.get("drop_first", False)
        return f"    df = pd.get_dummies(df, columns={_py_lit(cols)}, drop_first={_py_lit(drop_first)}, dtype=int)"
    if op == "label_encode":
        return (
            f"    for _c in {_py_lit(cols)}:\n"
            f"        df[_c] = pd.Categorical(df[_c]).codes"
        )
    return f"    # (unhandled op: {op})"


def _sklearn_lines(history: list[dict[str, Any]]) -> tuple[list[str], dict[str, list[str]]]:
    """Build sklearn ColumnTransformer source + per-column-kind imports collected."""
    # Group columns by what kind of impute/encoder we applied.
    numeric_impute: str | None = None
    cat_impute: str | None = None
    encoder: str | None = None
    numeric_cols: set[str] = set()
    categorical_cols: set[str] = set()

    for entry in history:
        op = entry["operation"]
        params = entry.get("params") or {}
        cols = params.get("columns") or []
        if op == "fill_mean":
            numeric_impute = "SimpleImputer(strategy='mean')"
            numeric_cols.update(cols)
        elif op == "fill_median":
            numeric_impute = "SimpleImputer(strategy='median')"
            numeric_cols.update(cols)
        elif op == "fill_mode":
            cat_impute = "SimpleImputer(strategy='most_frequent')"
            categorical_cols.update(cols)
        elif op == "fill_constant":
            v = _py_lit(params.get("value", 0))
            cat_impute = f"SimpleImputer(strategy='constant', fill_value={v})"
            categorical_cols.update(cols)
        elif op == "one_hot_encode":
            encoder = "OneHotEncoder(handle_unknown='ignore', sparse_output=False)"
            categorical_cols.update(cols)
        elif op == "label_encode":
            encoder = "OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)"
            categorical_cols.update(cols)

    if not (numeric_impute or cat_impute or encoder):
        return [], {}

    src: list[str] = []
    src.append("def build_pipeline(numeric_cols=None, categorical_cols=None):")
    src.append("    \"\"\"Return a ColumnTransformer mirroring the studio operations.\"\"\"")
    src.append("    if numeric_cols is None:")
    src.append(f"        numeric_cols = {_py_lit(sorted(numeric_cols))}")
    src.append("    if categorical_cols is None:")
    src.append(f"        categorical_cols = {_py_lit(sorted(categorical_cols))}")
    src.append("")

    parts = []
    if numeric_impute:
        src.append("    num_steps = [")
        src.append(f"        ('impute', {numeric_impute}),")
        src.append("        ('scale', StandardScaler()),")
        src.append("    ]")
        src.append("    num_pipe = Pipeline(num_steps)")
        parts.append("('num', num_pipe, numeric_cols)")

    cat_steps = []
    if cat_impute:
        cat_steps.append(f"        ('impute', {cat_impute}),")
    if encoder:
        cat_steps.append(f"        ('encode', {encoder}),")
    if cat_steps:
        src.append("    cat_steps = [")
        src.extend(cat_steps)
        src.append("    ]")
        src.append("    cat_pipe = Pipeline(cat_steps)")
        parts.append("('cat', cat_pipe, categorical_cols)")

    src.append("    return ColumnTransformer([")
    for p in parts:
        src.append(f"        {p},")
    src.append("    ], remainder='drop')")
    return src, {}


def build_pipeline_export(dataset_name: str, history: list[dict[str, Any]]) -> str:
    """Return a fully-runnable Python module as a string."""
    df_lines: list[str] = []

    for entry in history:
        op = entry["operation"]
        params = entry.get("params") or {}
        df_lines.append(f"    # step {entry['step']}: {entry.get('summary','').strip()}")
        line = _df_line(op, params)
        df_lines.append(line)

    if not df_lines:
        df_lines.append("    return df")
    else:
        df_lines.append("    return df")

    sklearn_lines, _ = _sklearn_lines(history)

    out: list[str] = []
    out.append('"""')
    out.append("Generated by AI Engineering Studio — Data Engineering Studio.")
    out.append("")
    out.append(f"Dataset:  {dataset_name}")
    out.append(f"Steps:    {len(history)} applied")
    out.append("")
    out.append("Two stages:")
    out.append("  1. `preprocess(df)`     — replays every transform exactly (pandas, deterministic).")
    out.append("  2. `build_pipeline()`   — equivalent sklearn ColumnTransformer (impute + scale + encode)")
    out.append("                            for production training pipelines.")
    out.append('"""')
    out.append("from __future__ import annotations")
    out.append("")
    out.append("import numpy as np")
    out.append("import pandas as pd")
    if sklearn_lines:
        out.append("from sklearn.compose import ColumnTransformer")
        out.append("from sklearn.impute import SimpleImputer")
        out.append("from sklearn.pipeline import Pipeline")
        out.append("from sklearn.preprocessing import (")
        out.append("    OneHotEncoder,")
        out.append("    OrdinalEncoder,")
        out.append("    StandardScaler,")
        out.append(")")
    out.append("")
    out.append("")
    out.append("def preprocess(df: pd.DataFrame) -> pd.DataFrame:")
    out.append("    \"\"\"Replay all transforms applied in the studio, in order.\"\"\"")
    out.append("    df = df.copy()")
    out.extend(df_lines)
    out.append("")
    out.append("")
    if sklearn_lines:
        out.extend(sklearn_lines)
        out.append("")
        out.append("")
        out.append("if __name__ == '__main__':")
        out.append("    import argparse")
        out.append("    p = argparse.ArgumentParser()")
        out.append("    p.add_argument('--csv', required=True)")
        out.append("    args = p.parse_args()")
        out.append("    raw = pd.read_csv(args.csv)")
        out.append("    cleaned = preprocess(raw)")
        out.append("    print(f'rows: {len(raw)} -> {len(cleaned)}')")
        out.append("    print(f'cols: {raw.shape[1]} -> {cleaned.shape[1]}')")
    else:
        out.append("if __name__ == '__main__':")
        out.append("    import argparse")
        out.append("    p = argparse.ArgumentParser()")
        out.append("    p.add_argument('--csv', required=True)")
        out.append("    args = p.parse_args()")
        out.append("    raw = pd.read_csv(args.csv)")
        out.append("    cleaned = preprocess(raw)")
        out.append("    print(f'rows: {len(raw)} -> {len(cleaned)}')")
        out.append("    print(f'cols: {raw.shape[1]} -> {cleaned.shape[1]}')")

    return "\n".join(out) + "\n"
