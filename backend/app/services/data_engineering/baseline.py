"""Quick baseline-vs-dummy score on a dataset.

Trains two models in seconds:
  - Dummy (always-predict-majority / mean) — the floor any model should beat.
  - Decision tree (depth 3) — a simple-but-real model.

Returns both metrics + a delta. If a model can't even beat the dummy by a few
percent, the dataset/target signal is weak (good thing to tell the user up front).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


SAMPLE_LIMIT = 10_000


def _safe_float(x) -> float | None:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("e", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ], remainder="drop")


def _detect_task(df: pd.DataFrame, target: str) -> str:
    s = df[target]
    if s.dtype == object or pd.api.types.is_bool_dtype(s):
        return "classification"
    n_unique = int(s.nunique(dropna=True))
    if pd.api.types.is_integer_dtype(s) and n_unique <= 20:
        return "classification"
    if n_unique <= 10:
        return "classification"
    return "regression"


def compute_baseline(df: pd.DataFrame, target_col: str, random_state: int = 42) -> dict[str, Any]:
    if target_col not in df.columns:
        raise ValueError(f"target '{target_col}' not in dataset")

    work = df.dropna(subset=[target_col])
    if len(work) > SAMPLE_LIMIT:
        work = work.sample(SAMPLE_LIMIT, random_state=random_state).reset_index(drop=True)

    task = _detect_task(work, target_col)
    X = work.drop(columns=[target_col])
    y = work[target_col]

    stratify = y if task == "classification" else None
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=stratify
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=None
        )

    pre = _preprocessor(X)

    if task == "classification":
        dummy = Pipeline([("pre", pre), ("est", DummyClassifier(strategy="most_frequent"))])
        tree = Pipeline([("pre", pre), ("est", DecisionTreeClassifier(max_depth=3, random_state=random_state))])
        dummy.fit(X_tr, y_tr); tree.fit(X_tr, y_tr)
        d_pred, t_pred = dummy.predict(X_te), tree.predict(X_te)
        n_classes = int(pd.Series(y_te).nunique())
        avg = "binary" if n_classes == 2 else "weighted"
        f1_kwargs = {"average": avg, "zero_division": 0}
        if avg == "binary":
            # Text labels ("yes"/"no") need an explicit positive class;
            # sklearn's default pos_label=1 only works for 0/1 targets.
            f1_kwargs["pos_label"] = tree.classes_[1] if len(tree.classes_) == 2 else sorted(
                pd.Series(y_te).unique().tolist(), key=str)[1]

        primary_metric = "accuracy"
        dummy_score = _safe_float(accuracy_score(y_te, d_pred))
        tree_score = _safe_float(accuracy_score(y_te, t_pred))
        return {
            "task": "classification",
            "primary_metric": primary_metric,
            "dummy_score": dummy_score,
            "tree_score": tree_score,
            "delta": _safe_float((tree_score or 0) - (dummy_score or 0)),
            "extra": {
                "f1_dummy": _safe_float(f1_score(y_te, d_pred, **f1_kwargs)),
                "f1_tree": _safe_float(f1_score(y_te, t_pred, **f1_kwargs)),
                "n_classes": n_classes,
                "n_train": int(len(X_tr)),
                "n_test": int(len(X_te)),
            },
            "interpretation": _interpret_classif(dummy_score, tree_score),
        }

    # regression
    dummy = Pipeline([("pre", pre), ("est", DummyRegressor(strategy="mean"))])
    tree = Pipeline([("pre", pre), ("est", DecisionTreeRegressor(max_depth=3, random_state=random_state))])
    dummy.fit(X_tr, y_tr); tree.fit(X_tr, y_tr)
    d_pred, t_pred = dummy.predict(X_te), tree.predict(X_te)

    primary_metric = "r2"
    dummy_score = _safe_float(r2_score(y_te, d_pred))
    tree_score = _safe_float(r2_score(y_te, t_pred))
    return {
        "task": "regression",
        "primary_metric": primary_metric,
        "dummy_score": dummy_score,
        "tree_score": tree_score,
        "delta": _safe_float((tree_score or 0) - (dummy_score or 0)),
        "extra": {
            "mae_dummy": _safe_float(mean_absolute_error(y_te, d_pred)),
            "mae_tree": _safe_float(mean_absolute_error(y_te, t_pred)),
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
        },
        "interpretation": _interpret_reg(dummy_score, tree_score),
    }


def _interpret_classif(dummy: float | None, tree: float | None) -> str:
    if dummy is None or tree is None:
        return "Could not evaluate baseline."
    delta = tree - dummy
    if delta < 0.02:
        return ("Tree barely beats dummy — your features carry almost no signal for this target. "
                "Reconsider features, gather more data, or check for label noise.")
    if delta < 0.1:
        return ("Tree beats dummy by a small margin — the signal is real but weak. "
                "Better models / features should help.")
    return ("Tree beats dummy comfortably — the dataset has usable signal for this target.")


def _interpret_reg(dummy: float | None, tree: float | None) -> str:
    if dummy is None or tree is None:
        return "Could not evaluate baseline."
    delta = tree - dummy
    if delta < 0.05:
        return ("Tree's R² is barely above dummy — features explain little variance. "
                "Add features or consider whether the target is even predictable from this data.")
    if delta < 0.2:
        return "Tree beats dummy on R² — moderate signal."
    return "Tree clearly beats dummy on R² — strong predictive signal in your features."
