"""End-to-end training: preprocess + fit + evaluate + pickle + figures."""
from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .registry import algorithm_needs_scaling, coerce_hparam_value, get_algorithm
from .tuning import build_auto_grid, build_cv, summarise_cv_results


# ---------------------------------------------------------------- helpers

def detect_task(df: pd.DataFrame, target: str | None) -> str:
    if target is None or target not in df.columns:
        return "clustering"
    s = df[target]
    if s.dtype == object or pd.api.types.is_bool_dtype(s):
        return "classification"
    nunique = int(s.nunique(dropna=True))
    if pd.api.types.is_integer_dtype(s) and nunique <= 20:
        return "classification"
    if nunique <= 10:
        return "classification"
    return "regression"


def _build_preprocessor(X: pd.DataFrame, scale: bool = True) -> ColumnTransformer:
    """Build the auto-preprocessing pipeline.

    Numeric columns: median impute (+ StandardScaler if `scale`).
    Categorical columns: most-frequent impute + OneHotEncoder.
    """
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    num_pipe = Pipeline(num_steps)
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
    )


def _preprocessing_summary(X: pd.DataFrame, scale: bool) -> dict[str, Any]:
    """JSON-safe description of what the auto-preprocessor will do."""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return {
        "numeric_columns": len(num_cols),
        "categorical_columns": len(cat_cols),
        "steps": [
            {
                "kind": "impute_numeric",
                "applied": bool(num_cols),
                "method": "SimpleImputer(strategy='median')",
                "rationale": "Fill missing numeric values with the column median (robust to outliers).",
            },
            {
                "kind": "scale_numeric",
                "applied": bool(num_cols) and scale,
                "method": "StandardScaler()" if scale else "skipped",
                "rationale": (
                    "This algorithm benefits from feature scaling (linear / margin / distance / NN model)."
                    if scale else
                    "Tree-based / Naive Bayes algorithm — scaling has no effect, so it's skipped."
                ),
            },
            {
                "kind": "impute_categorical",
                "applied": bool(cat_cols),
                "method": "SimpleImputer(strategy='most_frequent')",
                "rationale": "Fill missing categorical values with the column mode.",
            },
            {
                "kind": "encode_categorical",
                "applied": bool(cat_cols),
                "method": "OneHotEncoder(handle_unknown='ignore', sparse_output=False)",
                "rationale": "Convert categorical columns to one-hot indicator columns (safe for any estimator).",
            },
        ],
    }


def _safe_float(x) -> float | None:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _fig_to_json(fig: go.Figure) -> dict[str, Any]:
    return json.loads(fig.to_json())


def _resolve_hparams(spec: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    pspec = spec.get("params", {})
    for k, v in (raw or {}).items():
        if k in pspec:
            coerced = coerce_hparam_value(pspec[k], v)
            if coerced is not None or pspec[k]["type"] == "int_or_none":
                out[k] = coerced
    # Apply defaults for params the user didn't touch — coerce them too,
    # so e.g. csv_int default "100" becomes (100,) for sklearn.
    for k, entry in pspec.items():
        if k not in out:
            out[k] = coerce_hparam_value(entry, entry.get("default"))
        if out.get(k) is None and entry["type"] != "int_or_none":
            out.pop(k, None)
    # Fixed (always-applied) params
    out.update(spec.get("fixed_params", {}))
    return out


# ---------------------------------------------------------------- main

def _default_scoring(task: str) -> str:
    if task == "classification":
        return "f1_weighted"
    if task == "regression":
        return "r2"
    return "silhouette"  # rarely used; clustering doesn't go through CV/HPT here


def train_model(
    df: pd.DataFrame,
    *,
    target_col: str | None,
    algorithm: str,
    hyperparams: dict[str, Any] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    pickle_path: Path | None = None,
    cv_strategy: str | None = None,
    cv_folds: int = 5,
    hpt_mode: str = "manual",
    hpt_n_iter: int = 20,
    hpt_scoring: str | None = None,
) -> dict[str, Any]:
    """Returns task/metrics/figures/etc. Additionally:
      - cv_used, cv_mean_score, cv_std_score when cv_strategy is set
      - hpt_best_params, hpt_cv_results, hpt_best_score when hpt_mode != 'manual'
    """

    task_inferred = detect_task(df, target_col)
    algo_task, spec = get_algorithm(algorithm)
    task = algo_task if (algo_task != task_inferred and algo_task != "clustering") else algo_task

    hparams = _resolve_hparams(spec, hyperparams or {})
    try:
        spec["factory"](random_state=random_state)
        hparams.setdefault("random_state", random_state)
    except TypeError:
        pass

    estimator = spec["factory"](**hparams)

    figures: dict[str, Any] = {}
    metrics: dict[str, Any] = {}

    # ---- Supervised
    if task in {"classification", "regression"}:
        if not target_col or target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' missing.")
        n_missing = int(df[target_col].isna().sum())
        if n_missing:
            df = df.dropna(subset=[target_col]).reset_index(drop=True)
        if df.empty:
            raise ValueError("No rows left after dropping missing target values.")

        X = df.drop(columns=[target_col])
        y = df[target_col]
        scale = algorithm_needs_scaling(algorithm)
        preprocessing_applied = _preprocessing_summary(X, scale=scale)
        pre = _build_preprocessor(X, scale=scale)
        pipe = Pipeline([("pre", pre), ("est", estimator)])

        stratify = y if task == "classification" else None
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=stratify
            )
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=None
            )

        # --- CV + HPT (Slice A) -----------------------------------------------
        # Resolve scoring for CV and HPT
        scoring = hpt_scoring or _default_scoring(task)
        cv = build_cv(cv_strategy, n_splits=cv_folds, random_state=random_state, task=task)

        cv_used = False
        cv_mean_score: float | None = None
        cv_std_score: float | None = None
        hpt_best_params: dict[str, Any] | None = None
        hpt_best_score: float | None = None
        hpt_cv_results: list[dict[str, Any]] | None = None

        if hpt_mode in ("grid", "random"):
            # Build the auto-grid from the algorithm's hyperparam spec
            spec_params = spec.get("params", {})
            param_grid = build_auto_grid(spec_params, estimator_prefix="est__")
            if not param_grid:
                raise ValueError(
                    f"Algorithm '{algorithm}' has no exposed hyperparameters to tune"
                )
            cv_for_search = cv or 3  # need *some* CV for HPT
            if hpt_mode == "grid":
                search = GridSearchCV(
                    pipe, param_grid=param_grid, scoring=scoring,
                    cv=cv_for_search, n_jobs=-1, refit=True,
                )
            else:
                search = RandomizedSearchCV(
                    pipe, param_distributions=param_grid, scoring=scoring,
                    n_iter=hpt_n_iter, cv=cv_for_search, n_jobs=-1, refit=True,
                    random_state=random_state,
                )
            search.fit(X_tr, y_tr)
            pipe = search.best_estimator_
            from .tuning import _jsonable
            hpt_best_params = {k.replace("est__", ""): _jsonable(v) for k, v in search.best_params_.items()}
            hpt_best_score = _safe_float(search.best_score_)
            hpt_cv_results = summarise_cv_results(search.cv_results_, top_n=10)
            # Surface the std of the best trial as cv_std_score
            best_row = next((r for r in hpt_cv_results if r["rank"] == 1), None)
            if best_row:
                cv_std_score = best_row.get("std_score")
            cv_used = True
        elif cv is not None:
            # CV without HPT — get cross-validated score on the training set
            scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=-1)
            cv_mean_score = _safe_float(float(scores.mean()))
            cv_std_score = _safe_float(float(scores.std()))
            cv_used = True
            pipe.fit(X_tr, y_tr)
        else:
            pipe.fit(X_tr, y_tr)

        # If HPT produced best_score_, surface it as cv_mean_score too
        if hpt_best_score is not None:
            cv_mean_score = hpt_best_score

        preds = pipe.predict(X_te)

        if task == "classification":
            n_classes = int(pd.Series(y_te).nunique())
            avg = "binary" if n_classes == 2 else "weighted"
            metrics = {
                "accuracy": _safe_float(accuracy_score(y_te, preds)),
                "precision": _safe_float(precision_score(y_te, preds, average=avg, zero_division=0)),
                "recall": _safe_float(recall_score(y_te, preds, average=avg, zero_division=0)),
                "f1": _safe_float(f1_score(y_te, preds, average=avg, zero_division=0)),
                "f1_macro": _safe_float(f1_score(y_te, preds, average="macro", zero_division=0)),
                "average": avg,
                "n_classes": n_classes,
            }
            try:
                proba = pipe.predict_proba(X_te)
                if n_classes == 2:
                    classes_ = pipe.classes_
                    auc = roc_auc_score((y_te == classes_[1]).astype(int), proba[:, 1])
                else:
                    auc = roc_auc_score(y_te, proba, multi_class="ovr", average="weighted")
                metrics["roc_auc"] = _safe_float(auc)
            except Exception:
                metrics["roc_auc"] = None

            # Confusion matrix figure
            labels = sorted(pd.Series(y_te).unique().tolist(), key=lambda x: str(x))
            cm = confusion_matrix(y_te, preds, labels=labels)
            cm_df = pd.DataFrame(
                cm,
                index=[f"true:{l}" for l in labels],
                columns=[f"pred:{l}" for l in labels],
            )
            fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues", aspect="auto")
            fig.update_layout(title="Confusion matrix", margin=dict(l=40, r=20, t=40, b=40), height=380)
            figures["confusion_matrix"] = _fig_to_json(fig)

            # Classification report as a JSON dict
            metrics["classification_report"] = classification_report(
                y_te, preds, output_dict=True, zero_division=0
            )

        else:  # regression
            rmse = math.sqrt(mean_squared_error(y_te, preds))
            metrics = {
                "r2": _safe_float(r2_score(y_te, preds)),
                "mae": _safe_float(mean_absolute_error(y_te, preds)),
                "mse": _safe_float(mean_squared_error(y_te, preds)),
                "rmse": _safe_float(rmse),
                "max_error": _safe_float(max_error(y_te, preds)),
            }

            comp = pd.DataFrame({"actual": y_te.values, "predicted": preds})
            fig = px.scatter(comp, x="actual", y="predicted", opacity=0.7,
                             title="Predicted vs Actual")
            mn, mx = float(comp["actual"].min()), float(comp["actual"].max())
            fig.add_shape(type="line", x0=mn, x1=mx, y0=mn, y1=mx, line=dict(dash="dash"))
            fig.update_layout(margin=dict(l=40, r=20, t=40, b=40), height=380)
            figures["pred_vs_actual"] = _fig_to_json(fig)

            resid = comp["actual"] - comp["predicted"]
            fig = px.scatter(x=comp["predicted"], y=resid,
                             labels={"x": "predicted", "y": "residual"},
                             title="Residuals", opacity=0.7)
            fig.add_hline(y=0, line_dash="dash")
            fig.update_layout(margin=dict(l=40, r=20, t=40, b=40), height=320)
            figures["residuals"] = _fig_to_json(fig)

        # Feature importance / |coef|
        try:
            est = pipe.named_steps["est"]
            pre = pipe.named_steps["pre"]
            names = list(pre.get_feature_names_out())
            importance = None
            title = ""
            if hasattr(est, "feature_importances_"):
                importance, title = est.feature_importances_, "Feature importance"
            elif hasattr(est, "coef_"):
                c = est.coef_
                importance, title = (np.abs(c).mean(axis=0) if c.ndim > 1 else np.abs(c)), "|coefficient| magnitude"
            if importance is not None and len(importance) == len(names):
                imp_df = pd.DataFrame({"feature": names, "importance": importance})
                imp_df = imp_df.sort_values("importance", ascending=True).tail(20)
                fig = px.bar(imp_df, x="importance", y="feature", orientation="h", title=title)
                fig.update_layout(margin=dict(l=40, r=20, t=40, b=40), height=420)
                figures["feature_importance"] = _fig_to_json(fig)
        except Exception:
            pass

        n_train, n_test = len(X_tr), len(X_te)

    # ---- Clustering
    else:
        X = df.copy()
        # Clustering algorithms (KMeans, DBSCAN, etc.) all benefit from scaling.
        preprocessing_applied = _preprocessing_summary(X, scale=True)
        pre = _build_preprocessor(X, scale=True)
        pipe = Pipeline([("pre", pre), ("est", estimator)])
        pipe.fit(X)

        est = pipe.named_steps["est"]
        labels = est.labels_ if hasattr(est, "labels_") else pipe.predict(X)
        labels = np.asarray(labels)

        n_clusters_found = int(len(set(labels.tolist())) - (1 if -1 in labels else 0))
        n_noise = int((labels == -1).sum())

        X_t = pipe.named_steps["pre"].transform(X)
        X_dense = X_t.toarray() if hasattr(X_t, "toarray") else X_t

        sil = db = ch = None
        if n_clusters_found >= 2:
            try:
                sil = _safe_float(silhouette_score(X_dense, labels))
            except Exception:
                pass
            try:
                db = _safe_float(davies_bouldin_score(X_dense, labels))
            except Exception:
                pass
            try:
                ch = _safe_float(calinski_harabasz_score(X_dense, labels))
            except Exception:
                pass

        metrics = {
            "n_clusters": n_clusters_found,
            "n_noise": n_noise,
            "silhouette": sil,
            "davies_bouldin": db,
            "calinski_harabasz": ch,
            "inertia": _safe_float(getattr(est, "inertia_", None)),
        }

        # PCA 2D scatter
        try:
            pca = PCA(n_components=2, random_state=random_state)
            coords = pca.fit_transform(X_dense)
            plot_df = pd.DataFrame({
                "PC1": coords[:, 0], "PC2": coords[:, 1],
                "cluster": [str(c) for c in labels.tolist()],
            })
            fig = px.scatter(plot_df, x="PC1", y="PC2", color="cluster", opacity=0.75,
                             title="PCA projection (coloured by cluster)")
            fig.update_layout(margin=dict(l=40, r=20, t=40, b=40), height=420)
            figures["pca_scatter"] = _fig_to_json(fig)
            metrics["pca_explained_variance"] = [
                _safe_float(pca.explained_variance_ratio_[0]),
                _safe_float(pca.explained_variance_ratio_[1]),
            ]
        except Exception:
            pass

        n_train, n_test = len(df), 0

    # ---- Persist pickle
    pickle_bytes = pickle.dumps(pipe)
    if pickle_path is not None:
        pickle_path.parent.mkdir(parents=True, exist_ok=True)
        pickle_path.write_bytes(pickle_bytes)

    # Column names used for inference
    if task in {"classification", "regression"}:
        feature_columns = [str(c) for c in df.drop(columns=[target_col]).columns]
    else:
        feature_columns = [str(c) for c in df.columns]

    # If HPT was used, prefer the BEST params over the user's initial hparams
    if task in {"classification", "regression"} and hpt_best_params:
        effective = {**hparams, **hpt_best_params}
    else:
        effective = hparams

    out: dict[str, Any] = {
        "task": task,
        "algorithm": algorithm,
        "algorithm_label": spec["label"],
        "hparams_effective": {k: (v if not isinstance(v, tuple) else list(v)) for k, v in effective.items()},
        "metrics": metrics,
        "figures": figures,
        "n_train": n_train,
        "n_test": n_test,
        "feature_columns": feature_columns,
        "target_column": target_col,
        "pickle_bytes": pickle_bytes,
        "preprocessing_applied": preprocessing_applied,
    }
    # CV / HPT extras (only meaningful for supervised tasks)
    if task in {"classification", "regression"}:
        out["cv_used"] = bool(cv_used)
        out["cv_strategy"] = cv_strategy
        out["cv_folds"] = int(cv_folds) if cv_used else None
        out["cv_mean_score"] = cv_mean_score
        out["cv_std_score"] = cv_std_score
        out["cv_scoring"] = scoring if cv_used else None
        out["hpt_mode"] = hpt_mode
        out["hpt_best_params"] = hpt_best_params
        out["hpt_best_score"] = hpt_best_score
        out["hpt_cv_results"] = hpt_cv_results
    return out
