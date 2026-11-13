"""Algorithm registry — factory + hyperparameter spec per algorithm.

The spec format is the same shape used by the frontend HyperparamForm to render
dynamic inputs. Each parameter entry:

    {type: "int" | "int_or_none" | "float" | "select" | "csv_int" | "bool",
     default: ..., min?, max?, step?, options?, label?, fmt?}

Fixed params (always applied, not user-editable) live under `fixed_params`.
"""
from __future__ import annotations

from typing import Any

from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAVE_XGB = True
except ImportError:
    _HAVE_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAVE_LGBM = True
except ImportError:
    _HAVE_LGBM = False


from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder


class _LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper for classifiers that require integer-encoded
    targets (e.g., XGBoost). Transparently encodes y on fit and decodes
    predict() output back to the original labels, so users of the pickled
    pipeline can pass string labels and get string predictions.
    """

    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator

    def fit(self, X, y, **kw):
        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y)
        self.base_estimator.fit(X, y_enc, **kw)
        self.classes_ = self.label_encoder_.classes_
        return self

    def predict(self, X):
        preds = self.base_estimator.predict(X)
        return self.label_encoder_.inverse_transform(preds.astype(int))

    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)


def _make_xgb_clf(**kwargs):
    return _LabelEncodedClassifier(XGBClassifier(**kwargs))


def _make_voting_clf(**kw):
    voting = kw.pop("voting", "soft")
    return VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=2000)),
            ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
            ("gb", GradientBoostingClassifier(random_state=42)),
        ],
        voting=voting,
        **kw,
    )


def _make_voting_reg(**kw):
    return VotingRegressor(
        estimators=[
            ("lr", LinearRegression()),
            ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
            ("gb", GradientBoostingRegressor(random_state=42)),
        ],
        **kw,
    )


def _make_stacking_clf(**kw):
    return StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
            ("gb", GradientBoostingClassifier(random_state=42)),
        ],
        final_estimator=LogisticRegression(max_iter=2000),
        **kw,
    )


def _make_stacking_reg(**kw):
    return StackingRegressor(
        estimators=[
            ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
            ("gb", GradientBoostingRegressor(random_state=42)),
        ],
        final_estimator=Ridge(),
        **kw,
    )


# ---------- import qualified names for code generation ----------
IMPORT_PATHS: dict[str, str] = {}


def _reg(name: str, cls: type) -> type:
    IMPORT_PATHS[name] = f"{cls.__module__}.{cls.__name__}"
    return cls


CLASSIFIERS: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "label": "Logistic Regression",
        "factory": _reg("logistic_regression", LogisticRegression),
        "description": "Linear classifier. Fast, interpretable coefficients, solid baseline.",
        "params": {
            "C": {"type": "float", "default": 1.0, "min": 0.001, "max": 100.0, "step": 0.1},
            "max_iter": {"type": "int", "default": 2000, "min": 100, "max": 20000, "step": 100},
            "penalty": {"type": "select", "default": "l2", "options": ["l2", "l1", "none"]},
            "solver": {"type": "select", "default": "lbfgs", "options": ["lbfgs", "liblinear", "saga"]},
        },
    },
    "knn_clf": {
        "label": "K-Nearest Neighbors",
        "factory": _reg("knn_clf", KNeighborsClassifier),
        "description": "Lazy learner — majority vote of k nearest points. Scale-sensitive.",
        "params": {
            "n_neighbors": {"type": "int", "default": 5, "min": 1, "max": 100, "step": 1},
            "weights": {"type": "select", "default": "uniform", "options": ["uniform", "distance"]},
            "p": {"type": "select", "default": 2, "options": [1, 2]},
        },
    },
    "decision_tree_clf": {
        "label": "Decision Tree",
        "factory": _reg("decision_tree_clf", DecisionTreeClassifier),
        "description": "Recursive splits on features. Interpretable, prone to overfit.",
        "params": {
            "max_depth": {"type": "int_or_none", "default": None, "min": 1, "max": 50, "step": 1},
            "min_samples_split": {"type": "int", "default": 2, "min": 2, "max": 50, "step": 1},
            "criterion": {"type": "select", "default": "gini", "options": ["gini", "entropy", "log_loss"]},
        },
    },
    "random_forest_clf": {
        "label": "Random Forest",
        "factory": _reg("random_forest_clf", RandomForestClassifier),
        "description": "Bagged decision trees. Robust default, handles non-linearity well.",
        "params": {
            "n_estimators": {"type": "int", "default": 200, "min": 10, "max": 1000, "step": 10},
            "max_depth": {"type": "int_or_none", "default": None, "min": 1, "max": 50, "step": 1},
            "criterion": {"type": "select", "default": "gini", "options": ["gini", "entropy", "log_loss"]},
        },
    },
    "gradient_boosting_clf": {
        "label": "Gradient Boosting",
        "factory": _reg("gradient_boosting_clf", GradientBoostingClassifier),
        "description": "Sequential trees correcting prior errors. Often best on tabular data.",
        "params": {
            "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 1000, "step": 10},
            "learning_rate": {"type": "float", "default": 0.1, "min": 0.001, "max": 2.0, "step": 0.01},
            "max_depth": {"type": "int", "default": 3, "min": 1, "max": 20, "step": 1},
        },
    },
    "svm_clf": {
        "label": "SVM",
        "factory": _reg("svm_clf", SVC),
        "description": "Margin-maximising classifier with kernel trick. Strong on small/medium data.",
        "fixed_params": {"probability": True},
        "params": {
            "C": {"type": "float", "default": 1.0, "min": 0.001, "max": 100.0, "step": 0.1},
            "kernel": {"type": "select", "default": "rbf", "options": ["rbf", "linear", "poly", "sigmoid"]},
            "gamma": {"type": "select", "default": "scale", "options": ["scale", "auto"]},
        },
    },
    "naive_bayes": {
        "label": "Gaussian Naive Bayes",
        "factory": _reg("naive_bayes", GaussianNB),
        "description": "Bayes rule assuming gaussian features. Fast, surprisingly strong.",
        "params": {
            "var_smoothing": {"type": "float", "default": 1e-9, "min": 1e-12, "max": 1.0, "step": 1e-9, "fmt": "%.2e"},
        },
    },
    "extra_trees_clf": {
        "label": "Extra Trees",
        "factory": _reg("extra_trees_clf", ExtraTreesClassifier),
        "description": "Like Random Forest but with randomised split thresholds — faster, comparable accuracy.",
        "params": {
            "n_estimators": {"type": "int", "default": 200, "min": 10, "max": 1000, "step": 10},
            "max_depth": {"type": "int_or_none", "default": None, "min": 1, "max": 50, "step": 1},
            "criterion": {"type": "select", "default": "gini", "options": ["gini", "entropy", "log_loss"]},
        },
    },
    "hist_gradient_boosting_clf": {
        "label": "Histogram Gradient Boosting",
        "factory": _reg("hist_gradient_boosting_clf", HistGradientBoostingClassifier),
        "description": "Fast sklearn boosting with histogram-binned features. Excellent default for tabular data.",
        "params": {
            "max_iter": {"type": "int", "default": 200, "min": 50, "max": 2000, "step": 50},
            "learning_rate": {"type": "float", "default": 0.1, "min": 0.001, "max": 2.0, "step": 0.01},
            "max_depth": {"type": "int_or_none", "default": None, "min": 1, "max": 20, "step": 1},
            "l2_regularization": {"type": "float", "default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1},
        },
    },
    "mlp_clf": {
        "label": "MLP (Neural Net)",
        "factory": _reg("mlp_clf", MLPClassifier),
        "description": "Shallow feed-forward neural network. Scale features first.",
        "params": {
            "hidden_layer_sizes": {"type": "csv_int", "default": "100", "label": "hidden_layer_sizes"},
            "alpha": {"type": "float", "default": 1e-4, "min": 1e-7, "max": 1.0, "step": 1e-4, "fmt": "%.5f"},
            "learning_rate_init": {"type": "float", "default": 1e-3, "min": 1e-6, "max": 1.0, "step": 1e-4, "fmt": "%.5f"},
            "max_iter": {"type": "int", "default": 500, "min": 50, "max": 5000, "step": 50},
            "activation": {"type": "select", "default": "relu", "options": ["relu", "tanh", "logistic"]},
        },
    },
    "voting_clf": {
        "label": "Voting (LR + RF + GB)",
        "factory": _reg("voting_clf", _make_voting_clf),
        "description": "Soft-vote ensemble of Logistic + Random Forest + Gradient Boosting.",
        "params": {
            "voting": {"type": "select", "default": "soft", "options": ["soft", "hard"]},
        },
    },
    "stacking_clf": {
        "label": "Stacking (RF + GB → LR)",
        "factory": _reg("stacking_clf", _make_stacking_clf),
        "description": "Stack RF + GB predictions, meta-learn with Logistic Regression.",
        "params": {},
    },
}

if _HAVE_XGB:
    CLASSIFIERS["xgboost_clf"] = {
        "label": "XGBoost",
        "factory": _reg("xgboost_clf", _make_xgb_clf),
        "description": "Industry-standard gradient boosting. Top-tier on tabular.",
        "fixed_params": {"eval_metric": "logloss"},
        "params": {
            "n_estimators": {"type": "int", "default": 300, "min": 10, "max": 2000, "step": 10},
            "max_depth": {"type": "int", "default": 6, "min": 1, "max": 20, "step": 1},
            "learning_rate": {"type": "float", "default": 0.1, "min": 0.001, "max": 2.0, "step": 0.01},
            "subsample": {"type": "float", "default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05},
            "colsample_bytree": {"type": "float", "default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05},
        },
    }

if _HAVE_LGBM:
    CLASSIFIERS["lightgbm_clf"] = {
        "label": "LightGBM",
        "factory": _reg("lightgbm_clf", LGBMClassifier),
        "description": "Fast leaf-wise gradient boosting. Great on large datasets.",
        "fixed_params": {"verbose": -1},
        "params": {
            "n_estimators": {"type": "int", "default": 300, "min": 10, "max": 2000, "step": 10},
            "num_leaves": {"type": "int", "default": 31, "min": 4, "max": 256, "step": 1},
            "learning_rate": {"type": "float", "default": 0.05, "min": 0.001, "max": 2.0, "step": 0.01},
            "max_depth": {"type": "int", "default": -1, "min": -1, "max": 20, "step": 1},
        },
    }

REGRESSORS: dict[str, dict[str, Any]] = {
    "linear_regression": {
        "label": "Linear Regression",
        "factory": _reg("linear_regression", LinearRegression),
        "description": "OLS baseline. Closed-form, fast, interpretable.",
        "params": {},
    },
    "ridge": {
        "label": "Ridge",
        "factory": _reg("ridge", Ridge),
        "description": "Linear + L2 penalty. Tames correlated features.",
        "params": {
            "alpha": {"type": "float", "default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1},
        },
    },
    "lasso": {
        "label": "Lasso",
        "factory": _reg("lasso", Lasso),
        "description": "Linear + L1 penalty. Drives some coefficients to zero (feature selection).",
        "params": {
            "alpha": {"type": "float", "default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01},
            "max_iter": {"type": "int", "default": 1000, "min": 100, "max": 20000, "step": 100},
        },
    },
    "elasticnet": {
        "label": "ElasticNet",
        "factory": _reg("elasticnet", ElasticNet),
        "description": "Mix of L1 + L2 penalties.",
        "params": {
            "alpha": {"type": "float", "default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01},
            "l1_ratio": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
        },
    },
    "knn_reg": {
        "label": "K-Nearest Neighbors Regressor",
        "factory": _reg("knn_reg", KNeighborsRegressor),
        "description": "Distance-weighted regression.",
        "params": {
            "n_neighbors": {"type": "int", "default": 5, "min": 1, "max": 100, "step": 1},
            "weights": {"type": "select", "default": "uniform", "options": ["uniform", "distance"]},
        },
    },
    "decision_tree_reg": {
        "label": "Decision Tree Regressor",
        "factory": _reg("decision_tree_reg", DecisionTreeRegressor),
        "description": "Recursive splits; interpretable but high-variance.",
        "params": {
            "max_depth": {"type": "int_or_none", "default": None, "min": 1, "max": 50, "step": 1},
            "min_samples_split": {"type": "int", "default": 2, "min": 2, "max": 50, "step": 1},
        },
    },
    "random_forest_reg": {
        "label": "Random Forest Regressor",
        "factory": _reg("random_forest_reg", RandomForestRegressor),
        "description": "Strong tabular default.",
        "params": {
            "n_estimators": {"type": "int", "default": 200, "min": 10, "max": 1000, "step": 10},
            "max_depth": {"type": "int_or_none", "default": None, "min": 1, "max": 50, "step": 1},
        },
    },
    "gradient_boosting_reg": {
        "label": "Gradient Boosting Regressor",
        "factory": _reg("gradient_boosting_reg", GradientBoostingRegressor),
        "description": "Top performer for tabular regression.",
        "params": {
            "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 1000, "step": 10},
            "learning_rate": {"type": "float", "default": 0.1, "min": 0.001, "max": 2.0, "step": 0.01},
            "max_depth": {"type": "int", "default": 3, "min": 1, "max": 20, "step": 1},
        },
    },
    "svr": {
        "label": "SVR",
        "factory": _reg("svr", SVR),
        "description": "Margin-based regression with kernel trick.",
        "params": {
            "C": {"type": "float", "default": 1.0, "min": 0.001, "max": 100.0, "step": 0.1},
            "kernel": {"type": "select", "default": "rbf", "options": ["rbf", "linear", "poly", "sigmoid"]},
        },
    },
    "extra_trees_reg": {
        "label": "Extra Trees Regressor",
        "factory": _reg("extra_trees_reg", ExtraTreesRegressor),
        "description": "Randomised-split tree ensemble; faster than RF.",
        "params": {
            "n_estimators": {"type": "int", "default": 200, "min": 10, "max": 1000, "step": 10},
            "max_depth": {"type": "int_or_none", "default": None, "min": 1, "max": 50, "step": 1},
        },
    },
    "hist_gradient_boosting_reg": {
        "label": "Histogram Gradient Boosting Regressor",
        "factory": _reg("hist_gradient_boosting_reg", HistGradientBoostingRegressor),
        "description": "Fast histogram-binned boosting regressor.",
        "params": {
            "max_iter": {"type": "int", "default": 200, "min": 50, "max": 2000, "step": 50},
            "learning_rate": {"type": "float", "default": 0.1, "min": 0.001, "max": 2.0, "step": 0.01},
            "max_depth": {"type": "int_or_none", "default": None, "min": 1, "max": 20, "step": 1},
        },
    },
    "mlp_reg": {
        "label": "MLP Regressor",
        "factory": _reg("mlp_reg", MLPRegressor),
        "description": "Shallow neural net for regression. Scale features first.",
        "params": {
            "hidden_layer_sizes": {"type": "csv_int", "default": "100", "label": "hidden_layer_sizes"},
            "alpha": {"type": "float", "default": 1e-4, "min": 1e-7, "max": 1.0, "step": 1e-4, "fmt": "%.5f"},
            "max_iter": {"type": "int", "default": 500, "min": 50, "max": 5000, "step": 50},
            "activation": {"type": "select", "default": "relu", "options": ["relu", "tanh", "logistic"]},
        },
    },
    "bayesian_ridge": {
        "label": "Bayesian Ridge",
        "factory": _reg("bayesian_ridge", BayesianRidge),
        "description": "Probabilistic linear regression with automatic regularisation.",
        "params": {
            "max_iter": {"type": "int", "default": 300, "min": 50, "max": 5000, "step": 50},
        },
    },
    "voting_reg": {
        "label": "Voting (Linear + RF + GB)",
        "factory": _reg("voting_reg", _make_voting_reg),
        "description": "Average predictions from Linear, Random Forest, and Gradient Boosting.",
        "params": {},
    },
    "stacking_reg": {
        "label": "Stacking (RF + GB → Ridge)",
        "factory": _reg("stacking_reg", _make_stacking_reg),
        "description": "Stack RF + GB predictions, meta-learn with Ridge.",
        "params": {},
    },
}

if _HAVE_XGB:
    REGRESSORS["xgboost_reg"] = {
        "label": "XGBoost Regressor",
        "factory": _reg("xgboost_reg", XGBRegressor),
        "description": "Industry-standard gradient boosting regressor.",
        "params": {
            "n_estimators": {"type": "int", "default": 300, "min": 10, "max": 2000, "step": 10},
            "max_depth": {"type": "int", "default": 6, "min": 1, "max": 20, "step": 1},
            "learning_rate": {"type": "float", "default": 0.1, "min": 0.001, "max": 2.0, "step": 0.01},
            "subsample": {"type": "float", "default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05},
            "colsample_bytree": {"type": "float", "default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05},
        },
    }

if _HAVE_LGBM:
    REGRESSORS["lightgbm_reg"] = {
        "label": "LightGBM Regressor",
        "factory": _reg("lightgbm_reg", LGBMRegressor),
        "description": "Fast leaf-wise boosting regressor.",
        "fixed_params": {"verbose": -1},
        "params": {
            "n_estimators": {"type": "int", "default": 300, "min": 10, "max": 2000, "step": 10},
            "num_leaves": {"type": "int", "default": 31, "min": 4, "max": 256, "step": 1},
            "learning_rate": {"type": "float", "default": 0.05, "min": 0.001, "max": 2.0, "step": 0.01},
            "max_depth": {"type": "int", "default": -1, "min": -1, "max": 20, "step": 1},
        },
    }

CLUSTERERS: dict[str, dict[str, Any]] = {
    "kmeans": {
        "label": "KMeans",
        "factory": _reg("kmeans", KMeans),
        "description": "Partition into k clusters by centroid distance. Fast.",
        "fixed_params": {"n_init": "auto"},
        "params": {
            "n_clusters": {"type": "int", "default": 4, "min": 2, "max": 50, "step": 1},
        },
    },
    "dbscan": {
        "label": "DBSCAN",
        "factory": _reg("dbscan", DBSCAN),
        "description": "Density-based; finds arbitrary shapes, flags outliers as -1.",
        "params": {
            "eps": {"type": "float", "default": 0.5, "min": 0.01, "max": 10.0, "step": 0.01},
            "min_samples": {"type": "int", "default": 5, "min": 1, "max": 100, "step": 1},
        },
    },
    "agglomerative": {
        "label": "Agglomerative",
        "factory": _reg("agglomerative", AgglomerativeClustering),
        "description": "Hierarchical bottom-up clustering.",
        "params": {
            "n_clusters": {"type": "int", "default": 4, "min": 2, "max": 50, "step": 1},
            "linkage": {"type": "select", "default": "ward", "options": ["ward", "complete", "average", "single"]},
        },
    },
    "gmm": {
        "label": "Gaussian Mixture",
        "factory": _reg("gmm", GaussianMixture),
        "description": "Soft clustering; assumes Gaussian components.",
        "params": {
            "n_components": {"type": "int", "default": 4, "min": 2, "max": 50, "step": 1},
        },
    },
}


def all_registries() -> dict[str, dict[str, dict[str, Any]]]:
    return {"classification": CLASSIFIERS, "regression": REGRESSORS, "clustering": CLUSTERERS}


# ---------- Algorithm-specific preprocessing requirements ----------
# Tree / forest / ensemble / NB models do not benefit from feature scaling.
# Linear / distance / margin / NN models do.
_NEEDS_SCALING_FALSE = {
    # classifiers
    "decision_tree_clf", "random_forest_clf", "gradient_boosting_clf",
    "naive_bayes", "extra_trees_clf", "hist_gradient_boosting_clf",
    "voting_clf", "stacking_clf", "xgboost_clf", "lightgbm_clf",
    # regressors
    "decision_tree_reg", "random_forest_reg", "gradient_boosting_reg",
    "extra_trees_reg", "hist_gradient_boosting_reg",
    "voting_reg", "stacking_reg", "xgboost_reg", "lightgbm_reg",
}


def algorithm_needs_scaling(key: str) -> bool:
    """True when the algorithm benefits from feature scaling."""
    return key not in _NEEDS_SCALING_FALSE


def list_algorithms(task: str | None = None) -> list[dict[str, Any]]:
    """Catalog entries for the frontend (factories stripped — JSON-safe)."""
    out: list[dict[str, Any]] = []
    regs = all_registries()
    tasks = [task] if task else list(regs.keys())
    for t in tasks:
        for key, spec in regs.get(t, {}).items():
            out.append({
                "key": key,
                "label": spec["label"],
                "task": t,
                "description": spec["description"],
                "params": spec.get("params", {}),
                "fixed_params": spec.get("fixed_params", {}),
                "needs_scaling": algorithm_needs_scaling(key),
            })
    return out


def get_algorithm(key: str) -> tuple[str, dict[str, Any]]:
    """Returns (task, spec) or raises KeyError."""
    for task, reg in all_registries().items():
        if key in reg:
            return task, reg[key]
    raise KeyError(f"Unknown algorithm: {key}")


def coerce_hparam_value(spec_entry: dict[str, Any], raw: Any) -> Any:
    """Coerce a user-submitted value to the spec's expected type."""
    t = spec_entry.get("type", "str")
    if raw is None or raw == "":
        return spec_entry.get("default")
    try:
        if t == "int":
            return int(raw)
        if t == "int_or_none":
            return None if raw in (None, "", "none", "None") else int(raw)
        if t == "float":
            return float(raw)
        if t == "bool":
            return bool(raw) if not isinstance(raw, str) else raw.lower() in {"1", "true", "yes"}
        if t == "csv_int":
            if isinstance(raw, (list, tuple)):
                return tuple(int(x) for x in raw)
            return tuple(int(x.strip()) for x in str(raw).split(",") if x.strip())
        if t == "select":
            # Preserve original option's type
            opts = spec_entry.get("options", [])
            for o in opts:
                if str(o) == str(raw):
                    return o
            return raw
    except (ValueError, TypeError):
        return spec_entry.get("default")
    return raw
