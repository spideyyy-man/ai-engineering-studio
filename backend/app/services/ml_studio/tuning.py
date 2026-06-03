"""Cross-validation strategies + hyperparameter tuning helpers."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    LeaveOneOut,
    StratifiedKFold,
    TimeSeriesSplit,
)


CV_STRATEGIES = ["none", "kfold", "stratified_kfold", "time_series", "group_kfold", "leave_one_out"]


def build_cv(strategy: str | None, n_splits: int = 5, random_state: int = 42, task: str = "classification"):
    """Return a sklearn cross-validator or None for plain train/test split."""
    if not strategy or strategy in ("none", ""):
        return None
    if strategy == "kfold":
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    if strategy == "stratified_kfold":
        # Only valid for classification
        if task != "classification":
            return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    if strategy == "time_series":
        return TimeSeriesSplit(n_splits=n_splits)
    if strategy == "group_kfold":
        return GroupKFold(n_splits=n_splits)
    if strategy == "leave_one_out":
        return LeaveOneOut()
    raise ValueError(f"Unknown CV strategy: {strategy}")


# ---------- Auto-grid generation ----------

def _grid_for_int(spec: dict[str, Any]) -> list[int]:
    lo = int(spec.get("min", 1))
    hi = int(spec.get("max", lo + 100))
    default = spec.get("default")
    # Pick 4-6 reasonable values
    if hi - lo <= 10:
        return list(range(lo, hi + 1))
    candidates: list[int] = []
    if isinstance(default, int):
        candidates.append(default)
    # Add some geometric spread
    for frac in (0.1, 0.25, 0.5, 0.75, 1.0):
        v = int(lo + (hi - lo) * frac)
        if v not in candidates:
            candidates.append(v)
    return sorted(set(candidates))[:6]


def _grid_for_float(spec: dict[str, Any]) -> list[float]:
    lo = float(spec.get("min", 0.0))
    hi = float(spec.get("max", 1.0))
    default = spec.get("default")
    # Log-spaced if range spans >2 orders of magnitude AND positive
    if lo > 0 and hi / max(lo, 1e-12) > 100:
        grid = list(np.logspace(np.log10(lo), np.log10(hi), num=5))
    else:
        grid = list(np.linspace(lo, hi, num=5))
    if isinstance(default, (int, float)) and default not in grid:
        grid.append(float(default))
    return sorted(set(round(x, 6) for x in grid))[:6]


def build_auto_grid(spec_params: dict[str, dict[str, Any]], estimator_prefix: str = "est__") -> dict[str, list[Any]]:
    """Construct a sklearn-style param_grid keyed for `Pipeline([('pre',...), ('est', estimator)])`."""
    grid: dict[str, list[Any]] = {}
    for name, spec in spec_params.items():
        t = spec.get("type")
        try:
            if t == "int":
                values = _grid_for_int(spec)
            elif t == "float":
                values = _grid_for_float(spec)
            elif t == "select":
                values = list(spec.get("options", []))
            elif t == "int_or_none":
                values = [None] + _grid_for_int(spec)[:3]
            elif t == "bool":
                values = [True, False]
            else:
                continue
            if values:
                grid[f"{estimator_prefix}{name}"] = values
        except Exception:
            continue
    return grid


def summarise_cv_results(cv_results_: dict[str, Any], top_n: int = 5) -> list[dict[str, Any]]:
    """Top-N rows of GridSearchCV / RandomizedSearchCV's cv_results_ — JSON-safe."""
    params_list = list(cv_results_.get("params", []))
    rank_raw = cv_results_.get("rank_test_score")
    rank = list(rank_raw) if rank_raw is not None else list(range(1, len(params_list) + 1))
    means = list(cv_results_.get("mean_test_score", []))
    stds = list(cv_results_.get("std_test_score", []))

    rows: list[dict[str, Any]] = []
    for i, p in enumerate(params_list):
        rows.append({
            "rank": int(rank[i]) if i < len(rank) else i + 1,
            "mean_score": float(means[i]) if i < len(means) and means[i] is not None else None,
            "std_score": float(stds[i]) if i < len(stds) and stds[i] is not None else None,
            "params": {k.replace("est__", ""): _jsonable(v) for k, v in p.items()},
        })
    rows.sort(key=lambda r: r["rank"])
    return rows[:top_n]


def _jsonable(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (str, int, bool)):
        return v
    if isinstance(v, float):
        return float(v) if v == v else None  # NaN check
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return str(v)
