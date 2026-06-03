"""ML Studio — Tab 2."""
from .registry import (
    CLASSIFIERS,
    CLUSTERERS,
    REGRESSORS,
    coerce_hparam_value,
    get_algorithm,
    list_algorithms,
)
from .trainer import detect_task, train_model

__all__ = [
    "CLASSIFIERS",
    "REGRESSORS",
    "CLUSTERERS",
    "list_algorithms",
    "get_algorithm",
    "coerce_hparam_value",
    "detect_task",
    "train_model",
]
