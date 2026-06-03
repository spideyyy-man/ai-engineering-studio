"""DL Studio — Tab 3.

Design any Keras network (Sequential or Functional), validate its shapes
statically, generate runnable code, and train tabular models in-studio when
TensorFlow is installed.
"""
from __future__ import annotations

from .architectures import list_architectures, scaffold_spec
from .builder import build_keras_model, tf_available, train_tabular
from .catalog import catalog_payload, list_layers
from .codegen import generate_code
from .validator import validate_spec

__all__ = [
    "list_architectures",
    "scaffold_spec",
    "list_layers",
    "catalog_payload",
    "validate_spec",
    "generate_code",
    "tf_available",
    "build_keras_model",
    "train_tabular",
]
