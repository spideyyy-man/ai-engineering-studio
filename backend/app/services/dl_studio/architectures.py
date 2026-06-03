"""DL Studio — architecture catalog + scaffolder.

`list_architectures` powers the preset picker. `scaffold_spec` turns a preset
(plus the concrete input shape / class count) into a *full, editable* ModelSpec:
an ordered list of layers, a compile block, and a training block. The user can
then add / remove / reorder layers in the builder — the preset is just a smart
starting point, never a cage.
"""
from __future__ import annotations

from typing import Any

from .catalog import default_loss, default_metrics, layer_defaults, output_kind


ARCHITECTURE_CATALOG: list[dict[str, Any]] = [
    {"key": "ann", "label": "Artificial Neural Network", "data_type": "tabular",
     "api": "sequential",
     "description": "Fully connected NN for tabular classification / regression.",
     "trainable_here": True},
    {"key": "ann_wide_deep", "label": "Wide & Deep (Functional)", "data_type": "tabular",
     "api": "functional",
     "description": "Functional-API tabular net — demonstrates the functional builder.",
     "trainable_here": True},
    {"key": "cnn", "label": "CNN", "data_type": "image",
     "api": "sequential",
     "description": "Convolutional network for image classification.",
     "trainable_here": False},
    {"key": "transfer_resnet", "label": "Transfer (ResNet50)", "data_type": "image",
     "api": "sequential",
     "description": "Transfer learning from ImageNet-pretrained ResNet50.",
     "trainable_here": False},
    {"key": "lstm", "label": "LSTM (Text)", "data_type": "text",
     "api": "sequential",
     "description": "Embedding + LSTM for text / sequence classification.",
     "trainable_here": False},
    {"key": "gru", "label": "GRU (Text)", "data_type": "text",
     "api": "sequential",
     "description": "Embedding + GRU — lighter than LSTM.",
     "trainable_here": False},
    {"key": "transformer", "label": "Transformer (Text)", "data_type": "text",
     "api": "functional",
     "description": "Embedding + self-attention encoder blocks for text.",
     "trainable_here": False},
    {"key": "lstm_ts", "label": "LSTM (Time Series)", "data_type": "timeseries",
     "api": "sequential",
     "description": "Sequence-to-one LSTM for forecasting.",
     "trainable_here": False},
]

ARCH_BY_KEY = {a["key"]: a for a in ARCHITECTURE_CATALOG}

# Which DL task family each data_type defaults to.
DATA_TYPE_TASK = {
    "tabular": "tabular_classification",   # overridden to regression by caller when applicable
    "image": "image_classification",
    "text": "text_classification",
    "timeseries": "timeseries_forecast",
}


def list_architectures(data_type: str | None = None) -> list[dict[str, Any]]:
    if data_type is None:
        return [dict(a) for a in ARCHITECTURE_CATALOG]
    return [dict(a) for a in ARCHITECTURE_CATALOG if a["data_type"] == data_type]


def _layer(kind: str, **overrides: Any) -> dict[str, Any]:
    params = layer_defaults(kind)
    params.update(overrides)
    return {"type": kind, "params": params, "name": None}


def _head(task: str, n_classes: int | None) -> dict[str, Any]:
    """The final Dense layer matching the task."""
    ok = output_kind(task, n_classes)
    if ok == "regression":
        return _layer("dense", units=1, activation="linear")
    if ok == "binary_classification":
        return _layer("dense", units=1, activation="sigmoid")
    return _layer("dense", units=int(n_classes or 2), activation="softmax")


def _body_layers(arch: str, input_shape: list[int], n_classes: int | None) -> list[dict[str, Any]]:
    if arch in ("ann", "ann_wide_deep"):
        return [
            _layer("dense", units=128, activation="relu"),
            _layer("batchnorm"),
            _layer("dropout", rate=0.3),
            _layer("dense", units=64, activation="relu"),
            _layer("dropout", rate=0.2),
        ]
    if arch == "cnn":
        return [
            _layer("conv2d", filters=32, kernel_size=3, padding="same", activation="relu"),
            _layer("maxpool2d", pool_size=2),
            _layer("conv2d", filters=64, kernel_size=3, padding="same", activation="relu"),
            _layer("maxpool2d", pool_size=2),
            _layer("conv2d", filters=128, kernel_size=3, padding="same", activation="relu"),
            _layer("globalavgpool2d"),
            _layer("dense", units=128, activation="relu"),
            _layer("dropout", rate=0.3),
        ]
    if arch == "transfer_resnet":
        return [
            _layer("resnet50_base", trainable=False),
            _layer("dense", units=256, activation="relu"),
            _layer("dropout", rate=0.3),
        ]
    if arch in ("lstm", "gru"):
        rnn = "lstm" if arch == "lstm" else "gru"
        return [
            _layer("embedding", input_dim=10000, output_dim=128),
            _layer(rnn, units=128, return_sequences=False, dropout=0.2),
            _layer("dense", units=64, activation="relu"),
            _layer("dropout", rate=0.3),
        ]
    if arch == "transformer":
        return [
            _layer("embedding", input_dim=10000, output_dim=128),
            _layer("transformer_block", num_heads=4, ff_dim=256, dropout=0.1),
            _layer("transformer_block", num_heads=4, ff_dim=256, dropout=0.1),
            _layer("globalmaxpool1d"),
            _layer("dense", units=64, activation="relu"),
            _layer("dropout", rate=0.2),
        ]
    if arch == "lstm_ts":
        return [
            _layer("lstm", units=64, return_sequences=True),
            _layer("lstm", units=32, return_sequences=False),
            _layer("dense", units=32, activation="relu"),
        ]
    # fallback
    return [_layer("dense", units=64, activation="relu")]


def scaffold_spec(
    *,
    architecture: str,
    data_type: str,
    input_shape: list[int],
    n_classes: int | None,
    is_regression: bool = False,
) -> dict[str, Any]:
    """Build a complete, editable ModelSpec for an architecture preset."""
    arch = ARCH_BY_KEY.get(architecture)
    if not arch:
        raise ValueError(f"unknown architecture '{architecture}'")

    task = DATA_TYPE_TASK[data_type]
    if data_type == "tabular" and is_regression:
        task = "tabular_regression"

    layers = _body_layers(architecture, input_shape, n_classes)
    layers.append(_head(task, n_classes))

    return {
        "api": arch["api"],
        "task": task,
        "data_type": data_type,
        "architecture": architecture,
        "input_shape": list(input_shape),
        "n_classes": n_classes,
        "layers": layers,
        "compile": {
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "loss": default_loss(task, n_classes),
            "metrics": default_metrics(task, n_classes),
        },
        "training": {
            "epochs": 20,
            "batch_size": 32,
            "validation_split": 0.2,
            "early_stopping": True,
            "reduce_lr_on_plateau": True,
        },
    }
