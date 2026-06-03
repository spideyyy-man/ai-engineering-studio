"""DL Studio — building-block catalog.

Layer / optimizer / loss / metric definitions used by every other DL module:
the scaffolder seeds specs from here, the validator infers shapes from the
layer `kind`, the code generator emits Keras from the same `kind`, and the
frontend renders the param forms straight from these specs (the param-spec
shape is identical to ML Studio's HyperparamSpec, so the existing
HyperparamForm component works unchanged).
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------- layer catalog
#
# Each layer entry:
#   key            stable identifier used in a LayerSpec.type
#   label          human label
#   category       core | conv | recurrent | embedding | attention | regularize | shape | output
#   data_types     which task families this layer is offered for (UI filter)
#   description    one-liner
#   params         {name: HyperparamSpec} — same shape ML Studio uses
#   defaults       convenience default params (mirrors params[].default)
#
# The validator maps `key` -> shape rule; the codegen maps `key` -> Keras class.

LAYER_CATALOG: list[dict[str, Any]] = [
    # ---- core
    {
        "key": "dense", "label": "Dense", "category": "core",
        "data_types": ["tabular", "image", "text", "timeseries"],
        "description": "Fully-connected layer. Applies units neurons over the last axis.",
        "params": {
            "units": {"type": "int", "default": 64, "min": 1, "max": 8192, "step": 1, "label": "Units"},
            "activation": {"type": "select", "default": "relu", "label": "Activation",
                           "options": ["relu", "tanh", "sigmoid", "elu", "selu", "gelu", "swish", "linear"]},
        },
    },
    {
        "key": "activation", "label": "Activation", "category": "core",
        "data_types": ["tabular", "image", "text", "timeseries"],
        "description": "Standalone activation function (shape-preserving).",
        "params": {
            "activation": {"type": "select", "default": "relu", "label": "Activation",
                           "options": ["relu", "tanh", "sigmoid", "elu", "selu", "gelu", "swish", "softmax"]},
        },
    },
    # ---- regularization
    {
        "key": "dropout", "label": "Dropout", "category": "regularize",
        "data_types": ["tabular", "image", "text", "timeseries"],
        "description": "Randomly zeroes a fraction of inputs during training.",
        "params": {
            "rate": {"type": "float", "default": 0.2, "min": 0.0, "max": 0.9, "step": 0.05, "label": "Rate"},
        },
    },
    {
        "key": "batchnorm", "label": "BatchNormalization", "category": "regularize",
        "data_types": ["tabular", "image", "text", "timeseries"],
        "description": "Normalizes activations per batch — stabilizes & speeds training.",
        "params": {},
    },
    {
        "key": "layernorm", "label": "LayerNormalization", "category": "regularize",
        "data_types": ["text", "timeseries", "tabular"],
        "description": "Normalizes across features for each sample (used in Transformers).",
        "params": {},
    },
    # ---- convolution (image / 1D sequence)
    {
        "key": "conv2d", "label": "Conv2D", "category": "conv",
        "data_types": ["image"],
        "description": "2D convolution for images.",
        "params": {
            "filters": {"type": "int", "default": 32, "min": 1, "max": 2048, "step": 1, "label": "Filters"},
            "kernel_size": {"type": "int", "default": 3, "min": 1, "max": 11, "step": 1, "label": "Kernel size"},
            "strides": {"type": "int", "default": 1, "min": 1, "max": 4, "step": 1, "label": "Stride"},
            "padding": {"type": "select", "default": "same", "options": ["same", "valid"], "label": "Padding"},
            "activation": {"type": "select", "default": "relu",
                           "options": ["relu", "elu", "selu", "gelu", "swish", "linear"], "label": "Activation"},
        },
    },
    {
        "key": "maxpool2d", "label": "MaxPooling2D", "category": "conv",
        "data_types": ["image"],
        "description": "Down-samples spatial dims by taking the max in each window.",
        "params": {
            "pool_size": {"type": "int", "default": 2, "min": 2, "max": 8, "step": 1, "label": "Pool size"},
        },
    },
    {
        "key": "globalavgpool2d", "label": "GlobalAveragePooling2D", "category": "conv",
        "data_types": ["image"],
        "description": "Averages each feature map to one value — replaces Flatten before the head.",
        "params": {},
    },
    {
        "key": "conv1d", "label": "Conv1D", "category": "conv",
        "data_types": ["text", "timeseries"],
        "description": "1D convolution over a sequence.",
        "params": {
            "filters": {"type": "int", "default": 64, "min": 1, "max": 2048, "step": 1, "label": "Filters"},
            "kernel_size": {"type": "int", "default": 3, "min": 1, "max": 11, "step": 1, "label": "Kernel size"},
            "padding": {"type": "select", "default": "same", "options": ["same", "valid"], "label": "Padding"},
            "activation": {"type": "select", "default": "relu",
                           "options": ["relu", "elu", "gelu", "swish", "linear"], "label": "Activation"},
        },
    },
    {
        "key": "globalmaxpool1d", "label": "GlobalMaxPooling1D", "category": "conv",
        "data_types": ["text", "timeseries"],
        "description": "Takes the max over the time axis — collapses a sequence to a vector.",
        "params": {},
    },
    # ---- recurrent
    {
        "key": "lstm", "label": "LSTM", "category": "recurrent",
        "data_types": ["text", "timeseries"],
        "description": "Long Short-Term Memory recurrent layer.",
        "params": {
            "units": {"type": "int", "default": 64, "min": 1, "max": 2048, "step": 1, "label": "Units"},
            "return_sequences": {"type": "bool", "default": False, "label": "Return sequences"},
            "dropout": {"type": "float", "default": 0.0, "min": 0.0, "max": 0.9, "step": 0.05, "label": "Dropout"},
        },
    },
    {
        "key": "gru", "label": "GRU", "category": "recurrent",
        "data_types": ["text", "timeseries"],
        "description": "Gated Recurrent Unit — lighter than LSTM.",
        "params": {
            "units": {"type": "int", "default": 64, "min": 1, "max": 2048, "step": 1, "label": "Units"},
            "return_sequences": {"type": "bool", "default": False, "label": "Return sequences"},
            "dropout": {"type": "float", "default": 0.0, "min": 0.0, "max": 0.9, "step": 0.05, "label": "Dropout"},
        },
    },
    {
        "key": "simplernn", "label": "SimpleRNN", "category": "recurrent",
        "data_types": ["text", "timeseries"],
        "description": "Vanilla recurrent layer (rarely best — useful for teaching).",
        "params": {
            "units": {"type": "int", "default": 64, "min": 1, "max": 1024, "step": 1, "label": "Units"},
            "return_sequences": {"type": "bool", "default": False, "label": "Return sequences"},
        },
    },
    {
        "key": "bidirectional_lstm", "label": "Bidirectional LSTM", "category": "recurrent",
        "data_types": ["text", "timeseries"],
        "description": "LSTM that reads the sequence both forwards and backwards (doubles units out).",
        "params": {
            "units": {"type": "int", "default": 64, "min": 1, "max": 1024, "step": 1, "label": "Units"},
            "return_sequences": {"type": "bool", "default": False, "label": "Return sequences"},
        },
    },
    # ---- embedding
    {
        "key": "embedding", "label": "Embedding", "category": "embedding",
        "data_types": ["text"],
        "description": "Maps integer token ids to dense vectors. First layer for text.",
        "params": {
            "input_dim": {"type": "int", "default": 10000, "min": 2, "max": 500000, "step": 1, "label": "Vocab size"},
            "output_dim": {"type": "int", "default": 128, "min": 2, "max": 1024, "step": 1, "label": "Embed dim"},
        },
    },
    # ---- attention
    {
        "key": "transformer_block", "label": "Transformer Encoder Block", "category": "attention",
        "data_types": ["text", "timeseries"],
        "description": "Multi-head self-attention + feed-forward + residual norm (shape-preserving).",
        "params": {
            "num_heads": {"type": "int", "default": 4, "min": 1, "max": 32, "step": 1, "label": "Heads"},
            "ff_dim": {"type": "int", "default": 256, "min": 8, "max": 4096, "step": 1, "label": "Feed-forward dim"},
            "dropout": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.9, "step": 0.05, "label": "Dropout"},
        },
    },
    # ---- shape ops
    {
        "key": "flatten", "label": "Flatten", "category": "shape",
        "data_types": ["image", "text", "timeseries"],
        "description": "Flattens all non-batch dims into a single vector.",
        "params": {},
    },
    # ---- pretrained base
    {
        "key": "resnet50_base", "label": "ResNet50 (pretrained base)", "category": "conv",
        "data_types": ["image"],
        "description": "ImageNet-pretrained ResNet50 feature extractor (outputs a 2048-vector).",
        "params": {
            "trainable": {"type": "bool", "default": False, "label": "Fine-tune base"},
        },
    },
]

LAYER_BY_KEY: dict[str, dict[str, Any]] = {l["key"]: l for l in LAYER_CATALOG}


def list_layers(data_type: str | None = None) -> list[dict[str, Any]]:
    out = LAYER_CATALOG
    if data_type:
        out = [l for l in LAYER_CATALOG if data_type in l["data_types"]]
    # Strip nothing — frontend wants params for its forms.
    return [dict(l) for l in out]


def layer_defaults(kind: str) -> dict[str, Any]:
    spec = LAYER_BY_KEY.get(kind)
    if not spec:
        return {}
    return {k: v.get("default") for k, v in spec.get("params", {}).items()}


# ---------------------------------------------------------------- optimizers / losses / metrics

OPTIMIZERS: list[dict[str, Any]] = [
    {"key": "adam", "label": "Adam", "keras": "Adam",
     "description": "Adaptive moments — the sensible default for most networks."},
    {"key": "adamw", "label": "AdamW", "keras": "AdamW",
     "description": "Adam with decoupled weight decay — better generalization for Transformers."},
    {"key": "sgd", "label": "SGD (momentum)", "keras": "SGD",
     "description": "Stochastic gradient descent — classic, needs LR tuning."},
    {"key": "rmsprop", "label": "RMSprop", "keras": "RMSprop",
     "description": "Good for recurrent nets."},
    {"key": "nadam", "label": "Nadam", "keras": "Nadam",
     "description": "Adam + Nesterov momentum."},
]
OPTIMIZER_KEYS = {o["key"] for o in OPTIMIZERS}

# Losses are auto-selected by task but overridable.
LOSSES_BY_TASK: dict[str, list[dict[str, str]]] = {
    "binary_classification": [
        {"key": "binary_crossentropy", "keras": "binary_crossentropy", "label": "Binary cross-entropy"},
        {"key": "hinge", "keras": "hinge", "label": "Hinge"},
    ],
    "multiclass_classification": [
        {"key": "sparse_categorical_crossentropy", "keras": "sparse_categorical_crossentropy",
         "label": "Sparse categorical cross-entropy (integer labels)"},
        {"key": "categorical_crossentropy", "keras": "categorical_crossentropy",
         "label": "Categorical cross-entropy (one-hot labels)"},
    ],
    "regression": [
        {"key": "mse", "keras": "mse", "label": "Mean squared error"},
        {"key": "mae", "keras": "mae", "label": "Mean absolute error"},
        {"key": "huber", "keras": "huber", "label": "Huber (robust)"},
    ],
}

METRICS_BY_TASK: dict[str, list[str]] = {
    "binary_classification": ["accuracy", "AUC", "Precision", "Recall"],
    "multiclass_classification": ["accuracy"],
    "regression": ["mae", "mse", "RootMeanSquaredError"],
}


def output_kind(task: str, n_classes: int | None) -> str:
    """Normalize a DL task family to one of: binary_classification /
    multiclass_classification / regression. Drives default loss/metrics/head."""
    if task in ("tabular_regression", "timeseries_forecast"):
        return "regression"
    # classification family
    if n_classes is not None and n_classes <= 2:
        return "binary_classification"
    return "multiclass_classification"


def default_loss(task: str, n_classes: int | None) -> str:
    ok = output_kind(task, n_classes)
    return LOSSES_BY_TASK[ok][0]["key"]


def default_metrics(task: str, n_classes: int | None) -> list[str]:
    ok = output_kind(task, n_classes)
    return list(METRICS_BY_TASK[ok])


def catalog_payload() -> dict[str, Any]:
    """Everything the frontend needs to render builder dropdowns."""
    return {
        "layers": list_layers(),
        "optimizers": OPTIMIZERS,
        "losses_by_task": LOSSES_BY_TASK,
        "metrics_by_task": METRICS_BY_TASK,
    }
