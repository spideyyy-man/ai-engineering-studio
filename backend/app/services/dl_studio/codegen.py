"""DL Studio — Keras source generation from a ModelSpec.

Emits a runnable, self-contained `model.py` for either the Sequential or the
Functional API. Always available (no TensorFlow needed to *generate* the code),
so the studio is useful as a pure design + codegen tool even before TF is
installed.
"""
from __future__ import annotations

from typing import Any

from ..codegen import render_template
from .catalog import OPTIMIZERS

_OPT_KERAS = {o["key"]: o["keras"] for o in OPTIMIZERS}


def _kw(params: dict[str, Any], *keys: str) -> str:
    """Render selected params as keras kwargs, skipping Nones."""
    parts = []
    for k in keys:
        if k in params and params[k] is not None:
            v = params[k]
            parts.append(f"{k}={v!r}" if isinstance(v, str) else f"{k}={v}")
    return ", ".join(parts)


def _layer_call(layer: dict[str, Any]) -> str | list[str]:
    """Map one LayerSpec to a Keras constructor expression (or a small block
    for composite layers like the transformer encoder)."""
    kind = layer["type"]
    p = layer.get("params", {}) or {}
    if kind == "dense":
        return f"layers.Dense({_kw(p, 'units', 'activation')})"
    if kind == "activation":
        return f"layers.Activation({p.get('activation', 'relu')!r})"
    if kind == "dropout":
        return f"layers.Dropout({p.get('rate', 0.2)})"
    if kind == "batchnorm":
        return "layers.BatchNormalization()"
    if kind == "layernorm":
        return "layers.LayerNormalization()"
    if kind == "flatten":
        return "layers.Flatten()"
    if kind == "conv2d":
        return f"layers.Conv2D({_kw(p, 'filters', 'kernel_size', 'strides', 'padding', 'activation')})"
    if kind == "maxpool2d":
        return f"layers.MaxPooling2D(pool_size={p.get('pool_size', 2)})"
    if kind == "globalavgpool2d":
        return "layers.GlobalAveragePooling2D()"
    if kind == "conv1d":
        return f"layers.Conv1D({_kw(p, 'filters', 'kernel_size', 'padding', 'activation')})"
    if kind == "globalmaxpool1d":
        return "layers.GlobalMaxPooling1D()"
    if kind == "lstm":
        return f"layers.LSTM({_kw(p, 'units', 'return_sequences', 'dropout')})"
    if kind == "gru":
        return f"layers.GRU({_kw(p, 'units', 'return_sequences', 'dropout')})"
    if kind == "simplernn":
        return f"layers.SimpleRNN({_kw(p, 'units', 'return_sequences')})"
    if kind == "bidirectional_lstm":
        return f"layers.Bidirectional(layers.LSTM({_kw(p, 'units', 'return_sequences')}))"
    if kind == "embedding":
        return f"layers.Embedding({_kw(p, 'input_dim', 'output_dim')})"
    if kind == "globalmaxpool1d":
        return "layers.GlobalMaxPooling1D()"
    if kind == "resnet50_base":
        return (
            "tf.keras.applications.ResNet50(include_top=False, weights='imagenet', "
            f"pooling='avg')  # trainable={bool(p.get('trainable', False))}"
        )
    if kind == "transformer_block":
        # Composite — emitted as a helper call; helper is defined in the template.
        return (
            f"transformer_encoder(num_heads={p.get('num_heads', 4)}, "
            f"ff_dim={p.get('ff_dim', 256)}, dropout={p.get('dropout', 0.1)})"
        )
    return f"layers.Layer()  # TODO: unmapped layer '{kind}'"


def _uses_transformer(spec: dict[str, Any]) -> bool:
    return any(l["type"] == "transformer_block" for l in spec.get("layers", []))


def generate_code(spec: dict[str, Any]) -> str:
    api = spec.get("api", "sequential")
    input_shape = tuple(spec.get("input_shape", []))
    layers = spec.get("layers", []) or []
    comp = spec.get("compile", {})
    train = spec.get("training", {})

    opt_keras = _OPT_KERAS.get(comp.get("optimizer", "adam"), "Adam")
    lr = comp.get("learning_rate", 1e-3)
    loss = comp.get("loss", "mse")
    metrics = comp.get("metrics", ["accuracy"])

    if api == "functional":
        body_lines = [f"inputs = keras.Input(shape={input_shape})", "x = inputs"]
        for layer in layers:
            call = _layer_call(layer)
            if layer["type"] == "resnet50_base":
                body_lines.append(f"x = {call}(x)")
            else:
                body_lines.append(f"x = {call}(x)")
        body_lines.append("outputs = x")
        body_lines.append("model = keras.Model(inputs, outputs)")
        build_body = "\n    ".join(body_lines)
    else:
        seq_lines = [f"layers.Input(shape={input_shape}),"]
        for layer in layers:
            seq_lines.append(f"{_layer_call(layer)},")
        joined = "\n        ".join(seq_lines)
        build_body = f"model = keras.Sequential([\n        {joined}\n    ])"

    return render_template(
        "dl_studio/keras_model.py.j2",
        {
            "api": api,
            "build_body": build_body,
            "needs_transformer": _uses_transformer(spec),
            "optimizer_keras": opt_keras,
            "learning_rate": lr,
            "loss": loss,
            "metrics": metrics,
            "epochs": train.get("epochs", 20),
            "batch_size": train.get("batch_size", 32),
            "validation_split": train.get("validation_split", 0.2),
            "early_stopping": train.get("early_stopping", True),
            "reduce_lr": train.get("reduce_lr_on_plateau", True),
            "task": spec.get("task", ""),
            "data_type": spec.get("data_type", ""),
            "input_shape": list(input_shape),
        },
    )
