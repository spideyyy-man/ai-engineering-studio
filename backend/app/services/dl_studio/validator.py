"""DL Studio — static shape inference & validation.

Pure Python (no TensorFlow): walks the layer stack, infers the output shape of
each layer, estimates its parameter count, and surfaces errors/warnings *before*
training. This is what makes the builder developer-friendly — a shape mismatch
is caught at design time with a plain-language message, not as a stack trace
three minutes into a training run.

Shapes are tuples of ints WITHOUT the batch axis, e.g. a tabular row with 8
features is (8,); a 28x28 RGB image is (28, 28, 3); a length-100 token sequence
is (100,).
"""
from __future__ import annotations

import math
from typing import Any

from .catalog import LAYER_BY_KEY


class ShapeError(ValueError):
    pass


def _prod(shape: tuple[int, ...]) -> int:
    out = 1
    for d in shape:
        out *= d
    return out


def _conv_out(size: int, kernel: int, stride: int, padding: str) -> int:
    if padding == "same":
        return math.ceil(size / stride)
    # valid
    return math.floor((size - kernel) / stride) + 1


# ---------------------------------------------------------------- per-layer shape rules

def _infer_layer(kind: str, params: dict[str, Any], shp: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    """Return (output_shape, trainable_param_estimate). Raises ShapeError on misuse."""
    p = params or {}

    if kind == "dense":
        units = int(p.get("units", 64))
        if len(shp) == 0:
            raise ShapeError("Dense needs at least a 1-D input.")
        in_dim = shp[-1]
        out = shp[:-1] + (units,)
        params_n = (in_dim + 1) * units
        return out, params_n

    if kind in ("activation", "dropout", "batchnorm", "layernorm"):
        pn = 0
        if kind == "batchnorm" and shp:
            pn = 4 * shp[-1]  # gamma, beta (trainable) + moving mean/var
        if kind == "layernorm" and shp:
            pn = 2 * shp[-1]
        return shp, pn

    if kind == "flatten":
        if not shp:
            raise ShapeError("Flatten needs a multi-dimensional input.")
        return (_prod(shp),), 0

    if kind == "conv2d":
        if len(shp) != 3:
            raise ShapeError(
                f"Conv2D expects a 3-D input (H, W, C) but got {shp}. "
                "Add it before any Flatten/Dense, on image data."
            )
        h, w, c = shp
        f = int(p.get("filters", 32))
        k = int(p.get("kernel_size", 3))
        s = int(p.get("strides", 1))
        pad = p.get("padding", "same")
        oh, ow = _conv_out(h, k, s, pad), _conv_out(w, k, s, pad)
        if oh <= 0 or ow <= 0:
            raise ShapeError(f"Conv2D kernel {k}/stride {s} shrinks {shp} below 1 — reduce kernel or use 'same'.")
        return (oh, ow, f), (k * k * c + 1) * f

    if kind == "maxpool2d":
        if len(shp) != 3:
            raise ShapeError(f"MaxPooling2D expects (H, W, C) but got {shp}.")
        h, w, c = shp
        pool = int(p.get("pool_size", 2))
        return (h // pool, w // pool, c), 0

    if kind == "globalavgpool2d":
        if len(shp) != 3:
            raise ShapeError(f"GlobalAveragePooling2D expects (H, W, C) but got {shp}.")
        return (shp[2],), 0

    if kind == "conv1d":
        if len(shp) != 2:
            raise ShapeError(f"Conv1D expects (timesteps, channels) but got {shp}.")
        t, c = shp
        f = int(p.get("filters", 64))
        k = int(p.get("kernel_size", 3))
        pad = p.get("padding", "same")
        ot = _conv_out(t, k, 1, pad)
        return (ot, f), (k * c + 1) * f

    if kind == "globalmaxpool1d":
        if len(shp) != 2:
            raise ShapeError(f"GlobalMaxPooling1D expects (timesteps, features) but got {shp}.")
        return (shp[1],), 0

    if kind in ("lstm", "gru", "simplernn", "bidirectional_lstm"):
        if len(shp) != 2:
            raise ShapeError(
                f"{LAYER_BY_KEY[kind]['label']} expects a 2-D sequence input "
                f"(timesteps, features) but got {shp}. Use an Embedding first for text."
            )
        t, feat = shp
        units = int(p.get("units", 64))
        ret = bool(p.get("return_sequences", False))
        mult = 2 if kind == "bidirectional_lstm" else 1
        gate = {"lstm": 4, "gru": 3, "simplernn": 1, "bidirectional_lstm": 4}[kind]
        # rough param estimate (per direction): gate*(units*(feat+units)+units)
        pn = mult * gate * (units * (feat + units) + units)
        out_units = units * mult
        return ((t, out_units) if ret else (out_units,)), pn

    if kind == "embedding":
        if len(shp) != 1:
            raise ShapeError(f"Embedding expects a 1-D integer sequence (timesteps,) but got {shp}.")
        vocab = int(p.get("input_dim", 10000))
        dim = int(p.get("output_dim", 128))
        return (shp[0], dim), vocab * dim

    if kind == "transformer_block":
        if len(shp) != 2:
            raise ShapeError(f"Transformer block expects (timesteps, features) but got {shp}.")
        t, d = shp
        ff = int(p.get("ff_dim", 256))
        heads = int(p.get("num_heads", 4))
        # attention proj (4*d*d) + ffn (2*d*ff) + norms — rough
        pn = 4 * d * d + 2 * d * ff + 4 * d
        if d % heads != 0:
            raise ShapeError(
                f"Transformer feature dim {d} must be divisible by num_heads {heads}. "
                "Adjust the Embedding output_dim or the head count."
            )
        return (t, d), pn

    if kind == "resnet50_base":
        if len(shp) != 3:
            raise ShapeError(f"ResNet50 base expects an image (H, W, 3) but got {shp}.")
        return (2048,), 23_587_712  # known ResNet50 backbone param count

    raise ShapeError(f"unknown layer type '{kind}'")


# ---------------------------------------------------------------- public API

def validate_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Walk the layer stack and return a structured validation report.

    Report shape:
      {
        ok: bool,
        input_shape, output_shape,
        total_params, total_params_human,
        layers: [{index, type, label, output_shape, params, error}],
        errors: [str], warnings: [str]
      }
    """
    input_shape = tuple(int(d) for d in spec.get("input_shape", []))
    layers = spec.get("layers", []) or []
    task = spec.get("task", "")
    n_classes = spec.get("n_classes")

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    total_params = 0

    shp = input_shape
    fatal = False
    for i, layer in enumerate(layers):
        kind = layer.get("type")
        label = LAYER_BY_KEY.get(kind, {}).get("label", kind)
        row: dict[str, Any] = {
            "index": i, "type": kind, "label": label,
            "input_shape": list(shp), "output_shape": None, "params": 0, "error": None,
        }
        if fatal:
            row["error"] = "skipped (earlier layer failed)"
            rows.append(row)
            continue
        if kind not in LAYER_BY_KEY:
            row["error"] = f"unknown layer type '{kind}'"
            errors.append(f"Layer {i + 1}: unknown type '{kind}'")
            fatal = True
            rows.append(row)
            continue
        try:
            shp, pn = _infer_layer(kind, layer.get("params", {}), shp)
            row["output_shape"] = list(shp)
            row["params"] = int(pn)
            total_params += int(pn)
        except ShapeError as e:
            row["error"] = str(e)
            errors.append(f"Layer {i + 1} ({label}): {e}")
            fatal = True
        rows.append(row)

    # ---- head / task sanity checks (only if the stack itself is valid)
    if not fatal and layers:
        last = layers[-1]
        if last.get("type") != "dense":
            warnings.append("Last layer is usually a Dense output head matching your task.")
        else:
            units = int(last.get("params", {}).get("units", 0))
            act = last.get("params", {}).get("activation")
            if task == "tabular_regression" or task == "timeseries_forecast":
                if units != 1:
                    warnings.append(f"Regression output should have units=1 (got {units}).")
                if act not in ("linear", None):
                    warnings.append(f"Regression output should use 'linear' activation (got '{act}').")
            elif n_classes is not None:
                if n_classes <= 2:
                    if units != 1 or act != "sigmoid":
                        warnings.append("Binary classification head should be Dense(1, 'sigmoid').")
                else:
                    if units != n_classes:
                        warnings.append(
                            f"Multiclass head units ({units}) should equal n_classes ({n_classes})."
                        )
                    if act != "softmax":
                        warnings.append("Multiclass head should use 'softmax' activation.")

    if not layers:
        errors.append("Model has no layers — add at least an output layer.")

    return {
        "ok": len(errors) == 0,
        "api": spec.get("api", "sequential"),
        "input_shape": list(input_shape),
        "output_shape": list(shp) if not fatal and layers else None,
        "total_params": int(total_params),
        "total_params_human": _human(total_params),
        "layers": rows,
        "errors": errors,
        "warnings": warnings,
    }


def _human(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)
