"""DL Studio — TensorFlow model builder + tabular trainer.

Everything in this module is *gated* on TensorFlow being importable. The rest of
the studio (scaffold / validate / codegen) works without it; only live in-studio
training needs `pip install tensorflow`. When TF is missing, `tf_available()`
returns False and the router returns a clean, actionable message instead of a
crash.

In-studio training is wired for TABULAR data (classification & regression),
reusing the same auto-preprocessing philosophy as ML Studio. Image/text/time-
series models can be designed, validated and exported as code here; their
training runs against the generated script with your own data pipeline.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def tf_available() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except Exception:
        return False


def _safe_float(x) -> float | None:
    try:
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else v
    except (TypeError, ValueError):
        return None


def _fig_to_json(fig: go.Figure) -> dict[str, Any]:
    return json.loads(fig.to_json())


# ---------------------------------------------------------------- keras build

def build_keras_model(spec: dict[str, Any]):
    """Construct & compile a tf.keras model from a ModelSpec. TF required."""
    from tensorflow import keras
    from tensorflow.keras import layers as L

    api = spec.get("api", "sequential")
    input_shape = tuple(spec.get("input_shape", []))
    comp = spec.get("compile", {})

    def make_layer(layer: dict[str, Any]):
        kind = layer["type"]
        p = layer.get("params", {}) or {}
        if kind == "dense":
            return L.Dense(int(p.get("units", 64)), activation=p.get("activation", "relu"))
        if kind == "activation":
            return L.Activation(p.get("activation", "relu"))
        if kind == "dropout":
            return L.Dropout(float(p.get("rate", 0.2)))
        if kind == "batchnorm":
            return L.BatchNormalization()
        if kind == "layernorm":
            return L.LayerNormalization()
        if kind == "flatten":
            return L.Flatten()
        if kind == "conv2d":
            return L.Conv2D(int(p.get("filters", 32)), int(p.get("kernel_size", 3)),
                            strides=int(p.get("strides", 1)), padding=p.get("padding", "same"),
                            activation=p.get("activation", "relu"))
        if kind == "maxpool2d":
            return L.MaxPooling2D(pool_size=int(p.get("pool_size", 2)))
        if kind == "globalavgpool2d":
            return L.GlobalAveragePooling2D()
        if kind == "conv1d":
            return L.Conv1D(int(p.get("filters", 64)), int(p.get("kernel_size", 3)),
                            padding=p.get("padding", "same"), activation=p.get("activation", "relu"))
        if kind == "globalmaxpool1d":
            return L.GlobalMaxPooling1D()
        if kind == "lstm":
            return L.LSTM(int(p.get("units", 64)), return_sequences=bool(p.get("return_sequences", False)),
                          dropout=float(p.get("dropout", 0.0)))
        if kind == "gru":
            return L.GRU(int(p.get("units", 64)), return_sequences=bool(p.get("return_sequences", False)),
                         dropout=float(p.get("dropout", 0.0)))
        if kind == "simplernn":
            return L.SimpleRNN(int(p.get("units", 64)), return_sequences=bool(p.get("return_sequences", False)))
        if kind == "bidirectional_lstm":
            return L.Bidirectional(L.LSTM(int(p.get("units", 64)),
                                          return_sequences=bool(p.get("return_sequences", False))))
        if kind == "embedding":
            return L.Embedding(int(p.get("input_dim", 10000)), int(p.get("output_dim", 128)))
        if kind == "resnet50_base":
            base = keras.applications.ResNet50(include_top=False, weights="imagenet", pooling="avg")
            base.trainable = bool(p.get("trainable", False))
            return base
        raise ValueError(f"layer '{kind}' is not buildable in-studio")

    def transformer_block(x, p):
        dim = x.shape[-1]
        heads = int(p.get("num_heads", 4))
        attn = L.MultiHeadAttention(num_heads=heads, key_dim=max(1, dim // heads))(x, x)
        attn = L.Dropout(float(p.get("dropout", 0.1)))(attn)
        x = L.LayerNormalization(epsilon=1e-6)(x + attn)
        ff = L.Dense(int(p.get("ff_dim", 256)), activation="relu")(x)
        ff = L.Dense(dim)(ff)
        ff = L.Dropout(float(p.get("dropout", 0.1)))(ff)
        return L.LayerNormalization(epsilon=1e-6)(x + ff)

    layers_spec = spec.get("layers", []) or []
    if api == "functional":
        inputs = keras.Input(shape=input_shape)
        x = inputs
        for layer in layers_spec:
            if layer["type"] == "transformer_block":
                x = transformer_block(x, layer.get("params", {}))
            else:
                x = make_layer(layer)(x)
        model = keras.Model(inputs, x)
    else:
        model = keras.Sequential([keras.Input(shape=input_shape)])
        for layer in layers_spec:
            if layer["type"] == "transformer_block":
                raise ValueError("transformer_block requires the Functional API.")
            model.add(make_layer(layer))

    opt_map = {"adam": keras.optimizers.Adam, "adamw": getattr(keras.optimizers, "AdamW", keras.optimizers.Adam),
               "sgd": keras.optimizers.SGD, "rmsprop": keras.optimizers.RMSprop,
               "nadam": keras.optimizers.Nadam}
    opt_cls = opt_map.get(comp.get("optimizer", "adam"), keras.optimizers.Adam)
    model.compile(
        optimizer=opt_cls(learning_rate=float(comp.get("learning_rate", 1e-3))),
        loss=comp.get("loss", "mse"),
        metrics=list(comp.get("metrics", ["accuracy"])),
    )
    return model


# ---------------------------------------------------------------- tabular trainer

def train_tabular(
    df: pd.DataFrame,
    *,
    spec: dict[str, Any],
    target_col: str,
    save_dir: Path,
    random_state: int = 42,
) -> dict[str, Any]:
    """Preprocess → fit → evaluate a tabular DL model. TF required.

    Saves `model.keras` + `preprocess.pkl` into save_dir. Returns metrics,
    Plotly figures (learning curves + diagnostics), and history.
    """
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
    from sklearn.model_selection import train_test_split

    task = spec.get("task", "tabular_classification")
    is_reg = task == "tabular_regression"

    if target_col not in df.columns:
        raise ValueError(f"target column '{target_col}' not in dataset")
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    if df.empty:
        raise ValueError("no rows left after dropping missing target values")

    X = df.drop(columns=[target_col])
    y_raw = df[target_col]

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
    ], remainder="drop")

    Xt = pre.fit_transform(X)
    Xt = np.asarray(Xt, dtype="float32")
    n_features = Xt.shape[1]

    # Target encoding
    label_encoder = None
    if is_reg:
        y = y_raw.astype("float32").to_numpy()
        n_classes = None
    else:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.astype(str)).astype("int64")
        n_classes = int(len(label_encoder.classes_))

    # The spec was scaffolded with a placeholder input shape; bind it to the
    # real post-preprocessing feature count so the built model matches the data.
    spec = {**spec, "input_shape": [n_features], "n_classes": n_classes}

    stratify = None if is_reg else y
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xt, y, test_size=0.2, random_state=random_state, stratify=stratify
    )

    model = build_keras_model(spec)

    from tensorflow import keras

    train_cfg = spec.get("training", {})
    callbacks = []
    if train_cfg.get("early_stopping", True):
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True))
    if train_cfg.get("reduce_lr_on_plateau", True):
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6))

    hist = model.fit(
        X_tr, y_tr,
        validation_split=float(train_cfg.get("validation_split", 0.2)),
        epochs=int(train_cfg.get("epochs", 20)),
        batch_size=int(train_cfg.get("batch_size", 32)),
        callbacks=callbacks,
        verbose=0,
    )
    history = {k: [_safe_float(v) for v in vals] for k, vals in hist.history.items()}

    # ---- Evaluate on held-out test set
    metrics: dict[str, Any] = {}
    figures: dict[str, Any] = {}
    preds_raw = model.predict(X_te, verbose=0)

    if is_reg:
        preds = preds_raw.reshape(-1)
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        rmse = math.sqrt(mean_squared_error(y_te, preds))
        metrics = {
            "r2": _safe_float(r2_score(y_te, preds)),
            "mae": _safe_float(mean_absolute_error(y_te, preds)),
            "mse": _safe_float(mean_squared_error(y_te, preds)),
            "rmse": _safe_float(rmse),
        }
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_te.tolist(), y=preds.tolist(), mode="markers",
                                 marker=dict(opacity=0.6), name="pred"))
        lo, hi = float(np.min(y_te)), float(np.max(y_te))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                 line=dict(dash="dash"), name="ideal"))
        fig.update_layout(title="Predicted vs Actual", height=360,
                          xaxis_title="actual", yaxis_title="predicted",
                          margin=dict(l=40, r=20, t=40, b=40))
        figures["pred_vs_actual"] = _fig_to_json(fig)
    else:
        if n_classes <= 2:
            pred_cls = (preds_raw.reshape(-1) >= 0.5).astype("int64")
        else:
            pred_cls = np.argmax(preds_raw, axis=1)
        from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                     precision_score, recall_score)
        avg = "binary" if n_classes == 2 else "weighted"
        metrics = {
            "accuracy": _safe_float(accuracy_score(y_te, pred_cls)),
            "precision": _safe_float(precision_score(y_te, pred_cls, average=avg, zero_division=0)),
            "recall": _safe_float(recall_score(y_te, pred_cls, average=avg, zero_division=0)),
            "f1": _safe_float(f1_score(y_te, pred_cls, average=avg, zero_division=0)),
            "n_classes": n_classes,
        }
        labels = list(range(n_classes))
        cm = confusion_matrix(y_te, pred_cls, labels=labels)
        class_names = [str(c) for c in (label_encoder.classes_ if label_encoder is not None else labels)]
        fig = go.Figure(data=go.Heatmap(
            z=cm, x=[f"pred:{c}" for c in class_names], y=[f"true:{c}" for c in class_names],
            colorscale="Blues", text=cm, texttemplate="%{text}"))
        fig.update_layout(title="Confusion matrix", height=380,
                          margin=dict(l=40, r=20, t=40, b=40))
        figures["confusion_matrix"] = _fig_to_json(fig)

    # ---- Learning curves
    loss_fig = go.Figure()
    if "loss" in history:
        loss_fig.add_trace(go.Scatter(y=history["loss"], mode="lines", name="loss"))
    if "val_loss" in history:
        loss_fig.add_trace(go.Scatter(y=history["val_loss"], mode="lines", name="val_loss"))
    loss_fig.update_layout(title="Training loss", height=320, xaxis_title="epoch",
                           margin=dict(l=40, r=20, t=40, b=40))
    figures["loss_curve"] = _fig_to_json(loss_fig)

    metric_keys = [k for k in history if k not in ("loss", "val_loss") and not k.startswith("val_")]
    if metric_keys:
        mk = metric_keys[0]
        mfig = go.Figure()
        mfig.add_trace(go.Scatter(y=history[mk], mode="lines", name=mk))
        if f"val_{mk}" in history:
            mfig.add_trace(go.Scatter(y=history[f"val_{mk}"], mode="lines", name=f"val_{mk}"))
        mfig.update_layout(title=f"Training {mk}", height=320, xaxis_title="epoch",
                           margin=dict(l=40, r=20, t=40, b=40))
        figures["metric_curve"] = _fig_to_json(mfig)

    # ---- Persist artifacts
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "model.keras"
    model.save(model_path)
    joblib.dump(pre, save_dir / "preprocess.pkl")
    if label_encoder is not None:
        joblib.dump(label_encoder, save_dir / "label_encoder.pkl")

    return {
        "task": task,
        "metrics": metrics,
        "figures": figures,
        "history": history,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "n_features": int(n_features),
        "n_classes": n_classes,
        "epochs_ran": len(history.get("loss", [])),
        "feature_columns": [str(c) for c in X.columns],
        "target_column": target_col,
        "class_names": [str(c) for c in label_encoder.classes_] if label_encoder is not None else None,
        "model_path": str(model_path),
        "effective_input_shape": [n_features],
    }
