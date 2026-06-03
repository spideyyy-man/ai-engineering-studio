"""Tab 2 — ML Studio. Catalog + detect-task + train + downloads."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..core.errors import NotFoundError, ValidationError
from ..core.paths import model_path
from ..database import get_db
from ..models import Dataset, MLModel
from ..schemas import (
    AlgorithmInfo,
    DetectTaskRequest,
    DetectTaskResponse,
    MLRunRead,
    MLTrainRequest,
    MLTrainResponse,
)
from ..services.codegen import render_template
from ..services.data_engineering import load_dataframe
from ..services.ml_studio import detect_task, list_algorithms, train_model

router = APIRouter(prefix="/ml", tags=["ml"])


# ---------------------------------------------------------------- catalog

@router.get("/algorithms", response_model=list[AlgorithmInfo])
def get_algorithms(task: str | None = Query(None)):
    return list_algorithms(task)


@router.post("/detect-task", response_model=DetectTaskResponse)
def detect(payload: DetectTaskRequest, db: Session = Depends(get_db)):
    ds = db.get(Dataset, payload.dataset_id)
    if not ds:
        raise NotFoundError("dataset")
    df = load_dataframe(ds.storage_path)
    task = detect_task(df, payload.target_col)
    if task == "clustering":
        rationale = "No target column provided — defaulting to unsupervised clustering."
    elif task == "classification":
        s = df[payload.target_col]
        rationale = (
            f"Target '{payload.target_col}' has dtype {s.dtype} with "
            f"{int(s.nunique(dropna=True))} unique values — classification."
        )
    else:
        s = df[payload.target_col]
        rationale = (
            f"Target '{payload.target_col}' has dtype {s.dtype} with "
            f"{int(s.nunique(dropna=True))} unique values — regression."
        )
    return DetectTaskResponse(task_type=task, rationale=rationale)


# ---------------------------------------------------------------- train

@router.post("/train", response_model=MLTrainResponse)
def train(payload: MLTrainRequest, db: Session = Depends(get_db)):
    ds = db.get(Dataset, payload.dataset_id)
    if not ds:
        raise NotFoundError("dataset")
    if ds.project_id != payload.project_id:
        raise ValidationError("dataset does not belong to project")

    df = load_dataframe(ds.storage_path)

    # Create the DB row first so we have a stable model_id for paths.
    row = MLModel(
        project_id=payload.project_id,
        dataset_id=payload.dataset_id,
        task_type="pending",
        algorithm=payload.algorithm,
        hyperparams_json=payload.hyperparams or {},
        metrics_json={},
        status="running",
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    out_dir = model_path(payload.project_id, row.id)
    pkl_path = out_dir / "model.pkl"
    sample_path = out_dir / "sample_usage.py"

    try:
        result = train_model(
            df,
            target_col=payload.target_col,
            algorithm=payload.algorithm,
            hyperparams=payload.hyperparams,
            test_size=payload.test_size,
            random_state=payload.random_state,
            pickle_path=pkl_path,
            cv_strategy=payload.cv_strategy,
            cv_folds=payload.cv_folds,
            hpt_mode=payload.hpt_mode,
            hpt_n_iter=payload.hpt_n_iter,
            hpt_scoring=payload.hpt_scoring,
        )
    except Exception as e:
        row.status = "failed"
        row.metrics_json = {"error": str(e)}
        db.commit()
        raise ValidationError(f"training failed: {e}")

    # Render sample usage code
    sample_code = render_template(
        "ml_studio/usage_sample.py.j2",
        {
            "algorithm_label": result["algorithm_label"],
            "task": result["task"],
            "dataset_name": ds.name,
            "target_column": result["target_column"],
            "n_train": result["n_train"],
            "n_test": result["n_test"],
            "hparams_effective": result["hparams_effective"],
            "feature_columns": result["feature_columns"],
        },
    )
    sample_path.write_text(sample_code, encoding="utf-8")

    row.task_type = result["task"]
    row.metrics_json = {
        "metrics": result["metrics"],
        "figures_keys": list(result["figures"].keys()),
        "feature_columns": result["feature_columns"],
        "target_column": result["target_column"],
        "n_train": result["n_train"],
        "n_test": result["n_test"],
        "hparams_effective": result["hparams_effective"],
        "algorithm_label": result["algorithm_label"],
    }
    row.artifact_path = str(pkl_path)
    row.code_path = str(sample_path)
    row.status = "succeeded"
    db.commit()

    return MLTrainResponse(
        model_id=row.id,
        task=result["task"],
        algorithm=result["algorithm"],
        algorithm_label=result["algorithm_label"],
        metrics=result["metrics"],
        figures=result["figures"],
        hparams_effective=result["hparams_effective"],
        feature_columns=result["feature_columns"],
        target_column=result["target_column"],
        n_train=result["n_train"],
        n_test=result["n_test"],
        pickle_filename="model.pkl",
        pickle_download_url=f"/api/v1/ml/models/{row.id}/download?fmt=pkl",
        sample_filename="sample_usage.py",
        sample_code=sample_code,
        sample_download_url=f"/api/v1/ml/models/{row.id}/download?fmt=sample",
        preprocessing_applied=result.get("preprocessing_applied"),
        cv_used=result.get("cv_used", False),
        cv_strategy=result.get("cv_strategy"),
        cv_folds=result.get("cv_folds"),
        cv_mean_score=result.get("cv_mean_score"),
        cv_std_score=result.get("cv_std_score"),
        cv_scoring=result.get("cv_scoring"),
        hpt_mode=result.get("hpt_mode"),
        hpt_best_params=result.get("hpt_best_params"),
        hpt_best_score=result.get("hpt_best_score"),
        hpt_cv_results=result.get("hpt_cv_results"),
    )


# ---------------------------------------------------------------- registry

@router.get("/models", response_model=list[MLRunRead])
def list_models(project_id: str | None = Query(None), db: Session = Depends(get_db)):
    q = db.query(MLModel)
    if project_id:
        q = q.filter(MLModel.project_id == project_id)
    return q.order_by(MLModel.created_at.desc()).all()


@router.get("/models/{model_id}", response_model=MLRunRead)
def get_model(model_id: str, db: Session = Depends(get_db)):
    row = db.get(MLModel, model_id)
    if not row:
        raise NotFoundError("model")
    return row


@router.get("/models/{model_id}/download")
def download_model(model_id: str, fmt: str = Query("pkl"), db: Session = Depends(get_db)):
    row = db.get(MLModel, model_id)
    if not row:
        raise NotFoundError("model")
    if fmt == "pkl":
        p = Path(row.artifact_path or "")
        if not p.exists():
            raise NotFoundError("pickle artifact")
        return FileResponse(p, media_type="application/octet-stream", filename="model.pkl")
    if fmt == "sample":
        p = Path(row.code_path or "")
        if not p.exists():
            raise NotFoundError("sample code")
        return FileResponse(p, media_type="text/x-python", filename="sample_usage.py")
    raise ValidationError(f"unknown fmt '{fmt}' — expected pkl or sample")
