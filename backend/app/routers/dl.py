"""Tab 3 — DL Studio.

Endpoints:
  GET  /dl/architectures   preset catalog (optionally filtered by data_type)
  GET  /dl/layers          layer building-block catalog
  GET  /dl/catalog         layers + optimizers + losses + metrics (for builder UI)
  POST /dl/scaffold        preset -> full editable ModelSpec
  POST /dl/validate        static shape inference + warnings (no TF needed)
  POST /dl/codegen         ModelSpec -> runnable Keras source (no TF needed)
  POST /dl/train           build + train (tabular, TF-gated) -> metrics + artifacts
  GET  /dl/models          trained DL model registry
  GET  /dl/models/{id}     one DL model
  GET  /dl/models/{id}/download?fmt=model|code
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..core.errors import NotFoundError, ValidationError
from ..core.paths import model_path
from ..database import get_db
from ..models import Dataset, DLModel
from ..schemas import (
    DLCodegenRequest,
    DLCodegenResponse,
    DLRunRead,
    DLTrainRequest,
    DLTrainResponse,
    ScaffoldRequest,
    ValidateRequest,
    ValidationReport,
)
from ..services.data_engineering import load_dataframe
from ..services.dl_studio import (
    catalog_payload,
    generate_code,
    list_architectures,
    list_layers,
    scaffold_spec,
    tf_available,
    train_tabular,
    validate_spec,
)

router = APIRouter(prefix="/dl", tags=["dl"])


# ---------------------------------------------------------------- catalog

@router.get("/architectures")
def get_architectures(data_type: str | None = Query(None)):
    return list_architectures(data_type)


@router.get("/layers")
def get_layers(data_type: str | None = Query(None)):
    return list_layers(data_type)


@router.get("/catalog")
def get_catalog():
    return {**catalog_payload(), "tf_available": tf_available()}


# ---------------------------------------------------------------- design

@router.post("/scaffold")
def scaffold(payload: ScaffoldRequest):
    try:
        return scaffold_spec(
            architecture=payload.architecture,
            data_type=payload.data_type,
            input_shape=payload.input_shape,
            n_classes=payload.n_classes,
            is_regression=payload.is_regression,
        )
    except ValueError as e:
        raise ValidationError(str(e))


@router.post("/validate", response_model=ValidationReport)
def validate(payload: ValidateRequest):
    return validate_spec(payload.spec.model_dump())


@router.post("/codegen", response_model=DLCodegenResponse)
def codegen(payload: DLCodegenRequest):
    code = generate_code(payload.spec.model_dump())
    return DLCodegenResponse(filename="model.py", code=code)


# ---------------------------------------------------------------- train

@router.post("/train", response_model=DLTrainResponse)
def train(payload: DLTrainRequest, db: Session = Depends(get_db)):
    spec = payload.spec.model_dump()

    # Always validate + generate code (works without TF).
    report = validate_spec(spec)
    code = generate_code(spec)

    if not report["ok"]:
        return DLTrainResponse(
            model_id="", task=spec.get("task", ""), api=spec.get("api", "sequential"),
            architecture=spec.get("architecture"), tf_available=tf_available(),
            trained=False, message="Fix the shape errors before training.",
            validation=ValidationReport(**report), code=code,
        )

    if not tf_available():
        return DLTrainResponse(
            model_id="", task=spec.get("task", ""), api=spec.get("api", "sequential"),
            architecture=spec.get("architecture"), tf_available=False, trained=False,
            message=("TensorFlow isn't installed, so the network was designed, shape-validated "
                     "and exported as code — but not trained. Install the optional deep-learning "
                     "dependencies with `pip install -r requirements-dl.txt` (Python 3.11–3.12), "
                     'or set `INSTALL_DL: "true"` in docker-compose.yml.'),
            validation=ValidationReport(**report), code=code,
        )

    data_type = spec.get("data_type", "tabular")
    if data_type != "tabular":
        return DLTrainResponse(
            model_id="", task=spec.get("task", ""), api=spec.get("api", "sequential"),
            architecture=spec.get("architecture"), tf_available=True, trained=False,
            message=(f"In-studio training currently supports tabular data. Your {data_type} "
                     "architecture was validated and exported — run the generated script with your "
                     f"{data_type} pipeline to train it."),
            validation=ValidationReport(**report), code=code,
        )

    ds = db.get(Dataset, payload.dataset_id)
    if not ds:
        raise NotFoundError("dataset")
    if ds.project_id != payload.project_id:
        raise ValidationError("dataset does not belong to project")
    if not payload.target_col:
        raise ValidationError("a target column is required for tabular training")

    df = load_dataframe(ds.storage_path)

    row = DLModel(
        project_id=payload.project_id,
        dataset_id=payload.dataset_id,
        architecture=spec.get("architecture") or spec.get("api", "custom"),
        data_type=data_type,
        config_json=spec,
        epochs=int(spec.get("training", {}).get("epochs", 0)),
        batch_size=int(spec.get("training", {}).get("batch_size", 32)),
        optimizer=spec.get("compile", {}).get("optimizer", "adam"),
        status="running",
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    out_dir = model_path(payload.project_id, row.id)
    try:
        result = train_tabular(
            df, spec=spec, target_col=payload.target_col,
            save_dir=out_dir, random_state=payload.random_state,
        )
    except Exception as e:
        row.status = "failed"
        row.metrics_json = {"error": str(e)}
        db.commit()
        raise ValidationError(f"training failed: {e}")

    code_path = out_dir / "model.py"
    code_path.write_text(code, encoding="utf-8")

    row.status = "succeeded"
    row.epochs = result["epochs_ran"]
    row.metrics_json = {"metrics": result["metrics"], "n_train": result["n_train"],
                        "n_test": result["n_test"], "n_features": result["n_features"],
                        "class_names": result["class_names"]}
    row.history_json = result["history"]
    row.artifact_path = result["model_path"]
    row.code_path = str(code_path)
    db.commit()

    return DLTrainResponse(
        model_id=row.id, task=result["task"], api=spec.get("api", "sequential"),
        architecture=spec.get("architecture"), tf_available=True, trained=True,
        message="Trained successfully.",
        metrics=result["metrics"], figures=result["figures"], history=result["history"],
        validation=ValidationReport(**report),
        n_train=result["n_train"], n_test=result["n_test"], n_features=result["n_features"],
        n_classes=result["n_classes"], epochs_ran=result["epochs_ran"],
        feature_columns=result["feature_columns"], target_column=result["target_column"],
        class_names=result["class_names"], code=code, code_filename="model.py",
        model_download_url=f"/api/v1/dl/models/{row.id}/download?fmt=model",
        code_download_url=f"/api/v1/dl/models/{row.id}/download?fmt=code",
    )


# ---------------------------------------------------------------- registry

@router.get("/models", response_model=list[DLRunRead])
def list_models(project_id: str | None = Query(None), db: Session = Depends(get_db)):
    q = db.query(DLModel)
    if project_id:
        q = q.filter(DLModel.project_id == project_id)
    return q.order_by(DLModel.created_at.desc()).all()


@router.get("/models/{model_id}", response_model=DLRunRead)
def get_model(model_id: str, db: Session = Depends(get_db)):
    row = db.get(DLModel, model_id)
    if not row:
        raise NotFoundError("model")
    return row


@router.get("/models/{model_id}/download")
def download_model(model_id: str, fmt: str = Query("model"), db: Session = Depends(get_db)):
    row = db.get(DLModel, model_id)
    if not row:
        raise NotFoundError("model")
    if fmt == "model":
        p = Path(row.artifact_path or "")
        if not p.exists():
            raise NotFoundError("model artifact")
        return FileResponse(p, media_type="application/octet-stream", filename="model.keras")
    if fmt == "code":
        p = Path(row.code_path or "")
        if not p.exists():
            raise NotFoundError("training script")
        return FileResponse(p, media_type="text/x-python", filename="model.py")
    raise ValidationError(f"unknown fmt '{fmt}' — expected model or code")
