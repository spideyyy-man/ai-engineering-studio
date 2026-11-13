"""Tab 1 — Data Engineering & EDA. Fully implemented."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..config import get_settings
from ..core.errors import NotFoundError, ValidationError
from ..core.paths import codegen_path, dataset_path, safe_filename
from ..database import get_db
from ..models import Dataset, DatasetTransform, Project
from ..schemas import (
    AutoFixRequest,
    AutoFixResponse,
    AutoFixStep,
    BaselineResponse,
    CleanDatasetRequest,
    CodegenRequest,
    CodegenResponse,
    DatasetRead,
    DatasetSummary,
    HistoryEntry,
    HistoryResponse,
    MessageResponse,
    PipelineExportResponse,
    ProfileResponse,
    QualityReport,
    RecommendationsResponse,
    RecommendedFix,
    SuggestedFixResponse,
    TaskDetectionResponse,
    TransformOperation,
    TransformRequest,
    TransformResponse,
    UndoResponse,
)
from ..services.codegen import render_template
from ..services.data_engineering import (
    apply_operation,
    build_pipeline_export,
    build_quality_report,
    build_recommendations,
    compute_baseline,
    detect_task_type,
    list_operations,
    load_dataframe,
    load_snapshot,
    profile_dataframe,
    recommend_fix,
    save_snapshot,
    save_upload,
    snapshot_path,
    stats_diff,
    suggest_operations,
    summarize_for_card,
)
from ..services.data_engineering.transformer import OPERATIONS

router = APIRouter(prefix="/data", tags=["data"])


# ---------------------------------------------------------------- upload

@router.post("/upload", response_model=DatasetRead, status_code=201)
def upload_dataset(
    project_id: str = Form(...),
    name: str | None = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    project = db.get(Project, project_id)
    if not project:
        raise NotFoundError("project")

    if not file.filename:
        raise ValidationError("file has no name")

    # Never trust the client's filename: strip any directory part and odd
    # characters, then pick a name that doesn't overwrite an existing dataset.
    filename = safe_filename(file.filename, default="dataset")
    ext = Path(filename).suffix.lower()
    if ext not in {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json"}:
        raise ValidationError(f"unsupported file type {ext}")

    dest = dataset_path(project_id, filename)
    stem, counter = Path(filename).stem, 1
    while dest.exists():
        dest = dataset_path(project_id, f"{stem}_{counter}{ext}")
        counter += 1

    max_bytes = get_settings().max_upload_mb * 1024 * 1024
    size = save_upload(file.file, dest, max_bytes=max_bytes)

    # Profile immediately so the UI can show metadata
    df = load_dataframe(dest)
    summary = summarize_for_card(df)

    ds = Dataset(
        project_id=project_id,
        name=name or filename,
        source_type=ext.lstrip("."),
        storage_path=str(dest),
        n_rows=summary["n_rows"],
        n_cols=summary["n_cols"],
        size_bytes=size,
        dtypes_json={c["name"]: c["dtype"] for c in summary["columns"]},
        profile_json={},
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return ds


# ---------------------------------------------------------------- list / detail

@router.get("/datasets", response_model=list[DatasetRead])
def list_datasets(
    project_id: str | None = Query(None),
    db: Session = Depends(get_db),
):
    q = db.query(Dataset)
    if project_id:
        q = q.filter(Dataset.project_id == project_id)
    return q.order_by(Dataset.created_at.desc()).all()


@router.get("/datasets/{dataset_id}", response_model=DatasetSummary)
def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise NotFoundError("dataset")
    df = load_dataframe(ds.storage_path)
    summary = summarize_for_card(df)
    return DatasetSummary(id=ds.id, name=ds.name, **summary)


# ---------------------------------------------------------------- profile

@router.get("/datasets/{dataset_id}/profile", response_model=ProfileResponse)
def profile_dataset(dataset_id: str, db: Session = Depends(get_db)):
    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise NotFoundError("dataset")
    df = load_dataframe(ds.storage_path)
    payload = profile_dataframe(df)

    # Cache the JSON payload on the dataset row (skip figures — they're large)
    ds.profile_json = {k: v for k, v in payload.items() if k != "figures"}
    db.commit()
    return payload


# ---------------------------------------------------------------- quality

@router.get("/datasets/{dataset_id}/quality", response_model=QualityReport)
def quality_dataset(
    dataset_id: str,
    target_col: str | None = Query(None),
    db: Session = Depends(get_db),
):
    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise NotFoundError("dataset")
    df = load_dataframe(ds.storage_path)
    return build_quality_report(df, target_col=target_col)


# ---------------------------------------------------------------- recommendations

@router.get("/datasets/{dataset_id}/recommendations", response_model=RecommendationsResponse)
def recommendations_dataset(
    dataset_id: str,
    target_col: str | None = Query(None),
    db: Session = Depends(get_db),
):
    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise NotFoundError("dataset")
    df = load_dataframe(ds.storage_path)
    recs = build_recommendations(df, target_col=target_col)
    return RecommendationsResponse(recommendations=recs)


# ---------------------------------------------------------------- code generation

@router.post("/datasets/{dataset_id}/codegen", response_model=CodegenResponse)
def codegen_pipeline(
    dataset_id: str,
    payload: CodegenRequest,
    db: Session = Depends(get_db),
):
    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise NotFoundError("dataset")

    steps = payload.steps
    drop_cols: list[str] = []
    has_scaling = False
    has_encoding = False
    deduplicate = False
    scaler_class = "StandardScaler"

    for s in steps:
        if s.step == "drop":
            drop_cols.extend(s.columns)
        elif s.step == "scaling":
            has_scaling = True
            if "RobustScaler" in s.code_hint:
                scaler_class = "RobustScaler"
        elif s.step == "encoding":
            has_encoding = True
        elif s.step == "deduplicate":
            deduplicate = True

    code = render_template(
        "data_engineering/sklearn_pipeline.py.j2",
        {
            "dataset_name": ds.name,
            "target": payload.target,
            "drop_cols": sorted(set(drop_cols)),
            "deduplicate": deduplicate,
            "has_scaling": has_scaling,
            "has_encoding": has_encoding,
            "scaler_class": scaler_class,
        },
    )

    filename = f"preprocess_{ds.id[:8]}.py"
    out_path = codegen_path(ds.project_id, "data_engineering", filename)
    out_path.write_text(code, encoding="utf-8")

    return CodegenResponse(
        filename=filename,
        code=code,
        download_url=f"/api/v1/data/datasets/{ds.id}/codegen/download",
    )


@router.get("/datasets/{dataset_id}/codegen/download")
def download_codegen(dataset_id: str, db: Session = Depends(get_db)):
    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise NotFoundError("dataset")
    # Return the most recent generated file
    out_dir = codegen_path(ds.project_id, "data_engineering", "_").parent
    candidates = sorted(out_dir.glob(f"preprocess_{ds.id[:8]}*.py"))
    if not candidates:
        raise NotFoundError("generated code")
    return FileResponse(candidates[-1], media_type="text/x-python", filename=candidates[-1].name)


# ---------------------------------------------------------------- delete

@router.delete("/datasets/{dataset_id}", response_model=MessageResponse)
def delete_dataset(dataset_id: str, db: Session = Depends(get_db)):
    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise NotFoundError("dataset")
    # Best-effort file removal
    try:
        Path(ds.storage_path).unlink(missing_ok=True)
    except Exception:
        pass
    db.delete(ds)
    db.commit()
    return MessageResponse(message=f"dataset {dataset_id} deleted at {datetime.utcnow().isoformat()}")


# ---------------------------------------------------------------- transformations

@router.get("/transformations", response_model=list[TransformOperation])
def list_transform_ops():
    """Catalog of every data-engineering operation available."""
    return list_operations()


@router.get("/transformations/for-issue/{issue_code}", response_model=SuggestedFixResponse)
def suggest_fixes(issue_code: str):
    keys = suggest_operations(issue_code)
    ops = [op for op in list_operations() if op["key"] in keys]
    return SuggestedFixResponse(issue_code=issue_code, operations=ops)


@router.get("/datasets/{dataset_id}/recommended-fix", response_model=RecommendedFix | None)
def recommended_fix(
    dataset_id: str,
    issue_code: str = Query(...),
    column: str | None = Query(None),
    db: Session = Depends(get_db),
):
    """Return a single suggested operation+params for one (issue, column) pair."""
    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise NotFoundError("dataset")
    df = load_dataframe(ds.storage_path)
    rec = recommend_fix(df, issue_code, column)
    if not rec:
        return None
    return RecommendedFix(
        issue_code=issue_code,
        column=column,
        operation=rec["operation"],
        params=rec["params"],
        rationale=rec["rationale"],
    )


@router.post("/datasets/{dataset_id}/auto-fix", response_model=AutoFixResponse)
def auto_fix(
    dataset_id: str,
    payload: AutoFixRequest,
    target_col: str | None = Query(None),
    db: Session = Depends(get_db),
):
    """Run the recommended fix for every diagnosed issue; persist once at the end.

    Strategy: load df once, apply each recommended op in sequence in memory,
    skip issues with no recommendation, and only write to disk after all steps.
    """
    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise NotFoundError("dataset")

    df = load_dataframe(ds.storage_path)
    n_rows_before, n_cols_before = df.shape

    quality = build_quality_report(df, target_col=target_col)
    steps: list[AutoFixStep] = []

    for issue in quality["issues"]:
        code = issue["code"]
        col = issue.get("column")

        # If a previous step removed the column, this issue is stale — skip.
        if col and col not in df.columns:
            steps.append(AutoFixStep(
                issue_code=code, column=col, operation=None,
                status="skipped",
                summary=f"Column '{col}' was removed by an earlier fix.",
            ))
            continue

        rec = recommend_fix(df, code, col)
        if not rec:
            steps.append(AutoFixStep(
                issue_code=code, column=col, operation=None,
                status="skipped",
                summary="No automated transform — handle at training time."
                if code == "class_imbalance" else "No recommendation registered.",
            ))
            continue

        try:
            df, summary = apply_operation(df, rec["operation"], rec["params"])
            steps.append(AutoFixStep(
                issue_code=code, column=col, operation=rec["operation"],
                status="applied", summary=summary, rationale=rec["rationale"],
            ))
        except Exception as e:
            steps.append(AutoFixStep(
                issue_code=code, column=col, operation=rec["operation"],
                status="failed", error=str(e), rationale=rec["rationale"],
            ))

    n_rows_after, n_cols_after = df.shape
    n_applied = sum(1 for s in steps if s.status == "applied")
    n_skipped = sum(1 for s in steps if s.status == "skipped")
    n_failed = sum(1 for s in steps if s.status == "failed")

    dtypes = {str(c): str(t) for c, t in df.dtypes.items()}

    if payload.inplace:
        path = Path(ds.storage_path)
        _write_df(df, path, ds.source_type)
        ds.n_rows = int(n_rows_after)
        ds.n_cols = int(n_cols_after)
        ds.size_bytes = int(path.stat().st_size) if path.exists() else ds.size_bytes
        ds.dtypes_json = dtypes
        ds.profile_json = {}
        db.commit()
        db.refresh(ds)
        return AutoFixResponse(
            dataset_id=ds.id, inplace=True, steps=steps,
            n_applied=n_applied, n_skipped=n_skipped, n_failed=n_failed,
            n_rows_before=int(n_rows_before), n_rows_after=int(n_rows_after),
            n_cols_before=int(n_cols_before), n_cols_after=int(n_cols_after),
            new_dataset=None,
        )

    # inplace=False — write a single new dataset
    orig = Path(ds.storage_path)
    base = orig.stem
    new_name = (payload.new_name or f"{ds.name}_autofix").strip() or f"{ds.name}_autofix"
    new_filename = f"{base}_autofix{orig.suffix}"
    new_path = dataset_path(ds.project_id, new_filename)
    counter = 1
    while new_path.exists():
        new_filename = f"{base}_autofix_{counter}{orig.suffix}"
        new_path = dataset_path(ds.project_id, new_filename)
        counter += 1

    _write_df(df, new_path, ds.source_type)
    size = int(new_path.stat().st_size) if new_path.exists() else 0

    new_ds = Dataset(
        project_id=ds.project_id,
        name=new_name,
        source_type=ds.source_type,
        storage_path=str(new_path),
        n_rows=int(n_rows_after),
        n_cols=int(n_cols_after),
        size_bytes=size,
        dtypes_json=dtypes,
        profile_json={},
    )
    db.add(new_ds)
    db.commit()
    db.refresh(new_ds)

    return AutoFixResponse(
        dataset_id=ds.id, inplace=False, steps=steps,
        n_applied=n_applied, n_skipped=n_skipped, n_failed=n_failed,
        n_rows_before=int(n_rows_before), n_rows_after=int(n_rows_after),
        n_cols_before=int(n_cols_before), n_cols_after=int(n_cols_after),
        new_dataset=DatasetRead.model_validate(new_ds),
    )


# ---------------------------------------------------------------- clean-dataset (iterative)

@router.post("/datasets/{dataset_id}/clean-dataset", response_model=AutoFixResponse)
def clean_dataset(
    dataset_id: str,
    payload: CleanDatasetRequest,
    db: Session = Depends(get_db),
):
    """Iterative auto-fix until convergence — produces a brand-new dataset.

    Each round:
      1. Re-scan quality on the current (in-memory) dataframe.
      2. For every issue with a recommended fix, apply it.
      3. Stop early if a round resolves 0 issues OR all issues are unfixable
         (e.g., class_imbalance, which is handled at training time).
    Caps at max_rounds (default 5) to avoid pathological loops with
    operations that don't quite converge (rare with our op set).
    """
    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise NotFoundError("dataset")

    df = load_dataframe(ds.storage_path)
    n_rows_before, n_cols_before = df.shape

    all_steps: list[AutoFixStep] = []
    rounds_run = 0
    last_issue_count = -1
    unfixable_codes: set[str] = set()

    for _ in range(max(1, payload.max_rounds)):
        rounds_run += 1
        quality = build_quality_report(df, target_col=payload.target_col)
        issues = quality["issues"]
        if not issues:
            break

        # If this round's issue count didn't decrease and every remaining issue
        # is unfixable, we're done.
        fixable_now = [
            i for i in issues
            if recommend_fix(df, i["code"], i.get("column")) is not None
        ]
        if not fixable_now:
            unfixable_codes.update({i["code"] for i in issues})
            break

        applied_any = False
        for issue in issues:
            code = issue["code"]
            col = issue.get("column")
            if col and col not in df.columns:
                all_steps.append(AutoFixStep(
                    issue_code=code, column=col, operation=None,
                    status="skipped",
                    summary=f"Column '{col}' was removed by an earlier fix.",
                ))
                continue
            rec = recommend_fix(df, code, col)
            if not rec:
                unfixable_codes.add(code)
                all_steps.append(AutoFixStep(
                    issue_code=code, column=col, operation=None,
                    status="skipped",
                    summary="No automated transform — handle at training time."
                    if code == "class_imbalance" else "No recommendation registered.",
                ))
                continue
            try:
                df, summary = apply_operation(df, rec["operation"], rec["params"])
                all_steps.append(AutoFixStep(
                    issue_code=code, column=col, operation=rec["operation"],
                    status="applied", summary=summary, rationale=rec["rationale"],
                ))
                applied_any = True
            except Exception as e:
                all_steps.append(AutoFixStep(
                    issue_code=code, column=col, operation=rec["operation"],
                    status="failed", error=str(e), rationale=rec["rationale"],
                ))

        # Bail if we made no progress this round
        if not applied_any:
            break
        # Convergence: same or more issues than last round → stop
        if last_issue_count == len(issues):
            break
        last_issue_count = len(issues)

    # Final issue count
    final_quality = build_quality_report(df, target_col=payload.target_col)
    final_issues_count = len(final_quality["issues"])
    # Build readable unfixable summary
    if final_issues_count:
        for i in final_quality["issues"]:
            if recommend_fix(df, i["code"], i.get("column")) is None:
                unfixable_codes.add(i["code"])

    n_rows_after, n_cols_after = df.shape
    n_applied = sum(1 for s in all_steps if s.status == "applied")
    n_skipped = sum(1 for s in all_steps if s.status == "skipped")
    n_failed = sum(1 for s in all_steps if s.status == "failed")

    dtypes = {str(c): str(t) for c, t in df.dtypes.items()}

    # Always create a new dataset for clean-dataset
    orig = Path(ds.storage_path)
    base = orig.stem
    new_name = (payload.new_name or f"{ds.name}_clean").strip() or f"{ds.name}_clean"
    new_filename = f"{base}_clean{orig.suffix}"
    new_path = dataset_path(ds.project_id, new_filename)
    counter = 1
    while new_path.exists():
        new_filename = f"{base}_clean_{counter}{orig.suffix}"
        new_path = dataset_path(ds.project_id, new_filename)
        counter += 1

    _write_df(df, new_path, ds.source_type)
    size = int(new_path.stat().st_size) if new_path.exists() else 0

    new_ds = Dataset(
        project_id=ds.project_id,
        name=new_name,
        source_type=ds.source_type,
        storage_path=str(new_path),
        n_rows=int(n_rows_after),
        n_cols=int(n_cols_after),
        size_bytes=size,
        dtypes_json=dtypes,
        profile_json={},
    )
    db.add(new_ds)
    db.commit()
    db.refresh(new_ds)

    return AutoFixResponse(
        dataset_id=ds.id, inplace=False, steps=all_steps,
        n_applied=n_applied, n_skipped=n_skipped, n_failed=n_failed,
        n_rows_before=int(n_rows_before), n_rows_after=int(n_rows_after),
        n_cols_before=int(n_cols_before), n_cols_after=int(n_cols_after),
        new_dataset=DatasetRead.model_validate(new_ds),
        rounds_run=rounds_run,
        final_issues_count=final_issues_count,
        unfixable=sorted(unfixable_codes),
    )


def _write_df(df, path: Path, source_type: str) -> None:
    """Persist a DataFrame back to disk in the same format as the source."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix == ".tsv":
        df.to_csv(path, sep="\t", index=False)
    elif suffix in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
    elif suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif suffix == ".json":
        df.to_json(path, orient="records", indent=2)
    else:
        # Default: rewrite as CSV
        df.to_csv(path, index=False)


def _affected_columns(df_before, df_after, op: str, params: dict) -> list[str]:
    """Columns to include in stats_diff — the op's `columns` param, plus any cols that
    appeared/disappeared."""
    cols = list(dict.fromkeys(list(params.get("columns") or [])))
    added = set(df_after.columns) - set(df_before.columns)
    removed = set(df_before.columns) - set(df_after.columns)
    for c in sorted(added | removed):
        if c not in cols:
            cols.append(c)
    return cols


def _record_history(
    db: Session,
    *,
    ds: Dataset,
    operation: str,
    params: dict,
    summary: str,
    n_rows_before: int,
    n_rows_after: int,
    n_cols_before: int,
    n_cols_after: int,
    snapshot_relpath: str | None,
    stats_diff_payload: dict,
) -> DatasetTransform:
    next_step = 1 + (db.query(DatasetTransform)
                     .filter(DatasetTransform.dataset_id == ds.id)
                     .count())
    entry = DatasetTransform(
        dataset_id=ds.id,
        step=next_step,
        operation=operation,
        params_json=params,
        summary=summary,
        snapshot_path=snapshot_relpath,
        n_rows_before=int(n_rows_before),
        n_rows_after=int(n_rows_after),
        n_cols_before=int(n_cols_before),
        n_cols_after=int(n_cols_after),
        stats_diff_json=stats_diff_payload,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


@router.post("/datasets/{dataset_id}/transform", response_model=TransformResponse)
def transform_dataset(
    dataset_id: str,
    payload: TransformRequest,
    db: Session = Depends(get_db),
):
    """Apply an operation; persist either over the original file or as a new dataset.
    inplace=True snapshots the prior state and records a history entry for Undo."""
    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise NotFoundError("dataset")
    if payload.operation not in OPERATIONS:
        raise ValidationError(f"Unknown operation: {payload.operation}")

    df = load_dataframe(ds.storage_path)
    n_rows_before, n_cols_before = df.shape

    new_df, summary = apply_operation(df, payload.operation, payload.params)
    n_rows_after, n_cols_after = new_df.shape
    dtypes = {str(c): str(t) for c, t in new_df.dtypes.items()}

    diff_cols = _affected_columns(df, new_df, payload.operation, payload.params)
    diff = stats_diff(df, new_df, diff_cols)

    if payload.inplace:
        # 1. snapshot pre-state to history dir
        next_step = 1 + (db.query(DatasetTransform)
                         .filter(DatasetTransform.dataset_id == ds.id)
                         .count())
        snap = snapshot_path(ds.project_id, ds.id, next_step)
        save_snapshot(df, snap)

        # 2. overwrite file
        path = Path(ds.storage_path)
        _write_df(new_df, path, ds.source_type)
        ds.n_rows = int(n_rows_after)
        ds.n_cols = int(n_cols_after)
        ds.size_bytes = int(path.stat().st_size) if path.exists() else ds.size_bytes
        ds.dtypes_json = dtypes
        ds.profile_json = {}  # invalidate cached profile
        db.commit()
        db.refresh(ds)

        # 3. record history entry
        entry = _record_history(
            db, ds=ds,
            operation=payload.operation, params=payload.params, summary=summary,
            n_rows_before=n_rows_before, n_rows_after=n_rows_after,
            n_cols_before=n_cols_before, n_cols_after=n_cols_after,
            snapshot_relpath=str(snap), stats_diff_payload=diff,
        )

        return TransformResponse(
            dataset_id=ds.id,
            inplace=True,
            summary=summary,
            n_rows_before=int(n_rows_before),
            n_rows_after=int(n_rows_after),
            n_cols_before=int(n_cols_before),
            n_cols_after=int(n_cols_after),
            new_dataset=None,
            stats_diff=diff,
            history_step=entry.step,
        )

    # inplace=False — write a new file + new Dataset row in the same project
    orig = Path(ds.storage_path)
    base = orig.stem
    new_name = (payload.new_name or f"{ds.name}_v2").strip() or f"{ds.name}_v2"
    new_filename = f"{base}_{payload.operation}{orig.suffix}"
    new_path = dataset_path(ds.project_id, new_filename)
    # Avoid clobbering: append a numeric suffix if needed
    counter = 1
    while new_path.exists():
        new_filename = f"{base}_{payload.operation}_{counter}{orig.suffix}"
        new_path = dataset_path(ds.project_id, new_filename)
        counter += 1

    _write_df(new_df, new_path, ds.source_type)
    size = int(new_path.stat().st_size) if new_path.exists() else 0

    new_ds = Dataset(
        project_id=ds.project_id,
        name=new_name,
        source_type=ds.source_type,
        storage_path=str(new_path),
        n_rows=int(n_rows_after),
        n_cols=int(n_cols_after),
        size_bytes=size,
        dtypes_json=dtypes,
        profile_json={},
    )
    db.add(new_ds)
    db.commit()
    db.refresh(new_ds)

    return TransformResponse(
        dataset_id=ds.id,
        inplace=False,
        summary=summary,
        n_rows_before=int(n_rows_before),
        n_rows_after=int(n_rows_after),
        n_cols_before=int(n_cols_before),
        n_cols_after=int(n_cols_after),
        new_dataset=DatasetRead.model_validate(new_ds),
        stats_diff=diff,
        history_step=None,
    )


# ---------------------------------------------------------------- history / undo

@router.get("/datasets/{dataset_id}/history", response_model=HistoryResponse)
def get_history(dataset_id: str, db: Session = Depends(get_db)):
    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise NotFoundError("dataset")
    entries = (db.query(DatasetTransform)
               .filter(DatasetTransform.dataset_id == ds.id)
               .order_by(DatasetTransform.step.asc())
               .all())
    return HistoryResponse(dataset_id=ds.id, entries=entries)


@router.post("/datasets/{dataset_id}/undo", response_model=UndoResponse)
def undo_last(dataset_id: str, db: Session = Depends(get_db)):
    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise NotFoundError("dataset")
    last = (db.query(DatasetTransform)
            .filter(DatasetTransform.dataset_id == ds.id)
            .order_by(DatasetTransform.step.desc())
            .first())
    if not last:
        raise ValidationError("nothing to undo — no history for this dataset")
    if not last.snapshot_path or not Path(last.snapshot_path).exists():
        raise ValidationError("undo snapshot missing for the last step")

    snap_df = load_snapshot(Path(last.snapshot_path))
    _write_df(snap_df, Path(ds.storage_path), ds.source_type)

    # Update dataset metadata to the snapshot's state
    ds.n_rows = int(snap_df.shape[0])
    ds.n_cols = int(snap_df.shape[1])
    ds.size_bytes = (
        int(Path(ds.storage_path).stat().st_size)
        if Path(ds.storage_path).exists()
        else ds.size_bytes
    )
    ds.dtypes_json = {str(c): str(t) for c, t in snap_df.dtypes.items()}
    ds.profile_json = {}
    db.delete(last)
    db.commit()
    db.refresh(ds)

    remaining = (db.query(DatasetTransform)
                 .filter(DatasetTransform.dataset_id == ds.id)
                 .count())
    return UndoResponse(
        dataset_id=ds.id,
        undone_step=last.step,
        operation=last.operation,
        n_rows_after=int(snap_df.shape[0]),
        n_cols_after=int(snap_df.shape[1]),
        history_remaining=remaining,
    )


# ---------------------------------------------------------------- pipeline export

@router.get("/datasets/{dataset_id}/pipeline-export", response_model=PipelineExportResponse)
def pipeline_export(dataset_id: str, db: Session = Depends(get_db)):
    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise NotFoundError("dataset")
    entries = (db.query(DatasetTransform)
               .filter(DatasetTransform.dataset_id == ds.id)
               .order_by(DatasetTransform.step.asc())
               .all())
    history = [{
        "step": e.step,
        "operation": e.operation,
        "params": e.params_json or {},
        "summary": e.summary or "",
    } for e in entries]
    code = build_pipeline_export(ds.name, history)
    return PipelineExportResponse(filename=f"preprocess_{ds.id[:8]}.py", code=code)


# ---------------------------------------------------------------- task detection

@router.get("/datasets/{dataset_id}/detect-task", response_model=TaskDetectionResponse)
def detect_task(
    dataset_id: str,
    target_col: str | None = Query(None),
    db: Session = Depends(get_db),
):
    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise NotFoundError("dataset")
    df = load_dataframe(ds.storage_path)
    return detect_task_type(df, target_col=target_col)


# ---------------------------------------------------------------- baseline

@router.get("/datasets/{dataset_id}/baseline", response_model=BaselineResponse)
def baseline(
    dataset_id: str,
    target_col: str = Query(...),
    db: Session = Depends(get_db),
):
    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise NotFoundError("dataset")
    df = load_dataframe(ds.storage_path)
    try:
        return compute_baseline(df, target_col=target_col)
    except Exception as e:
        raise ValidationError(f"baseline failed: {e}")


# ---------------------------------------------------------------- embedding viz (t-SNE)

@router.get("/datasets/{dataset_id}/embedding-viz")
def embedding_viz(
    dataset_id: str,
    method: str = Query("tsne", description="tsne (default)"),
    color_col: str | None = Query(None),
    sample_rows: int = Query(2000, ge=100, le=10000),
    db: Session = Depends(get_db),
):
    """Compute a 2-D embedding (default t-SNE) and return a Plotly figure JSON."""
    import json
    import numpy as np
    import pandas as pd
    import plotly.express as px
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.manifold import TSNE
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise NotFoundError("dataset")
    df = load_dataframe(ds.storage_path)

    if len(df) > sample_rows:
        df = df.sample(sample_rows, random_state=42).reset_index(drop=True)

    target = df[color_col].astype(str) if color_col and color_col in df.columns else None
    X = df.drop(columns=[color_col]) if color_col and color_col in df.columns else df.copy()

    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    pre = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("e", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ], remainder="drop")
    Xt = pre.fit_transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    if Xt.shape[1] == 0:
        raise ValidationError("No usable features for embedding.")

    perplexity = min(30, max(5, len(df) // 3))
    try:
        coords = TSNE(n_components=2, perplexity=perplexity, init="pca",
                      random_state=42, n_jobs=-1).fit_transform(Xt)
    except Exception as e:
        raise ValidationError(f"t-SNE failed: {e}")

    plot_df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})
    if target is not None:
        plot_df["color"] = target.values
        fig = px.scatter(plot_df, x="x", y="y", color="color", opacity=0.75,
                         title=f"t-SNE projection (coloured by {color_col})")
    else:
        fig = px.scatter(plot_df, x="x", y="y", opacity=0.75, title="t-SNE projection")
    fig.update_layout(margin=dict(l=40, r=20, t=40, b=40), height=460)

    return {
        "method": "tsne",
        "n_points": int(len(plot_df)),
        "perplexity": perplexity,
        "figure": json.loads(fig.to_json()),
    }
