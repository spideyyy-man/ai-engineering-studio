from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..core.errors import NotFoundError
from ..database import get_db
from ..models import Project
from ..schemas import MessageResponse, ProjectCreate, ProjectRead, ProjectUpdate

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("", response_model=list[ProjectRead])
def list_projects(db: Session = Depends(get_db)):
    return db.query(Project).order_by(Project.updated_at.desc()).all()


@router.post("", response_model=ProjectRead, status_code=201)
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)):
    p = Project(name=payload.name, description=payload.description)
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


@router.get("/{project_id}", response_model=ProjectRead)
def get_project(project_id: str, db: Session = Depends(get_db)):
    p = db.get(Project, project_id)
    if not p:
        raise NotFoundError("project")
    return p


@router.patch("/{project_id}", response_model=ProjectRead)
def update_project(project_id: str, payload: ProjectUpdate, db: Session = Depends(get_db)):
    p = db.get(Project, project_id)
    if not p:
        raise NotFoundError("project")
    if payload.name is not None:
        p.name = payload.name
    if payload.description is not None:
        p.description = payload.description
    db.commit()
    db.refresh(p)
    return p


@router.delete("/{project_id}", response_model=MessageResponse)
def delete_project(project_id: str, db: Session = Depends(get_db)):
    p = db.get(Project, project_id)
    if not p:
        raise NotFoundError("project")
    db.delete(p)
    db.commit()
    return MessageResponse(message="deleted")
