"""Tab 6 — Deployment Studio. Catalog implemented; generators in Phase 5."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..schemas import DeployGenerateRequest
from ..services.deployment_studio import catalog

router = APIRouter(prefix="/deploy", tags=["deploy"])


@router.get("/catalog")
def get_catalog():
    return catalog()


@router.post("/generate")
def generate(payload: DeployGenerateRequest):
    raise HTTPException(status_code=501, detail="Deployment generators are scheduled for Phase 5.")
