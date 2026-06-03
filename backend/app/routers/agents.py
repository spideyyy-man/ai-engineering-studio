"""Tab 5 — Agent Studio. Catalog implemented; runtime in Phase 4."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..schemas import AgentCreate, AgentRunRequest
from ..services.agent_studio import catalog

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("/catalog")
def get_catalog():
    return catalog()


@router.post("")
def create_agent(payload: AgentCreate):
    raise HTTPException(status_code=501, detail="Agent runtime is scheduled for Phase 4.")


@router.post("/{agent_id}/run")
def run_agent(agent_id: str, payload: AgentRunRequest):
    raise HTTPException(status_code=501, detail="Agent runtime is scheduled for Phase 4.")
