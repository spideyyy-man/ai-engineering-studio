from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class DeployGenerateRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    project_id: str
    target_type: str   # fastapi, docker, streamlit, dash
    model_id: str | None = None
    rag_index_id: str | None = None
    agent_id: str | None = None
    config: dict[str, Any] = {}


class DeploymentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    target_type: str
    artifact_path: str
    status: str
    created_at: datetime
