from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class AgentToolConfig(BaseModel):
    name: str
    type: str  # search, sql, rag, calculator, http
    config: dict[str, Any] = {}


class AgentCreate(BaseModel):
    project_id: str
    name: str
    agent_type: str = "react"
    llm_provider: str | None = None
    llm_model: str | None = None
    system_prompt: str = ""
    tools: list[AgentToolConfig] = []
    config: dict[str, Any] = {}


class AgentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    name: str
    agent_type: str
    llm_provider: str
    llm_model: str
    system_prompt: str
    tools_json: list[dict[str, Any]] = []
    config_json: dict[str, Any] = {}
    created_at: datetime


class AgentRunRequest(BaseModel):
    input: str


class AgentRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    agent_id: str
    input: str
    output: str
    trace_json: dict[str, Any] = {}
    total_tokens: int
    cost_usd: float
    duration_ms: int
    status: str
    created_at: datetime
