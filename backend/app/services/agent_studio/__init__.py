"""Agent Studio — Tab 5. Agent types + tool registry (stub)."""
from __future__ import annotations

from typing import Any


AGENT_TYPES = [
    {"key": "react", "label": "ReAct",
     "description": "Reason → Act → Observe loop. Strong general-purpose default."},
    {"key": "tool_calling", "label": "Tool-Calling",
     "description": "Native function-calling. Cleaner traces; needs a tool-calling LLM."},
    {"key": "multi_agent", "label": "Multi-Agent (Supervisor)",
     "description": "Supervisor routes tasks to specialist agents."},
]

TOOL_REGISTRY = [
    {"key": "search", "label": "Web Search", "config_schema": {"engine": "str"}},
    {"key": "sql", "label": "SQL Query", "config_schema": {"db_uri": "str"}},
    {"key": "rag", "label": "RAG Lookup", "config_schema": {"index_id": "str"}},
    {"key": "calculator", "label": "Calculator", "config_schema": {}},
    {"key": "http", "label": "HTTP Request", "config_schema": {"base_url": "str", "headers": "dict"}},
]


def catalog() -> dict[str, Any]:
    return {"agent_types": AGENT_TYPES, "tools": TOOL_REGISTRY}
