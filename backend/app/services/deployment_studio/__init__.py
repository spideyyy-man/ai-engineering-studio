"""Deployment Studio — Tab 6. Targets registry (stub)."""
from __future__ import annotations

from typing import Any


DEPLOYMENT_TARGETS = [
    {"key": "fastapi", "label": "FastAPI Service",
     "description": "Generate /predict endpoint + Dockerfile + requirements."},
    {"key": "docker", "label": "Docker Container",
     "description": "Wrap an existing model + FastAPI in a production-ready image."},
    {"key": "streamlit", "label": "Streamlit Demo",
     "description": "Interactive demo app with a prediction form."},
    {"key": "dash", "label": "Dash App",
     "description": "Plotly Dash app for model exploration."},
]


def catalog() -> dict[str, Any]:
    return {"targets": DEPLOYMENT_TARGETS}
