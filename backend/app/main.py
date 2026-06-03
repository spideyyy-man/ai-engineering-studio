"""FastAPI entrypoint."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .core.errors import register_exception_handlers
from .core.logging import configure_logging, logger
from .database import init_db
from .routers import (
    agents,
    data,
    deployment,
    dl,
    ml,
    projects,
    rag,
    settings as settings_router,
)


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings.ensure_dirs()
    init_db()
    logger.info("{} starting on :{} ({})", settings.app_name, settings.app_port, settings.app_env)
    yield
    logger.info("shutdown complete")


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Production-grade AI engineering platform — ML, DL, RAG, Agents, MLOps.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)


API_PREFIX = "/api/v1"

app.include_router(projects.router, prefix=API_PREFIX)
app.include_router(data.router, prefix=API_PREFIX)
app.include_router(ml.router, prefix=API_PREFIX)
app.include_router(dl.router, prefix=API_PREFIX)
app.include_router(rag.router, prefix=API_PREFIX)
app.include_router(agents.router, prefix=API_PREFIX)
app.include_router(deployment.router, prefix=API_PREFIX)
app.include_router(settings_router.router, prefix=API_PREFIX)


@app.get("/")
def root():
    return {
        "name": settings.app_name,
        "version": app.version,
        "env": settings.app_env,
        "docs": "/docs",
    }


@app.get("/healthz")
def health():
    return {"status": "ok"}
