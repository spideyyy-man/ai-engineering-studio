from __future__ import annotations

from sqlalchemy import ForeignKey, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base
from ._mixins import TimestampMixin, UUIDPKMixin


class Deployment(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "deployments"

    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id", ondelete="CASCADE"))
    target_type: Mapped[str] = mapped_column(String(32))  # fastapi, docker, streamlit, dash
    artifact_path: Mapped[str] = mapped_column(String(500))
    config_json: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="generated")

    project = relationship("Project", back_populates="deployments")
