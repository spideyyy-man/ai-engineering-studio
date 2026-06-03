from __future__ import annotations

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base
from ._mixins import TimestampMixin, UUIDPKMixin


class Project(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "projects"

    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    user_id: Mapped[str | None] = mapped_column(String(36), nullable=True)  # reserved for multi-user

    datasets = relationship("Dataset", back_populates="project", cascade="all, delete-orphan")
    ml_models = relationship("MLModel", back_populates="project", cascade="all, delete-orphan")
    dl_models = relationship("DLModel", back_populates="project", cascade="all, delete-orphan")
    rag_indexes = relationship("RAGIndex", back_populates="project", cascade="all, delete-orphan")
    agents = relationship("Agent", back_populates="project", cascade="all, delete-orphan")
    deployments = relationship("Deployment", back_populates="project", cascade="all, delete-orphan")
    runs = relationship("Run", back_populates="project", cascade="all, delete-orphan")
