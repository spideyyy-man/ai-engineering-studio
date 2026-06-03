from __future__ import annotations

from sqlalchemy import ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base
from ._mixins import TimestampMixin, UUIDPKMixin


class RAGIndex(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "rag_indexes"

    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(200))
    source_type: Mapped[str] = mapped_column(String(32))  # pdf, docx, txt, csv, url, youtube
    chunking_strategy: Mapped[str] = mapped_column(String(32), default="recursive")
    chunk_size: Mapped[int] = mapped_column(Integer, default=512)
    chunk_overlap: Mapped[int] = mapped_column(Integer, default=64)
    embedding_provider: Mapped[str] = mapped_column(String(32))
    embedding_model: Mapped[str] = mapped_column(String(128))
    vector_store: Mapped[str] = mapped_column(String(32))
    n_chunks: Mapped[int] = mapped_column(Integer, default=0)
    storage_path: Mapped[str] = mapped_column(String(500))
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)

    project = relationship("Project", back_populates="rag_indexes")
