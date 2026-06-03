from __future__ import annotations

from sqlalchemy import ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base
from ._mixins import TimestampMixin, UUIDPKMixin


class DLModel(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "dl_models"

    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id", ondelete="CASCADE"))
    dataset_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("datasets.id"), nullable=True)
    architecture: Mapped[str] = mapped_column(String(64))  # ann, cnn, lstm, gru, transformer
    data_type: Mapped[str] = mapped_column(String(32))    # tabular, image, text, timeseries
    config_json: Mapped[dict] = mapped_column(JSON, default=dict)
    epochs: Mapped[int] = mapped_column(Integer, default=0)
    batch_size: Mapped[int] = mapped_column(Integer, default=32)
    optimizer: Mapped[str] = mapped_column(String(32), default="adam")
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    history_json: Mapped[dict] = mapped_column(JSON, default=dict)
    artifact_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    code_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="pending")

    project = relationship("Project", back_populates="dl_models")
