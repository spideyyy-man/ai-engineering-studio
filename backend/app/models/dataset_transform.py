from __future__ import annotations

from sqlalchemy import ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base
from ._mixins import TimestampMixin, UUIDPKMixin


class DatasetTransform(Base, UUIDPKMixin, TimestampMixin):
    """One in-place transform applied to a dataset — tracks history for undo + export."""

    __tablename__ = "dataset_transforms"

    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id", ondelete="CASCADE"))
    step: Mapped[int] = mapped_column(Integer)
    operation: Mapped[str] = mapped_column(String(64))
    params_json: Mapped[dict] = mapped_column(JSON, default=dict)
    summary: Mapped[str] = mapped_column(Text, default="")
    snapshot_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    n_rows_before: Mapped[int] = mapped_column(Integer, default=0)
    n_rows_after: Mapped[int] = mapped_column(Integer, default=0)
    n_cols_before: Mapped[int] = mapped_column(Integer, default=0)
    n_cols_after: Mapped[int] = mapped_column(Integer, default=0)
    stats_diff_json: Mapped[dict] = mapped_column(JSON, default=dict)

    dataset = relationship("Dataset", back_populates="transforms")
