from __future__ import annotations

from sqlalchemy import ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base
from ._mixins import TimestampMixin, UUIDPKMixin


class Dataset(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "datasets"

    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(200))
    source_type: Mapped[str] = mapped_column(String(32))  # csv, excel, sql, mongo, api, sample
    storage_path: Mapped[str] = mapped_column(String(500))
    n_rows: Mapped[int] = mapped_column(Integer, default=0)
    n_cols: Mapped[int] = mapped_column(Integer, default=0)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    dtypes_json: Mapped[dict] = mapped_column(JSON, default=dict)
    profile_json: Mapped[dict] = mapped_column(JSON, default=dict)

    project = relationship("Project", back_populates="datasets")
    transforms = relationship(
        "DatasetTransform",
        back_populates="dataset",
        cascade="all, delete-orphan",
        order_by="DatasetTransform.step",
    )
