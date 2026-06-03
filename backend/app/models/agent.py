from __future__ import annotations

from sqlalchemy import Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base
from ._mixins import TimestampMixin, UUIDPKMixin


class Agent(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "agents"

    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(200))
    agent_type: Mapped[str] = mapped_column(String(32))   # react, tool_calling, multi_agent
    llm_provider: Mapped[str] = mapped_column(String(32))
    llm_model: Mapped[str] = mapped_column(String(128))
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    tools_json: Mapped[list] = mapped_column(JSON, default=list)
    config_json: Mapped[dict] = mapped_column(JSON, default=dict)

    project = relationship("Project", back_populates="agents")
    runs = relationship("AgentRun", back_populates="agent", cascade="all, delete-orphan")


class AgentRun(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "agent_runs"

    agent_id: Mapped[str] = mapped_column(String(36), ForeignKey("agents.id", ondelete="CASCADE"))
    input: Mapped[str] = mapped_column(Text)
    output: Mapped[str] = mapped_column(Text, default="")
    trace_json: Mapped[dict] = mapped_column(JSON, default=dict)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default="succeeded")

    agent = relationship("Agent", back_populates="runs")
