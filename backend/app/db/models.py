import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return uuid.uuid4().hex


class UTCDateTime(TypeDecorator):
    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value

    def process_result_value(self, value, dialect):
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value


class Base(DeclarativeBase):
    pass


class Visitor(Base):
    __tablename__ = "visitors"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    last_ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(400), nullable=True)

    runs: Mapped[list["Run"]] = relationship(
        back_populates="visitor", cascade="all, delete-orphan", passive_deletes=True
    )


class UsageEvent(Base):
    __tablename__ = "usage_events"
    __table_args__ = (
        Index("ix_usage_visitor_time", "visitor_id", "created_at"),
        Index("ix_usage_ip_time", "ip_address", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    visitor_id: Mapped[str] = mapped_column(String(36), index=True)
    ip_address: Mapped[str] = mapped_column(String(64))
    user_agent: Mapped[str | None] = mapped_column(String(400), nullable=True)
    run_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = (Index("ix_runs_visitor_created", "visitor_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    visitor_id: Mapped[str] = mapped_column(ForeignKey("visitors.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(20), default="running", index=True)
    stop_reason: Mapped[str | None] = mapped_column(String(40), nullable=True)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    population_size: Mapped[int] = mapped_column(Integer)
    max_generations: Mapped[int] = mapped_column(Integer)
    generations_completed: Mapped[int] = mapped_column(Integer, default=0)
    best_fitness: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_generation: Mapped[int | None] = mapped_column(Integer, nullable=True)
    champion: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    visitor: Mapped[Visitor] = relationship(back_populates="runs")
    generations: Mapped[list["Generation"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True, order_by="Generation.index"
    )


class Generation(Base):
    __tablename__ = "generations"
    __table_args__ = (UniqueConstraint("run_id", "index", name="uq_generation_run_index"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    index: Mapped[int] = mapped_column(Integer)
    best_fitness: Mapped[float] = mapped_column(Float)
    mean_fitness: Mapped[float] = mapped_column(Float)
    std_fitness: Mapped[float] = mapped_column(Float)
    species_count: Mapped[int] = mapped_column(Integer)
    best_genome_id: Mapped[int] = mapped_column(Integer)
    best_genome_nodes: Mapped[int] = mapped_column(Integer, default=0)
    best_genome_connections: Mapped[int] = mapped_column(Integer, default=0)
    ticks: Mapped[int] = mapped_column(Integer)
    duration_ms: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    run: Mapped[Run] = relationship(back_populates="generations")
