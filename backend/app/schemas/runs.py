from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def clean_name(value: str) -> str:
    value = " ".join(value.split())
    if not value:
        raise ValueError("Enter a name.")
    return value


RunStatus = Literal["running", "completed", "stopped", "interrupted", "failed"]
RunSort = Literal["newest", "oldest", "best_fitness", "generations", "name"]


class GenerationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    index: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    species_count: int
    best_genome_id: int
    best_genome_nodes: int
    best_genome_connections: int
    ticks: int
    duration_ms: int
    created_at: datetime


class RunSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    status: RunStatus
    stop_reason: str | None
    population_size: int
    max_generations: int
    generations_completed: int
    best_fitness: float | None
    best_generation: int | None
    created_at: datetime
    finished_at: datetime | None
    duration_seconds: float


class RunDetail(RunSummary):
    config: dict
    champion: dict | None
    notes: str | None
    generations: list[GenerationOut]


class RunPage(BaseModel):
    items: list[RunSummary]
    total: int
    page: int
    page_size: int
    pages: int


class RunUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=80)
    notes: str | None = Field(default=None, max_length=2000)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        return None if value is None else clean_name(value)


class DeleteResult(BaseModel):
    deleted: int


class BestRunPoint(BaseModel):
    id: str
    name: str
    best_fitness: float | None
    generations_completed: int
    created_at: datetime


class OverviewStats(BaseModel):
    total_runs: int
    completed_runs: int
    active_runs: int
    total_generations: int
    best_fitness: float | None
    best_run_id: str | None
    best_run_name: str | None
    training_seconds: float
    recent_runs: list[RunSummary]
    trend: list[BestRunPoint]
