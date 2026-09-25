from typing import Literal

from pydantic import BaseModel, Field, field_validator

SimulationSpeed = Literal["0.25", "0.5", "1", "2", "4", "8", "max"]

SPEED_FACTORS: dict[str, float | None] = {
    "0.25": 0.25,
    "0.5": 0.5,
    "1": 1.0,
    "2": 2.0,
    "4": 4.0,
    "8": 8.0,
    "max": None,
}


class RunConfig(BaseModel):
    name: str | None = Field(default=None, max_length=80)
    population_size: int = Field(default=50, ge=10, le=150)
    max_generations: int = Field(default=50, ge=1, le=300)
    track_length: Literal["short", "medium", "long"] = "medium"
    track_width: Literal["wide", "standard", "narrow"] = "standard"
    track_curviness: Literal["gentle", "standard", "twisty"] = "standard"
    track_mode: Literal["new", "same"] = "new"
    track_seed: int | None = Field(default=None, ge=0, le=2_147_483_647)
    weight_mutation_rate: float = Field(default=0.8, ge=0, le=1)
    add_connection_rate: float = Field(default=0.3, ge=0, le=1)
    add_node_rate: float = Field(default=0.2, ge=0, le=1)
    speed: SimulationSpeed = "1"

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return " ".join(value.split()) or None
