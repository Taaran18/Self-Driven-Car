import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Self-Driven Car API"
    app_env: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"

    data_dir: str = Field(default_factory=lambda: os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "./data"))
    database_file: str = "self-driven-car.db"

    frontend_url: str = "http://localhost:3000"
    allowed_origins: str = ""

    trust_proxy: bool = True

    trial_runs_per_day: int = Field(default=5, ge=1, le=1000)
    trial_runs_per_week: int = Field(default=20, ge=1, le=5000)
    usage_retention_days: int = Field(default=90, ge=8, le=730)

    admin_token: str = ""

    max_concurrent_simulations: int = Field(default=3, ge=1, le=32)
    max_simulation_minutes: int = Field(default=45, ge=1, le=720)
    idle_disconnect_minutes: int = Field(default=5, ge=1, le=120)
    ws_ticket_ttl_seconds: int = Field(default=60, ge=10, le=600)

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def database_path(self) -> Path:
        return Path(self.data_dir) / self.database_file

    @property
    def database_url(self) -> str:
        if self.database_file == ":memory:":
            return "sqlite+aiosqlite:///:memory:"
        return f"sqlite+aiosqlite:///{self.database_path.resolve()}"

    @property
    def origins(self) -> list[str]:
        extra = [o.strip().rstrip("/") for o in self.allowed_origins.split(",") if o.strip()]
        return sorted({self.frontend_url.rstrip("/"), *extra})


@lru_cache
def get_settings() -> Settings:
    return Settings()
