import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import admin, health, me, runs, simulation
from app.core.config import get_settings
from app.core.errors import register_error_handlers
from app.core.logging import configure_logging
from app.core.middleware import RequestContextMiddleware
from app.db.session import dispose_db, get_sessionmaker, init_db
from app.services.run_service import mark_orphaned_runs
from app.services.usage_service import purge_old_usage
from app.simulation.manager import SimulationManager, TicketStore
from app.simulation.source import code_steps

logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.simulations = SimulationManager(settings.max_concurrent_simulations)
    app.state.tickets = TicketStore(settings.ws_ticket_ttl_seconds)
    await init_db()
    async with get_sessionmaker()() as db:
        orphaned = await mark_orphaned_runs(db)
        purged = await purge_old_usage(db)
    code_steps()
    logger.info("startup", extra={"env": settings.app_env, "orphaned_runs": orphaned, "purged_usage_events": purged})
    yield
    sessions = app.state.simulations.stop_all("server_shutdown")
    await asyncio.gather(*(asyncio.to_thread(s.join, 3) for s in sessions))
    await dispose_db()
    logger.info("shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(
        title=settings.app_name,
        lifespan=lifespan,
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None,
        openapi_url=None if settings.is_production else "/openapi.json",
    )
    app.add_middleware(RequestContextMiddleware, allowed_origins=settings.origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["content-type", "x-request-id", "x-visitor-id", "authorization"],
    )
    register_error_handlers(app)
    for module in (health, me, runs, admin, simulation):
        app.include_router(module.router)
    return app


app = create_app()
