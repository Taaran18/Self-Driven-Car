from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.db.session import get_sessionmaker

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request):
    try:
        async with get_sessionmaker()() as db:
            await db.execute(text("SELECT 1"))
        storage = "ok"
    except Exception:
        storage = "unavailable"
    manager = request.app.state.simulations
    body = {
        "status": "ok" if storage == "ok" else "degraded",
        "storage": storage,
        "simulations": {"active": manager.active_count, "max": manager.max_concurrent},
    }
    return JSONResponse(body, status_code=200 if storage == "ok" else 503)
