import hmac
from typing import Annotated

from fastapi import APIRouter, Query, Request, status

from app.core.config import get_settings
from app.core.errors import AppError, not_found
from app.core.rate_limit import rate_limiter
from app.core.visitor import client_ip
from app.db.session import DB
from app.services import usage_service

router = APIRouter(prefix="/api/admin", tags=["admin"])


def _authorize(request: Request) -> None:
    token = get_settings().admin_token
    if not token:
        raise not_found("This endpoint doesn't exist.")
    rate_limiter.hit(f"admin:{client_ip(request)}", limit=20, window_seconds=300)
    supplied = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(supplied.encode(), token.encode()):
        raise AppError(status.HTTP_401_UNAUTHORIZED, "admin_unauthorized", "That admin token isn't valid.")


@router.get("/usage")
async def usage_report(request: Request, db: DB, days: Annotated[int, Query(ge=1, le=90)] = 7):
    _authorize(request)
    report = await usage_service.admin_report(db, days)
    report["simulations"] = {
        "active": request.app.state.simulations.active_count,
        "max": request.app.state.simulations.max_concurrent,
    }
    return report
