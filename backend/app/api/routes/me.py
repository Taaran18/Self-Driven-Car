from fastapi import APIRouter, Request, Response, status
from fastapi.responses import JSONResponse

from app.core.rate_limit import rate_limiter
from app.core.visitor import CurrentVisitor
from app.db.models import Visitor
from app.db.session import DB
from app.schemas.runs import DeleteResult, OverviewStats
from app.services import run_service, usage_service

router = APIRouter(prefix="/api/me", tags=["me"])


def _active_run_ids(request: Request, visitor_id: str) -> set[str]:
    return request.app.state.simulations.run_ids_for(visitor_id)


@router.get("")
async def me(visitor: CurrentVisitor, db: DB):
    record = await db.get(Visitor, visitor.id)
    return {
        "id": visitor.id,
        "created_at": record.created_at if record else None,
        "usage": await usage_service.summary(db, visitor.ip),
    }


@router.get("/usage")
async def usage(visitor: CurrentVisitor, db: DB):
    return await usage_service.summary(db, visitor.ip)


@router.get("/overview", response_model=OverviewStats)
async def overview(visitor: CurrentVisitor, db: DB):
    return await run_service.overview(db, visitor.id)


@router.get("/export")
async def export_data(visitor: CurrentVisitor, db: DB):
    rate_limiter.hit(f"export:{visitor.id}", limit=5, window_seconds=300)
    record = await db.get(Visitor, visitor.id)
    data = await run_service.export_visitor(db, visitor.id, record.created_at if record else None)
    return JSONResponse(data, headers={"content-disposition": 'attachment; filename="self-driven-car-export.json"'})


@router.delete("/runs", response_model=DeleteResult)
async def delete_history(request: Request, visitor: CurrentVisitor, db: DB):
    deleted = await run_service.delete_runs(db, visitor.id, _active_run_ids(request, visitor.id))
    return DeleteResult(deleted=deleted)


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
async def delete_everything(request: Request, visitor: CurrentVisitor, db: DB):
    for session in request.app.state.simulations.stop_all_for(visitor.id):
        session.join(0)
    await run_service.delete_visitor(db, visitor.id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
