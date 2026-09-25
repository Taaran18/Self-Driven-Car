from typing import Annotated

from fastapi import APIRouter, Query, Request, Response, status

from app.core.errors import AppError
from app.core.visitor import CurrentVisitor
from app.db.session import DB
from app.schemas.runs import RunDetail, RunPage, RunSort, RunStatus, RunSummary, RunUpdate
from app.services import run_service

router = APIRouter(prefix="/api/runs", tags=["runs"])


@router.get("", response_model=RunPage)
async def list_runs(
    visitor: CurrentVisitor,
    db: DB,
    status_filter: Annotated[RunStatus | None, Query(alias="status")] = None,
    q: Annotated[str | None, Query(max_length=80)] = None,
    sort: RunSort = "newest",
    page: Annotated[int, Query(ge=1, le=10_000)] = 1,
    page_size: Annotated[int, Query(ge=5, le=100)] = 10,
):
    return await run_service.list_runs(db, visitor.id, status_filter, q, sort, page, page_size)


@router.get("/{run_id}", response_model=RunDetail)
async def get_run(run_id: str, visitor: CurrentVisitor, db: DB):
    return await run_service.get_run(db, visitor.id, run_id, with_generations=True)


@router.patch("/{run_id}", response_model=RunSummary)
async def update_run(run_id: str, body: RunUpdate, visitor: CurrentVisitor, db: DB):
    return await run_service.update_run(db, visitor.id, run_id, body)


@router.delete("/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_run(run_id: str, request: Request, visitor: CurrentVisitor, db: DB):
    if run_id in request.app.state.simulations.run_ids_for(visitor.id):
        raise AppError(
            status.HTTP_409_CONFLICT,
            "run_active",
            "This run is still training. Stop it in the simulator before deleting it.",
        )
    await run_service.delete_run(db, visitor.id, run_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
