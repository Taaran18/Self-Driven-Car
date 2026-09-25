import math

from sqlalchemy import Select, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.errors import not_found
from app.db.models import Generation, Run, Visitor, utcnow
from app.schemas.runs import OverviewStats, RunPage, RunSort, RunSummary, RunUpdate

_SORTS = {
    "newest": (Run.created_at.desc(),),
    "oldest": (Run.created_at.asc(),),
    "best_fitness": (Run.best_fitness.desc().nulls_last(), Run.created_at.desc()),
    "generations": (Run.generations_completed.desc(), Run.created_at.desc()),
    "name": (func.lower(Run.name).asc(), Run.created_at.desc()),
}


def _filtered(visitor_id: str, status: str | None, query: str | None) -> Select:
    stmt = select(Run).where(Run.visitor_id == visitor_id)
    if status:
        stmt = stmt.where(Run.status == status)
    if query:
        escaped = query.strip().lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        stmt = stmt.where(func.lower(Run.name).like(f"%{escaped}%", escape="\\"))
    return stmt


async def list_runs(
    db: AsyncSession,
    visitor_id: str,
    status: str | None,
    query: str | None,
    sort: RunSort,
    page: int,
    page_size: int,
) -> RunPage:
    base = _filtered(visitor_id, status, query)
    total = await db.scalar(select(func.count()).select_from(base.subquery())) or 0
    pages = max(1, math.ceil(total / page_size))
    page = min(page, pages)
    rows = await db.scalars(base.order_by(*_SORTS[sort]).offset((page - 1) * page_size).limit(page_size))
    return RunPage(
        items=[RunSummary.model_validate(r) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
        pages=pages,
    )


async def get_run(db: AsyncSession, visitor_id: str, run_id: str, with_generations: bool = False) -> Run:
    stmt = select(Run).where(Run.id == run_id, Run.visitor_id == visitor_id)
    if with_generations:
        stmt = stmt.options(selectinload(Run.generations))
    run = await db.scalar(stmt)
    if run is None:
        raise not_found("This run doesn't exist or was deleted.")
    return run


async def update_run(db: AsyncSession, visitor_id: str, run_id: str, data: RunUpdate) -> Run:
    run = await get_run(db, visitor_id, run_id)
    if data.name is not None:
        run.name = data.name
    if data.notes is not None:
        run.notes = data.notes.strip() or None
    await db.commit()
    return run


async def delete_run(db: AsyncSession, visitor_id: str, run_id: str) -> None:
    run = await get_run(db, visitor_id, run_id)
    await db.delete(run)
    await db.commit()


async def delete_runs(db: AsyncSession, visitor_id: str, exclude_ids: set[str]) -> int:
    stmt = delete(Run).where(Run.visitor_id == visitor_id)
    if exclude_ids:
        stmt = stmt.where(Run.id.not_in(exclude_ids))
    result = await db.execute(stmt)
    await db.commit()
    return result.rowcount or 0


async def overview(db: AsyncSession, visitor_id: str) -> OverviewStats:
    totals = (
        await db.execute(
            select(
                func.count(Run.id),
                func.coalesce(func.sum(Run.generations_completed), 0),
                func.coalesce(func.sum(Run.duration_seconds), 0.0),
            ).where(Run.visitor_id == visitor_id)
        )
    ).one()
    by_status = dict(
        (
            await db.execute(
                select(Run.status, func.count(Run.id)).where(Run.visitor_id == visitor_id).group_by(Run.status)
            )
        ).all()
    )
    best = await db.scalar(
        select(Run)
        .where(Run.visitor_id == visitor_id, Run.best_fitness.is_not(None))
        .order_by(Run.best_fitness.desc())
        .limit(1)
    )
    recent = list(
        await db.scalars(select(Run).where(Run.visitor_id == visitor_id).order_by(Run.created_at.desc()).limit(5))
    )
    trend = list(
        await db.scalars(select(Run).where(Run.visitor_id == visitor_id).order_by(Run.created_at.desc()).limit(20))
    )
    return OverviewStats(
        total_runs=totals[0],
        completed_runs=by_status.get("completed", 0),
        active_runs=by_status.get("running", 0),
        total_generations=int(totals[1]),
        best_fitness=best.best_fitness if best else None,
        best_run_id=best.id if best else None,
        best_run_name=best.name if best else None,
        training_seconds=float(totals[2]),
        recent_runs=[RunSummary.model_validate(r) for r in recent],
        trend=[
            {
                "id": r.id,
                "name": r.name,
                "best_fitness": r.best_fitness,
                "generations_completed": r.generations_completed,
                "created_at": r.created_at,
            }
            for r in reversed(trend)
        ],
    )


async def next_run_name(db: AsyncSession, visitor_id: str) -> str:
    count = await db.scalar(select(func.count(Run.id)).where(Run.visitor_id == visitor_id)) or 0
    return f"Training Run {count + 1}"


async def create_run(db: AsyncSession, run_id: str, visitor_id: str, name: str | None, config: dict) -> Run:
    run = Run(
        id=run_id,
        visitor_id=visitor_id,
        name=name or await next_run_name(db, visitor_id),
        status="running",
        config=config,
        population_size=config["population_size"],
        max_generations=config["max_generations"],
    )
    db.add(run)
    await db.commit()
    return run


async def record_generation(db: AsyncSession, run_id: str, stats: dict, champion: dict | None) -> None:
    db.add(
        Generation(
            run_id=run_id,
            index=stats["generation"],
            best_fitness=stats["best_fitness"],
            mean_fitness=stats["mean_fitness"],
            std_fitness=stats["std_fitness"],
            species_count=stats["species_count"],
            best_genome_id=stats["best_genome_id"],
            best_genome_nodes=stats["best_genome_nodes"],
            best_genome_connections=stats["best_genome_connections"],
            ticks=stats["ticks"],
            duration_ms=stats["duration_ms"],
        )
    )
    run = await db.get(Run, run_id)
    if run is not None:
        run.generations_completed = stats["generation"] + 1
        run.duration_seconds = stats["elapsed_seconds"]
        if run.best_fitness is None or stats["best_fitness"] > run.best_fitness:
            run.best_fitness = stats["best_fitness"]
            run.best_generation = stats["generation"]
            if champion is not None:
                run.champion = champion
    await db.commit()


async def finish_run(db: AsyncSession, run_id: str, status: str, reason: str, elapsed: float) -> Run | None:
    run = await db.get(Run, run_id)
    if run is None or run.status != "running":
        return run
    run.status = status
    run.stop_reason = reason
    run.finished_at = utcnow()
    run.duration_seconds = elapsed
    await db.commit()
    return run


async def mark_orphaned_runs(db: AsyncSession) -> int:
    result = await db.execute(
        update(Run)
        .where(Run.status == "running")
        .values(status="interrupted", stop_reason="server_restart", finished_at=utcnow())
    )
    await db.commit()
    return result.rowcount or 0


async def export_visitor(db: AsyncSession, visitor_id: str, created_at) -> dict:
    runs = await db.scalars(
        select(Run).where(Run.visitor_id == visitor_id).options(selectinload(Run.generations)).order_by(Run.created_at)
    )
    return {
        "exported_at": utcnow().isoformat(),
        "trial_id": visitor_id,
        "first_seen": created_at.isoformat() if created_at else None,
        "runs": [
            {
                "id": r.id,
                "name": r.name,
                "status": r.status,
                "stop_reason": r.stop_reason,
                "config": r.config,
                "best_fitness": r.best_fitness,
                "best_generation": r.best_generation,
                "generations_completed": r.generations_completed,
                "created_at": r.created_at.isoformat(),
                "finished_at": r.finished_at.isoformat() if r.finished_at else None,
                "duration_seconds": r.duration_seconds,
                "notes": r.notes,
                "champion": r.champion,
                "generations": [
                    {
                        "index": g.index,
                        "best_fitness": g.best_fitness,
                        "mean_fitness": g.mean_fitness,
                        "std_fitness": g.std_fitness,
                        "species_count": g.species_count,
                        "ticks": g.ticks,
                        "duration_ms": g.duration_ms,
                    }
                    for g in r.generations
                ],
            }
            for r in runs
        ],
    }


async def delete_visitor(db: AsyncSession, visitor_id: str) -> None:
    await db.execute(delete(Visitor).where(Visitor.id == visitor_id))
    await db.commit()
