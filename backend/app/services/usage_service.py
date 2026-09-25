import asyncio
from datetime import datetime, timedelta

from fastapi import status
from sqlalchemy import case, delete, func, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.errors import AppError
from app.core.visitor import VisitorIdentity
from app.db.models import Run, UsageEvent, Visitor, utcnow

_TOUCH_INTERVAL = timedelta(minutes=5)
_consume_lock = asyncio.Lock()


def day_start(now: datetime) -> datetime:
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def week_start(now: datetime) -> datetime:
    return day_start(now) - timedelta(days=now.weekday())


async def touch_visitor(db: AsyncSession, identity: VisitorIdentity) -> Visitor:
    now = utcnow()
    visitor = await db.get(Visitor, identity.id)
    if visitor is None:
        await db.execute(
            sqlite_insert(Visitor)
            .values(
                id=identity.id, created_at=now, last_seen_at=now, last_ip=identity.ip, user_agent=identity.user_agent
            )
            .on_conflict_do_nothing(index_elements=[Visitor.id])
        )
        await db.commit()
        visitor = await db.get(Visitor, identity.id)
    elif now - visitor.last_seen_at > _TOUCH_INTERVAL or visitor.last_ip != identity.ip:
        visitor.last_seen_at = now
        visitor.last_ip = identity.ip
        visitor.user_agent = identity.user_agent or visitor.user_agent
        await db.commit()
    return visitor


async def _count(db: AsyncSession, ip: str, since: datetime) -> int:
    return (
        await db.scalar(
            select(func.count(UsageEvent.id)).where(UsageEvent.ip_address == ip, UsageEvent.created_at >= since)
        )
        or 0
    )


async def summary(db: AsyncSession, ip: str) -> dict:
    settings = get_settings()
    now = utcnow()
    today, week = day_start(now), week_start(now)
    used_day = await _count(db, ip, today)
    used_week = await _count(db, ip, week)
    left_day = max(0, settings.trial_runs_per_day - used_day)
    left_week = max(0, settings.trial_runs_per_week - used_week)
    return {
        "runs_per_day": settings.trial_runs_per_day,
        "runs_per_week": settings.trial_runs_per_week,
        "used_today": used_day,
        "used_this_week": used_week,
        "left_today": min(left_day, left_week),
        "left_this_week": left_week,
        "day_resets_at": (today + timedelta(days=1)).isoformat(),
        "week_resets_at": (week + timedelta(days=7)).isoformat(),
    }


def limit_error(usage: dict) -> AppError:
    if usage["left_this_week"] <= 0:
        message = (
            f"You've used this week's {usage['runs_per_week']} trial runs. "
            "Your weekly limit resets on Monday at midnight UTC."
        )
    else:
        message = f"You've used today's {usage['runs_per_day']} trial runs. Your limit resets at midnight UTC."
    return AppError(status.HTTP_429_TOO_MANY_REQUESTS, "trial_limit", message, {"usage": usage})


async def ensure_available(db: AsyncSession, ip: str) -> dict:
    usage = await summary(db, ip)
    if usage["left_today"] <= 0:
        raise limit_error(usage)
    return usage


async def consume(db: AsyncSession, identity: VisitorIdentity, run_id: str | None) -> tuple[int, dict]:
    async with _consume_lock:
        await ensure_available(db, identity.ip)
        event = UsageEvent(
            visitor_id=identity.id, ip_address=identity.ip, user_agent=identity.user_agent, run_id=run_id
        )
        db.add(event)
        await db.commit()
        return event.id, await summary(db, identity.ip)


async def refund(db: AsyncSession, event_id: int) -> None:
    await db.execute(delete(UsageEvent).where(UsageEvent.id == event_id))
    await db.commit()


async def purge_old_usage(db: AsyncSession) -> int:
    cutoff = utcnow() - timedelta(days=get_settings().usage_retention_days)
    result = await db.execute(delete(UsageEvent).where(UsageEvent.created_at < cutoff))
    has_runs = select(Run.id).where(Run.visitor_id == Visitor.id).exists()
    await db.execute(delete(Visitor).where(Visitor.last_seen_at < cutoff, ~has_runs))
    await db.execute(update(Visitor).where(Visitor.last_seen_at < cutoff).values(last_ip=None, user_agent=None))
    await db.commit()
    return result.rowcount or 0


async def admin_report(db: AsyncSession, days: int) -> dict:
    now = utcnow()
    since = now - timedelta(days=days)
    today, week = day_start(now), week_start(now)
    runs_today = func.sum(case((UsageEvent.created_at >= today, 1), else_=0))
    runs_week = func.sum(case((UsageEvent.created_at >= week, 1), else_=0))

    ip_rows = (
        await db.execute(
            select(
                UsageEvent.ip_address,
                func.count(UsageEvent.id),
                runs_today,
                runs_week,
                func.count(func.distinct(UsageEvent.visitor_id)),
                func.min(UsageEvent.created_at),
                func.max(UsageEvent.created_at),
            )
            .where(UsageEvent.created_at >= since)
            .group_by(UsageEvent.ip_address)
            .order_by(func.count(UsageEvent.id).desc())
            .limit(500)
        )
    ).all()

    visitor_rows = (
        await db.execute(
            select(
                UsageEvent.visitor_id,
                func.count(UsageEvent.id),
                runs_today,
                runs_week,
                func.max(UsageEvent.created_at),
                func.max(UsageEvent.ip_address),
                func.max(UsageEvent.user_agent),
            )
            .where(UsageEvent.created_at >= since)
            .group_by(UsageEvent.visitor_id)
            .order_by(func.max(UsageEvent.created_at).desc())
            .limit(500)
        )
    ).all()

    totals = (
        await db.execute(
            select(
                func.count(UsageEvent.id),
                func.count(func.distinct(UsageEvent.visitor_id)),
                func.count(func.distinct(UsageEvent.ip_address)),
            ).where(UsageEvent.created_at >= since)
        )
    ).one()
    runs_saved = await db.scalar(select(func.count(Run.id))) or 0
    visitors_total = await db.scalar(select(func.count(Visitor.id))) or 0

    return {
        "generated_at": now.isoformat(),
        "window_days": days,
        "limits": {
            "runs_per_day": get_settings().trial_runs_per_day,
            "runs_per_week": get_settings().trial_runs_per_week,
        },
        "totals": {
            "runs": totals[0],
            "visitors": totals[1],
            "ips": totals[2],
            "runs_saved": runs_saved,
            "visitors_all_time": visitors_total,
        },
        "ips": [
            {
                "ip": r[0],
                "runs": r[1],
                "runs_today": int(r[2] or 0),
                "runs_this_week": int(r[3] or 0),
                "visitors": r[4],
                "first_run_at": r[5].isoformat() if r[5] else None,
                "last_run_at": r[6].isoformat() if r[6] else None,
            }
            for r in ip_rows
        ],
        "visitors": [
            {
                "visitor_id": r[0],
                "runs": r[1],
                "runs_today": int(r[2] or 0),
                "runs_this_week": int(r[3] or 0),
                "last_run_at": r[4].isoformat() if r[4] else None,
                "ip": r[5],
                "user_agent": r[6],
            }
            for r in visitor_rows
        ],
    }
