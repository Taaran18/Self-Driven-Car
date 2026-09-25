import re
import uuid
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request, status

from app.core.config import get_settings
from app.core.errors import AppError
from app.db.session import DB

VISITOR_HEADER = "x-visitor-id"
_WORKSPACE_NAMESPACE = uuid.UUID("3b2f7c1e-5a0d-4e8b-9c61-7d4a2f9e8b13")
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


@dataclass(frozen=True)
class VisitorIdentity:
    id: str
    ip: str
    user_agent: str | None


def valid_browser_id(value: str | None) -> bool:
    return bool(value) and bool(_UUID.match(value))


def workspace_id(browser_id: str, ip: str) -> str:
    return str(uuid.uuid5(_WORKSPACE_NAMESPACE, f"{browser_id}|{ip}"))


def client_ip_from(headers, fallback: str | None) -> str:
    if get_settings().trust_proxy:
        forwarded = headers.get("x-forwarded-for", "")
        hops = [hop.strip() for hop in forwarded.split(",") if hop.strip()]
        if hops:
            return hops[-1][:64]
    return (fallback or "unknown")[:64]


def client_ip(request: Request) -> str:
    return client_ip_from(request.headers, request.client.host if request.client else None)


def missing_visitor() -> AppError:
    return AppError(
        status.HTTP_400_BAD_REQUEST,
        "missing_visitor",
        "Your browser didn't send a trial ID. Reload the page and try again.",
    )


def identify(browser_id: str | None, ip: str, user_agent: str | None) -> VisitorIdentity:
    browser_id = (browser_id or "").strip().lower()
    if not valid_browser_id(browser_id):
        raise missing_visitor()
    return VisitorIdentity(id=workspace_id(browser_id, ip), ip=ip, user_agent=(user_agent or "")[:400] or None)


async def get_visitor(request: Request, db: DB) -> VisitorIdentity:
    from app.services.usage_service import touch_visitor

    identity = identify(request.headers.get(VISITOR_HEADER), client_ip(request), request.headers.get("user-agent"))
    await touch_visitor(db, identity)
    return identity


CurrentVisitor = Annotated[VisitorIdentity, Depends(get_visitor)]
