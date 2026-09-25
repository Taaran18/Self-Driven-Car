import asyncio
import secrets
import time
from dataclasses import dataclass
from threading import Lock

from app.core.visitor import VisitorIdentity
from app.simulation.session import SimulationSession


@dataclass
class Ticket:
    visitor: VisitorIdentity
    expires_at: float


class TicketStore:
    def __init__(self, ttl_seconds: int):
        self.ttl = ttl_seconds
        self._tickets: dict[str, Ticket] = {}
        self._lock = Lock()

    def issue(self, visitor: VisitorIdentity) -> str:
        token = secrets.token_urlsafe(24)
        now = time.monotonic()
        with self._lock:
            self._tickets = {k: v for k, v in self._tickets.items() if v.expires_at > now}
            self._tickets[token] = Ticket(visitor, now + self.ttl)
        return token

    def redeem(self, token: str | None) -> Ticket | None:
        if not token:
            return None
        with self._lock:
            ticket = self._tickets.pop(token, None)
        if ticket is None or ticket.expires_at < time.monotonic():
            return None
        return ticket


class CapacityError(Exception):
    pass


@dataclass
class ActiveSimulation:
    connection_id: str
    visitor_id: str
    session: SimulationSession
    run_id: str | None = None


class SimulationManager:
    def __init__(self, max_concurrent: int):
        self.max_concurrent = max_concurrent
        self._active: dict[str, ActiveSimulation] = {}
        self._lock = asyncio.Lock()

    @property
    def active_count(self) -> int:
        return sum(1 for a in self._active.values() if a.session.alive)

    async def claim(
        self, owner_key: str, connection_id: str, visitor_id: str, session: SimulationSession
    ) -> ActiveSimulation:
        async with self._lock:
            self._active = {k: v for k, v in self._active.items() if v.session.alive or k == owner_key}
            previous = self._active.get(owner_key)
            others = sum(1 for k, v in self._active.items() if k != owner_key and v.session.alive)
            if others >= self.max_concurrent:
                raise CapacityError
            if previous and previous.connection_id != connection_id and previous.session.alive:
                previous.session.stop("replaced")
            active = ActiveSimulation(connection_id, visitor_id, session)
            self._active[owner_key] = active
            return active

    def release(self, owner_key: str, connection_id: str) -> None:
        active = self._active.get(owner_key)
        if active and active.connection_id == connection_id:
            del self._active[owner_key]

    def run_ids_for(self, visitor_id: str) -> set[str]:
        return {a.run_id for a in self._active.values() if a.visitor_id == visitor_id and a.run_id and a.session.alive}

    def stop_all_for(self, visitor_id: str) -> list[SimulationSession]:
        sessions = [a.session for a in self._active.values() if a.visitor_id == visitor_id]
        for session in sessions:
            session.stop("data_deleted")
        return sessions

    def stop_all(self, reason: str) -> list[SimulationSession]:
        sessions = [a.session for a in self._active.values()]
        for session in sessions:
            session.stop(reason)
        self._active.clear()
        return sessions
